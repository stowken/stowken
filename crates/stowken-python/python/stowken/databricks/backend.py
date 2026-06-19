"""
``DatabricksBackend`` — Stowken storage backed by Unity Catalog Delta tables.

Conceptually mirrors the Rust ``StorageBackend`` trait, but writes through
PySpark + delta-spark instead of holding bytes locally. Chunking, hashing, and
varint+zstd compression are still performed by the Rust core (via the
existing ``stowken`` Python bindings); only the persistence layer differs.

Write path::

    raw conversation  →  Stowken.chunk_only  →  (manifest, segments[])
                                                        │
                                            ┌───────────┴────────────┐
                                            ▼                        ▼
                              MERGE INTO segments           INSERT INTO
                              ON segment_hash =             conversations
                              source.segment_hash           (segment_hashes, …)
                              WHEN MATCHED INCREMENT ref
                              WHEN NOT MATCHED INSERT

Read path (rehydration only — analytics never needs this)::

    conversations row  →  segment_hashes array  →  JOIN segments
                                                  →  Stowken.decompress  →  tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for a ``DatabricksBackend`` instance."""

    catalog: str
    schema: str
    # If True, run ``OPTIMIZE`` after each ingestion batch. Defaults to False
    # because Databricks runs predictive optimization on managed UC tables.
    optimize_after_write: bool = False


class DatabricksBackend:
    """
    Stowken storage backed by ``{catalog}.{schema}.segments`` +
    ``{catalog}.{schema}.conversations`` Delta tables.

    Construct on the driver of a Databricks cluster — the active
    ``SparkSession`` is required.
    """

    def __init__(self, spark: "SparkSession", config: BackendConfig) -> None:
        self._spark = spark
        self._config = config
        self._segments_table = f"{config.catalog}.{config.schema}.segments"
        self._conversations_table = f"{config.catalog}.{config.schema}.conversations"

    @property
    def spark(self) -> "SparkSession":
        return self._spark

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def segments_table(self) -> str:
        return self._segments_table

    @property
    def conversations_table(self) -> str:
        return self._conversations_table

    # ── Segment operations ────────────────────────────────────────────────

    def put_segments_batch(self, segments: "DataFrame") -> dict[str, int]:
        """
        MERGE a batch of unique segments into the segments table.

        Expects columns: ``segment_hash`` STRING, ``content_zstd`` BINARY,
        ``token_count`` INT, ``segment_type`` STRING, ``batch_refs`` BIGINT.

        Increments ``ref_count`` on matches; inserts new rows otherwise.
        Returns a dict with ``inserted`` and ``updated`` counts from the MERGE
        operation metrics so the caller can populate ``IngestResult``.
        """
        stage = "_stowken_segments_stage"
        segments.createOrReplaceTempView(stage)
        self._spark.sql(
            f"""
            MERGE INTO {self._segments_table} t
            USING {stage} s
            ON t.segment_hash = s.segment_hash
            WHEN MATCHED THEN UPDATE SET
                t.ref_count  = t.ref_count + s.batch_refs,
                t.last_seen  = current_timestamp()
            WHEN NOT MATCHED THEN INSERT (
                segment_hash, content_zstd, token_count, segment_type,
                ref_count, first_seen, last_seen
            ) VALUES (
                s.segment_hash, s.content_zstd, s.token_count, s.segment_type,
                s.batch_refs, current_timestamp(), current_timestamp()
            )
            """
        )
        # Pull insert/update counts from Delta history for this operation.
        last = self._spark.sql(
            f"DESCRIBE HISTORY {self._segments_table} LIMIT 1"
        ).first()
        metrics = (last["operationMetrics"] or {}) if last else {}
        return {
            "inserted": int(metrics.get("numTargetRowsInserted", 0)),
            "updated": int(metrics.get("numTargetRowsUpdated", 0)),
        }

    def put_segment_deltas(self, deltas: "DataFrame") -> int:
        """
        Re-encode existing segments as near-dedup deltas (the near-dedup pass).

        Expects columns: ``segment_hash`` STRING, ``content_zstd`` BINARY (a 0x04
        delta frame), ``canonical_hash`` STRING, ``canonical_similarity`` FLOAT.
        MATCHED-only MERGE — updates those columns in place; never inserts.
        Returns the number of rows rewritten.
        """
        if deltas.isEmpty():
            return 0
        stage = "_stowken_deltas_stage"
        deltas.createOrReplaceTempView(stage)
        self._spark.sql(
            f"""
            MERGE INTO {self._segments_table} t
            USING {stage} s
            ON t.segment_hash = s.segment_hash
            WHEN MATCHED THEN UPDATE SET
                t.content_zstd          = s.content_zstd,
                t.canonical_hash        = s.canonical_hash,
                t.canonical_similarity  = s.canonical_similarity,
                t.last_seen             = current_timestamp()
            """
        )
        last = self._spark.sql(
            f"DESCRIBE HISTORY {self._segments_table} LIMIT 1"
        ).first()
        metrics = (last["operationMetrics"] or {}) if last else {}
        return int(metrics.get("numTargetRowsUpdated", 0))

    def get_segment_blobs(self, hashes: Iterable[str]) -> "DataFrame":
        """Return segment rows for the given hashes (any order)."""
        hashes_list = list(hashes)
        if not hashes_list:
            return self._spark.createDataFrame(
                [], schema="segment_hash STRING"
            ).join(self._spark.table(self._segments_table), on="segment_hash")
        df = self._spark.createDataFrame(
            [(h,) for h in hashes_list], schema="segment_hash STRING"
        )
        return self._spark.table(self._segments_table).join(df, on="segment_hash", how="inner")

    def get_segment_blobs_with_canonical(self, hashes: Iterable[str]) -> "DataFrame":
        """
        Like :meth:`get_segment_blobs`, but also resolves each segment's
        canonical bytes for delta decoding. Adds a ``canon_zstd`` column holding
        the canonical segment's ``content_zstd`` (NULL when ``canonical_hash`` is
        NULL, i.e. the segment is self-contained). Pass both to
        ``stowken.decode_segment_text(content_zstd, tokenizer, canon_zstd)``.
        """
        from pyspark.sql.functions import col

        base = self.get_segment_blobs(hashes)
        canon = self._spark.table(self._segments_table).select(
            col("segment_hash").alias("_canon_hash"),
            col("content_zstd").alias("canon_zstd"),
        )
        return base.join(
            canon, base["canonical_hash"] == canon["_canon_hash"], how="left"
        ).drop("_canon_hash")

    # ── Conversation operations ───────────────────────────────────────────

    def put_conversation(self, manifest: dict[str, Any]) -> None:
        """Append a single conversation manifest row."""
        df = self._spark.createDataFrame([manifest])
        self.put_conversations_batch(df)

    def put_conversations_batch(self, manifests: "DataFrame") -> None:
        """Append a batch of conversation manifest rows."""
        (
            manifests.write.format("delta")
            .mode("append")
            .saveAsTable(self._conversations_table)
        )
        if self._config.optimize_after_write:
            self._spark.sql(f"OPTIMIZE {self._conversations_table}")

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Read a single conversation manifest by id."""
        from pyspark.sql.functions import col

        rows = (
            self._spark.table(self._conversations_table)
            .filter(col("conversation_id") == conversation_id)
            .limit(1)
            .collect()
        )
        if not rows:
            raise KeyError(conversation_id)
        return rows[0].asDict(recursive=True)

    # ── Maintenance ───────────────────────────────────────────────────────

    def garbage_collect(self) -> int:
        """
        Delete segments whose ``ref_count`` is 0. Returns the number of rows
        removed. Invoke after bulk conversation deletes.
        """
        before = (
            self._spark.table(self._segments_table)
            .filter("ref_count = 0")
            .count()
        )
        self._spark.sql(f"DELETE FROM {self._segments_table} WHERE ref_count = 0")
        return int(before)
