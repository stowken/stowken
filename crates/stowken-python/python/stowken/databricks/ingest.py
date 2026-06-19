"""
Ingestion entry points for Stowken-on-Databricks.

v1 input shape: a source Delta table with a column mapping that identifies
which column holds the conversation payload and which hold optional metadata
(application, model, user_id, timestamp, etc.).

Other shapes (JSONL in a Volume, Lakeflow Connect source) are intentionally
deferred until v1 is shipping.

NOTE: this module deliberately does NOT use ``from __future__ import
annotations``. PySpark's pandas-UDF type-hint inference (``infer_eval_type``)
reads function annotations via ``inspect.signature`` and requires real class
objects, not strings. With future annotations enabled, ``pd.Series`` becomes
``"pd.Series"`` and PySpark raises ``UNSUPPORTED_SIGNATURE``. ``Optional[X]``
syntax is therefore used in lieu of ``X | None`` to retain compatibility
with Python 3.8+.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .backend import DatabricksBackend


@dataclass
class SourceColumnMapping:
    """
    Maps columns in the source Delta table to Stowken's manifest schema.
    ``conversation`` is required; the rest are optional and fall back to NULL
    when unmapped.

    The ``conversation`` column should be a Spark struct (or string-typed JSON
    that the caller has already parsed) with at minimum a ``messages`` array
    of ``{role, content}`` structs. A ``model`` field on the struct is
    encouraged; absent that, ``default_model`` is applied.
    """

    conversation: str
    """Column holding the conversation payload."""

    conversation_id: Optional[str] = None
    application: Optional[str] = None
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    user_id: Optional[str] = None
    ts: Optional[str] = None
    input_tokens: Optional[str] = None
    output_tokens: Optional[str] = None
    cost_usd: Optional[str] = None
    latency_ms: Optional[str] = None
    tags: Optional[str] = None
    extra_metadata: List[str] = field(default_factory=list)
    """Columns to JSON-serialize into the ``raw_metadata`` field."""


@dataclass
class IngestResult:
    rows_read: int
    conversations_written: int
    segments_new: int
    segments_deduped: int
    bytes_raw: int
    bytes_stored: int

    @property
    def dedup_ratio(self) -> float:
        return self.bytes_raw / max(self.bytes_stored, 1)


def _make_chunk_udf(default_model: str, default_tokenizer: str):
    """
    Build a pandas UDF that calls ``Stowken.chunk_only`` on every row.

    Returns a struct column shaped like
    ``{id, application, model, tokenizer, total_tokens, metadata_json,
       segment_hashes[], segment_types[], segments: [{hash, segment_type,
       compressed_data, token_count, raw_size, position}]}``.
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    segment_struct = StructType(
        [
            StructField("hash", StringType(), False),
            StructField("segment_type", StringType(), False),
            StructField("compressed_data", BinaryType(), False),
            StructField("token_count", IntegerType(), False),
            StructField("raw_size", IntegerType(), False),
            StructField("position", IntegerType(), False),
        ]
    )
    return_struct = StructType(
        [
            StructField("conversation_id", StringType(), False),
            StructField("application", StringType(), True),
            StructField("model", StringType(), False),
            StructField("tokenizer", StringType(), False),
            StructField("total_tokens", LongType(), False),
            StructField("metadata_json", StringType(), True),
            StructField("segment_hashes", ArrayType(StringType()), False),
            StructField("segment_types", ArrayType(StringType()), False),
            StructField("segments", ArrayType(segment_struct), False),
        ]
    )

    @pandas_udf(returnType=return_struct)
    def chunk_udf(
        conv_series: pd.Series,
        tok_series: pd.Series,
    ) -> pd.DataFrame:
        # For struct return types, Databricks runtime requires the UDF to
        # produce a ``pd.DataFrame`` whose columns match the StructType's
        # fields (returning a ``pd.Series`` of dicts is accepted by signature
        # inference but rejected at execution time).
        #
        # Lazy-init a per-worker vault. The default MemoryBackend is fine —
        # we never persist through it; chunk_only is pure compute.
        import json as _json
        import stowken

        global _STOWKEN_WORKER_VAULT  # type: ignore[name-defined]
        try:
            vault = _STOWKEN_WORKER_VAULT  # type: ignore[name-defined]
        except NameError:
            vault = stowken.Stowken()
            globals()["_STOWKEN_WORKER_VAULT"] = vault

        def _normalize_conv(payload):
            # Spark Connect + pandas UDF sometimes serializes nested structs
            # as JSON strings. Accept the conversation payload in any of:
            #   - dict / Row with proper nested structure
            #   - dict where ``messages`` field is a JSON string
            #   - JSON string of the whole conversation
            # Normalize to a dict with ``messages`` as a list of dicts.
            if payload is None:
                return {}
            if isinstance(payload, str):
                payload = _json.loads(payload)
            conv = dict(payload) if not isinstance(payload, dict) else payload
            messages = conv.get("messages")
            if isinstance(messages, str):
                messages = _json.loads(messages)
            if messages is not None:
                conv["messages"] = [
                    _json.loads(m)
                    if isinstance(m, str)
                    else (dict(m) if not isinstance(m, dict) else m)
                    for m in messages
                ]
            return conv

        rows = []
        for i in range(len(conv_series)):
            payload = conv_series.iloc[i]
            tok = tok_series.iloc[i] if tok_series is not None else None
            conv = _normalize_conv(payload)
            conv.setdefault("model", default_model)
            conv["tokenizer"] = tok or conv.get("tokenizer") or default_tokenizer
            result = vault.chunk_only(conv)
            rows.append(
                {
                    "conversation_id": result["id"],
                    "application": result.get("application"),
                    "model": result["model"],
                    "tokenizer": result["tokenizer"],
                    "total_tokens": result["total_tokens"],
                    "metadata_json": result.get("metadata_json"),
                    "segment_hashes": [s["hash"] for s in result["segments"]],
                    "segment_types": [s["segment_type"] for s in result["segments"]],
                    "segments": result["segments"],
                }
            )
        return pd.DataFrame(rows)

    return chunk_udf


def ingest_from_delta_table(
    backend: "DatabricksBackend",
    source_table: str,
    mapping: "SourceColumnMapping",
    *,
    default_model: str = "unknown",
    default_tokenizer: str = "cl100k_base",
    batch_size: int = 10_000,
) -> "IngestResult":
    """
    Read conversations from ``source_table``, chunk + compress via Stowken,
    and write the segments + conversations Delta tables.

    Pure batch for v1. Structured-streaming ingestion is a v2 concern.

    ``batch_size`` is currently advisory — Spark partitioning controls real
    batching. Reserved so the v2 streaming path keeps API compat.
    """
    from pyspark.sql.functions import (
        col,
        count as count_,
        current_timestamp,
        explode,
        first,
        lit,
        sum as sum_,
    )

    spark = backend.spark
    src = spark.read.table(source_table)
    rows_read = src.count()

    if rows_read == 0:
        return IngestResult(0, 0, 0, 0, 0, 0)

    tokenizer_col = (
        col(mapping.tokenizer) if mapping.tokenizer else lit(None).cast("string")
    )
    chunk_udf = _make_chunk_udf(default_model, default_tokenizer)
    chunked = src.withColumn(
        "_chunked", chunk_udf(col(mapping.conversation), tokenizer_col)
    )
    # Materialize ``chunked`` ONCE to a temp Delta table, then read it back.
    # The ``chunk_udf`` (tokenize → SHA-256 → zstd per row) is the expensive
    # step, and ``chunked`` is consumed by four separate actions below (conv
    # write, seg MERGE, bytes_raw, bytes_stored). Serverless rejects
    # ``df.cache()``, so without this the UDF re-runs ~4× over every row
    # (measured: ~50 min for 100k rows). Staging it makes the UDF run exactly
    # once; the four downstream actions become cheap Delta scans. The staging
    # table is dropped before we return (overwrite mode also self-cleans on a
    # retry if a prior run died mid-flight).
    staging_table = f"{backend.segments_table}__chunk_staging"
    (
        chunked.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(staging_table)
    )
    chunked = backend.spark.table(staging_table)

    # ── Compose the conversations rows ────────────────────────────────────
    #
    # Every column is explicitly cast to its target type in the
    # ``conversations`` table. Delta's append-mode write does not auto-cast
    # type mismatches between source and target — even when both are
    # numeric (e.g. DOUBLE vs DECIMAL(12,6)). Casting up front decouples the
    # caller's source schema from this library's storage schema.
    def _opt(c, default_expr, cast_type):
        return (col(c) if c else default_expr).cast(cast_type)

    conv_rows = chunked.select(
        col("_chunked.conversation_id").cast("string").alias("conversation_id"),
        col("_chunked.segment_hashes").alias("segment_hashes"),
        col("_chunked.segment_types").alias("segment_types"),
        col("_chunked.application").cast("string").alias("application"),
        col("_chunked.model").cast("string").alias("model"),
        col("_chunked.tokenizer").cast("string").alias("tokenizer"),
        _opt(mapping.user_id, lit(None), "string").alias("user_id"),
        _opt(mapping.ts, current_timestamp(), "timestamp").alias("ts"),
        _opt(mapping.input_tokens, lit(None), "int").alias("input_tokens"),
        _opt(mapping.output_tokens, lit(None), "int").alias("output_tokens"),
        col("_chunked.total_tokens").cast("int").alias("total_tokens"),
        _opt(mapping.cost_usd, lit(None), "decimal(12,6)").alias("cost_usd"),
        _opt(mapping.latency_ms, lit(None), "int").alias("latency_ms"),
        _opt(mapping.tags, lit(None), "map<string,string>").alias("tags"),
        col("_chunked.metadata_json").cast("string").alias("raw_metadata"),
        lit(None).cast("string").alias("summary_text"),
        lit(None).cast("array<float>").alias("summary_embedding"),
        lit(None).cast("string").alias("summary_model"),
        lit(None).cast("timestamp").alias("summary_generated_at"),
    )

    # ── Compose the segments rows (deduplicated within this batch) ────────
    seg_rows = chunked.select(explode(col("_chunked.segments")).alias("seg")).select(
        col("seg.hash").alias("segment_hash"),
        col("seg.compressed_data").alias("content_zstd"),
        col("seg.token_count").alias("token_count"),
        col("seg.segment_type").alias("segment_type"),
    )
    seg_dedup = seg_rows.groupBy("segment_hash").agg(
        first("content_zstd").alias("content_zstd"),
        first("token_count").alias("token_count"),
        first("segment_type").alias("segment_type"),
        count_("*").alias("batch_refs"),
    )

    # ── Stats that need to be computed before the MERGE consumes the data ─
    bytes_raw_row = chunked.agg(
        sum_(col("_chunked.total_tokens") * lit(4)).alias("bytes_raw")
    ).first()
    bytes_raw = int(bytes_raw_row["bytes_raw"] or 0)

    # bytes_stored = compressed bytes of segments that DIDN'T already exist.
    existing = backend.spark.table(backend.segments_table).select("segment_hash")
    new_in_batch = seg_dedup.join(existing, on="segment_hash", how="left_anti")
    bytes_stored_row = new_in_batch.selectExpr(
        "coalesce(sum(LENGTH(content_zstd)), 0) AS bytes_stored"
    ).first()
    bytes_stored = int(bytes_stored_row["bytes_stored"] or 0)

    # ── Persist ───────────────────────────────────────────────────────────
    merge_metrics = backend.put_segments_batch(seg_dedup)
    backend.put_conversations_batch(conv_rows)

    # Drop the staging table now that every downstream action has consumed it.
    backend.spark.sql(f"DROP TABLE IF EXISTS {staging_table}")

    return IngestResult(
        rows_read=rows_read,
        conversations_written=rows_read,
        segments_new=merge_metrics["inserted"],
        segments_deduped=merge_metrics["updated"],
        bytes_raw=bytes_raw,
        bytes_stored=bytes_stored,
    )
