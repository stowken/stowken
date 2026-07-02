"""
Common analytics queries against the Stowken-on-Databricks data model.

Most queries run against the ``conversations`` table directly. Two of them
(``dedup_ratio``, ``top_system_prompts``) read ``segments``. Joins through
``segments`` to fetch compressed bytes are reserved for rehydration paths —
analytics doesn't pay the cost of decompression.

Queries return ``DataFrame`` so the caller can chain ``.display()``,
``.toPandas()``, or persist to a results table.

Enrichment: dashboard widgets that need human-readable text from segments
(the obvious one is ``top_system_prompts``) rely on a side table
``<catalog>.<schema>.segments_decoded``. Populate it with
``materialize_decoded_segments()`` after each ingest. ``top_system_prompts``
JOINs against it when present and gracefully falls back to hash-only output
when it doesn't exist (e.g. before the first enrichment run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from .backend import DatabricksBackend


def _decoded_table_name(backend: "DatabricksBackend") -> str:
    return f"{backend.config.catalog}.{backend.config.schema}.segments_decoded"


def _decoded_table_exists(backend: "DatabricksBackend") -> bool:
    try:
        return backend.spark.catalog.tableExists(_decoded_table_name(backend))
    except Exception:
        return False


def dedup_ratio(backend: "DatabricksBackend") -> "DataFrame":
    """
    Total raw bytes that would have been stored without dedup + compression,
    divided by total bytes actually on disk. The headline metric for
    proving the accelerator's value.

    Raw bytes = ``token_count * 4 * ref_count`` summed across unique
    segments. Stored bytes = compressed payload length summed across
    unique segments.
    """
    return backend.spark.sql(
        f"""
        SELECT
          BIGINT(SUM(BIGINT(token_count) * 4 * ref_count))                    AS bytes_raw,
          BIGINT(SUM(LENGTH(content_zstd)))                                   AS bytes_stored,
          CASE
            WHEN SUM(LENGTH(content_zstd)) = 0 THEN 0.0
            ELSE 1.0 * SUM(BIGINT(token_count) * 4 * ref_count) / SUM(LENGTH(content_zstd))
          END AS dedup_ratio,
          CASE
            WHEN SUM(BIGINT(token_count) * 4 * ref_count) = 0 THEN 0.0
            ELSE 100.0 * (1.0 - 1.0 * SUM(LENGTH(content_zstd))
                                  / SUM(BIGINT(token_count) * 4 * ref_count))
          END AS savings_percentage
        FROM {backend.segments_table}
        """
    )


def cost_by_application_and_model(
    backend: "DatabricksBackend", days: int = 30
) -> "DataFrame":
    """Token spend + cost over the last ``days`` grouped by application and model."""
    return backend.spark.sql(
        f"""
        SELECT
          application,
          model,
          COUNT(*)                                  AS conversation_count,
          BIGINT(SUM(input_tokens))                 AS input_tokens,
          BIGINT(SUM(output_tokens))                AS output_tokens,
          BIGINT(SUM(total_tokens))                 AS total_tokens,
          ROUND(SUM(cost_usd), 4)                   AS total_cost_usd,
          ROUND(AVG(cost_usd), 6)                   AS avg_cost_usd
        FROM {backend.conversations_table}
        WHERE ts >= current_timestamp() - INTERVAL {days} DAY
        GROUP BY application, model
        ORDER BY total_cost_usd DESC NULLS LAST
        """
    )


def top_system_prompts(
    backend: "DatabricksBackend", limit: int = 20
) -> "DataFrame":
    """
    Most heavily reused system prompts, ranked by ``ref_count``. Each row
    is a unique system-prompt segment plus how many conversations reference
    it.

    When ``<catalog>.<schema>.segments_decoded`` exists (populated by
    :func:`materialize_decoded_segments`), the result includes a
    ``decoded_text`` column — the actual prompt text. Without it, the
    ``decoded_text`` column is NULL.
    """
    if _decoded_table_exists(backend):
        return backend.spark.sql(
            f"""
            SELECT
              seg.segment_hash,
              dec.decoded_text,
              seg.token_count,
              seg.ref_count,
              seg.first_seen,
              seg.last_seen,
              LENGTH(seg.content_zstd) AS compressed_bytes,
              ROUND(1.0 * seg.token_count * 4 * seg.ref_count / NULLIF(LENGTH(seg.content_zstd), 0), 1)
                AS per_prompt_savings_ratio
            FROM {backend.segments_table} seg
            LEFT JOIN {_decoded_table_name(backend)} dec
              ON seg.segment_hash = dec.segment_hash
            WHERE seg.segment_type = 'system_prompt'
            ORDER BY seg.ref_count DESC
            LIMIT {limit}
            """
        )
    return backend.spark.sql(
        f"""
        SELECT
          segment_hash,
          CAST(NULL AS STRING) AS decoded_text,
          token_count,
          ref_count,
          first_seen,
          last_seen,
          LENGTH(content_zstd) AS compressed_bytes,
          ROUND(1.0 * token_count * 4 * ref_count / NULLIF(LENGTH(content_zstd), 0), 1)
            AS per_prompt_savings_ratio
        FROM {backend.segments_table}
        WHERE segment_type = 'system_prompt'
        ORDER BY ref_count DESC
        LIMIT {limit}
        """
    )


def materialize_decoded_segments(
    backend: "DatabricksBackend",
    *,
    segment_types: Iterable[str] = ("system_prompt",),
    top_n: int | None = 200,
    only_new: bool = True,
    tokenizer: str = "cl100k_base",
    batch_size: int = 5000,
) -> int:
    """
    Decode segment blobs and write the resulting text to
    ``<catalog>.<schema>.segments_decoded``, so dashboard queries (which can
    only call SQL) can surface human-readable prompt text.

    Defaults to decoding the top-200 most-referenced ``system_prompt``
    segments. That bounds the side table to ~100 KB of text on typical
    workloads — invisible against the main segments table — while still
    populating every row a typical "top prompts" widget would display.

    ``segment_types``: which types to decode. Each type's top-N is taken
    independently, so e.g. ``("system_prompt", "user_turn")`` with
    ``top_n=200`` writes up to 200 of each (400 rows total).

    ``top_n=None`` disables the per-type limit and decodes every matching
    segment — only recommended for small workloads or when you need every
    segment of a type decoded.

    ``only_new=True`` skips segments already in the decoded table, so this
    function is cheap to re-run after every ingest cycle.

    Returns the number of new rows written.
    """
    import stowken

    spark = backend.spark
    decoded_table = _decoded_table_name(backend)

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {decoded_table} (
          segment_hash STRING NOT NULL,
          decoded_text STRING NOT NULL,
          tokenizer    STRING NOT NULL,
          decoded_at   TIMESTAMP NOT NULL
        ) USING DELTA
        """
    )

    types_list = ", ".join(f"'{t}'" for t in segment_types)
    if top_n is None:
        ranked_sql = f"""
            SELECT segment_hash, content_zstd, canonical_hash
            FROM {backend.segments_table}
            WHERE segment_type IN ({types_list})
        """
    else:
        ranked_sql = f"""
            SELECT segment_hash, content_zstd, canonical_hash FROM (
              SELECT segment_hash, content_zstd, canonical_hash,
                     ROW_NUMBER() OVER (PARTITION BY segment_type ORDER BY ref_count DESC) AS rn
              FROM {backend.segments_table}
              WHERE segment_type IN ({types_list})
            ) WHERE rn <= {int(top_n)}
        """

    if only_new:
        pending_sql = f"""
            WITH candidates AS ({ranked_sql})
            SELECT c.segment_hash, c.content_zstd, c.canonical_hash
            FROM candidates c
            LEFT ANTI JOIN {decoded_table} d ON c.segment_hash = d.segment_hash
        """
    else:
        pending_sql = ranked_sql

    # Resolve each segment's canonical bytes so 0x04 (near-dedup delta) frames
    # can be decoded; self-contained frames get NULL here and ignore it.
    resolved_sql = f"""
        SELECT p.segment_hash, p.content_zstd, c.content_zstd AS canon_zstd
        FROM ({pending_sql}) p
        LEFT JOIN {backend.segments_table} c ON p.canonical_hash = c.segment_hash
    """
    pending_rows = spark.sql(resolved_sql).collect()
    if not pending_rows:
        return 0

    from datetime import datetime, timezone
    from pyspark.sql.types import (
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    out_schema = StructType(
        [
            StructField("segment_hash", StringType(), False),
            StructField("decoded_text", StringType(), False),
            StructField("tokenizer",    StringType(), False),
            StructField("decoded_at",   TimestampType(), False),
        ]
    )

    written = 0
    for i in range(0, len(pending_rows), batch_size):
        batch = pending_rows[i : i + batch_size]
        now = datetime.now(timezone.utc)
        out_rows = []
        for row in batch:
            blob = row.content_zstd
            if blob is None:
                continue
            canon = row.canon_zstd
            try:
                text = stowken.decode_segment_text(
                    bytes(blob),
                    tokenizer,
                    canonical_zstd=bytes(canon) if canon is not None else None,
                )
            except Exception:
                # Skip unreadable blobs rather than failing the whole batch
                continue
            out_rows.append(
                {
                    "segment_hash": row.segment_hash,
                    "decoded_text": text,
                    "tokenizer":    tokenizer,
                    "decoded_at":   now,
                }
            )
        if not out_rows:
            continue
        out_df = spark.createDataFrame(out_rows, schema=out_schema)
        out_df.createOrReplaceTempView("_stowken_decoded_stage")
        spark.sql(
            f"""
            MERGE INTO {decoded_table} t
            USING _stowken_decoded_stage s
            ON t.segment_hash = s.segment_hash
            WHEN MATCHED THEN UPDATE SET
              t.decoded_text = s.decoded_text,
              t.tokenizer    = s.tokenizer,
              t.decoded_at   = s.decoded_at
            WHEN NOT MATCHED THEN INSERT *
            """
        )
        written += len(out_rows)
    return written


def token_spend_over_time(
    backend: "DatabricksBackend", granularity: str = "day"
) -> "DataFrame":
    """Token volume + cost over time at day or hour granularity."""
    trunc = {"day": "DAY", "hour": "HOUR"}.get(granularity.lower())
    if trunc is None:
        raise ValueError(f"granularity must be 'day' or 'hour', got {granularity!r}")
    return backend.spark.sql(
        f"""
        SELECT
          DATE_TRUNC('{trunc}', ts)         AS bucket,
          COUNT(*)                          AS conversation_count,
          BIGINT(SUM(input_tokens))         AS input_tokens,
          BIGINT(SUM(output_tokens))        AS output_tokens,
          BIGINT(SUM(total_tokens))         AS total_tokens,
          ROUND(SUM(cost_usd), 4)           AS cost_usd
        FROM {backend.conversations_table}
        GROUP BY bucket
        ORDER BY bucket
        """
    )


def conversation_length_distribution(
    backend: "DatabricksBackend",
) -> "DataFrame":
    """
    p50 / p95 / p99 of ``total_tokens`` per conversation, grouped by
    application and model. Useful for capacity planning and surfacing
    unusually long sessions.
    """
    return backend.spark.sql(
        f"""
        SELECT
          application,
          model,
          COUNT(*)                                                AS conversation_count,
          ROUND(AVG(total_tokens), 0)                             AS mean_tokens,
          APPROX_PERCENTILE(total_tokens, 0.5)                    AS p50_tokens,
          APPROX_PERCENTILE(total_tokens, 0.95)                   AS p95_tokens,
          APPROX_PERCENTILE(total_tokens, 0.99)                   AS p99_tokens,
          MAX(total_tokens)                                       AS max_tokens
        FROM {backend.conversations_table}
        GROUP BY application, model
        ORDER BY conversation_count DESC
        """
    )


def find_outlier_conversations(
    backend: "DatabricksBackend", z_threshold: float = 3.0
) -> "DataFrame":
    """
    Conversations whose ``total_tokens`` is more than ``z_threshold`` standard
    deviations above the mean for their ``(application, model)`` group. Mirrors
    Stowken's ``find_outliers`` semantic-search helper, but operates on
    token-count outliers rather than embedding outliers.

    Includes ``summary_text`` when populated (run ``summarize_conversations``
    first) so dashboard viewers can see what the outlier conversation was
    about without having to click through.
    """
    return backend.spark.sql(
        f"""
        WITH stats AS (
          SELECT
            application,
            model,
            AVG(total_tokens)    AS mean_tokens,
            STDDEV(total_tokens) AS std_tokens
          FROM {backend.conversations_table}
          GROUP BY application, model
        )
        SELECT
          c.conversation_id,
          c.application,
          c.model,
          c.total_tokens,
          c.ts,
          c.summary_text,
          (c.total_tokens - s.mean_tokens) / NULLIF(s.std_tokens, 0) AS z_score
        FROM {backend.conversations_table} c
        JOIN stats s
          ON c.application <=> s.application
         AND c.model       <=> s.model
        WHERE (c.total_tokens - s.mean_tokens) / NULLIF(s.std_tokens, 0) > {z_threshold}
        ORDER BY z_score DESC
        """
    )
