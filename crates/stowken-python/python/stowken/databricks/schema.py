"""
Delta-table DDL for the Stowken-on-Databricks data model.

Two tables in Unity Catalog:

  ``segments``       — the dedup layer. One row per unique token segment,
                       keyed by SHA-256 hash of the raw token bytes. Storage
                       holds Stowken's varint+zstd-compressed segment payload
                       so the dedup + compression wins are preserved on disk.

  ``conversations``  — the manifest. One row per ingested conversation,
                       referencing an ordered list of ``segments.segment_hash``
                       values. Downstream analytics queries this table for
                       cost, usage, and traffic. Segment-content joins only
                       happen on rehydration.

Liquid clustering is preferred over partitioning so customers don't have to
tune partition cardinality. CDF is enabled on both tables so downstream
syncs (Vector Search, dashboards, exports) can incrementalize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _segments_ddl(catalog: str, schema: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.segments (
      segment_hash   STRING  NOT NULL COMMENT 'SHA-256 of raw token bytes',
      content_zstd   BINARY  NOT NULL COMMENT 'Stowken-compressed token payload; a 0x04 delta frame when canonical_hash is set',
      token_count    INT     NOT NULL,
      segment_type   STRING  NOT NULL COMMENT 'system_prompt | context | user_turn | assistant_turn | tool_call | tool_result | continuation',
      ref_count      BIGINT  NOT NULL COMMENT 'how many conversations reference this segment',
      canonical_hash STRING           COMMENT 'when set, content_zstd is a 0x04 delta against this canonical segment (near-dedup)',
      canonical_similarity FLOAT      COMMENT 'Jaccard similarity of this variant to its canonical (near-dedup); gates embedding reuse',
      first_seen     TIMESTAMP NOT NULL,
      last_seen      TIMESTAMP NOT NULL,
      CONSTRAINT segments_pk PRIMARY KEY (segment_hash) RELY
    )
    USING DELTA
    CLUSTER BY (segment_hash)
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true'
    )
    """


def _conversations_ddl(catalog: str, schema: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.conversations (
      conversation_id       STRING NOT NULL,
      segment_hashes        ARRAY<STRING> NOT NULL COMMENT 'ordered references into segments.segment_hash',
      segment_types         ARRAY<STRING> NOT NULL COMMENT 'parallel to segment_hashes',
      application           STRING,
      model                 STRING,
      tokenizer             STRING,
      user_id               STRING,
      ts                    TIMESTAMP NOT NULL,
      input_tokens          INT,
      output_tokens         INT,
      total_tokens          INT,
      cost_usd              DECIMAL(12,6),
      latency_ms            INT,
      tags                  MAP<STRING,STRING>,
      raw_metadata          STRING COMMENT 'opaque JSON for fields not promoted to columns',
      summary_text          STRING COMMENT 'natural-language summary, populated by the summarize step',
      summary_embedding     ARRAY<FLOAT> COMMENT 'L2-normalized embedding of summary_text',
      summary_model         STRING COMMENT 'which model produced the summary + embedding',
      summary_generated_at  TIMESTAMP COMMENT 'when the summary was last regenerated',
      CONSTRAINT conversations_pk PRIMARY KEY (conversation_id) RELY
    )
    USING DELTA
    CLUSTER BY (application, model, ts)
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true'
    )
    """


SEGMENTS_DDL = _segments_ddl
CONVERSATIONS_DDL = _conversations_ddl


def create_tables(spark: "SparkSession", catalog: str, schema: str) -> None:
    """
    Create the ``segments`` and ``conversations`` tables in ``catalog.schema``.

    Idempotent: uses ``CREATE TABLE IF NOT EXISTS``. Does not migrate existing
    schemas — for a schema change, evolve via ``ALTER TABLE`` in a migration
    script.

    The caller is responsible for ensuring the catalog + schema exist and that
    the executing principal has ``CREATE TABLE`` on the schema.
    """
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(_segments_ddl(catalog, schema))
    spark.sql(_conversations_ddl(catalog, schema))
    # Bring pre-existing tables up to the current schema (near-dedup columns).
    migrate_segments_add_near_dedup(spark, catalog, schema)


def migrate_segments_add_near_dedup(
    spark: "SparkSession", catalog: str, schema: str
) -> None:
    """
    Add the near-dedup columns (``canonical_hash``, ``canonical_similarity``) to
    an existing ``segments`` table. Each column is added independently if
    missing, so it's safe and idempotent on fresh tables, populated tables, and
    tables migrated by an earlier version (existing rows get NULL = "self-
    contained, not yet near-deduped").
    """
    table = f"{catalog}.{schema}.segments"
    to_add = []
    if not _has_column(spark, table, "canonical_hash"):
        to_add.append(
            "canonical_hash STRING COMMENT 'when set, content_zstd is a 0x04 "
            "delta against this canonical segment (near-dedup)'"
        )
    if not _has_column(spark, table, "canonical_similarity"):
        to_add.append(
            "canonical_similarity FLOAT COMMENT 'Jaccard similarity of this "
            "variant to its canonical (near-dedup); gates embedding reuse'"
        )
    if to_add:
        spark.sql(f"ALTER TABLE {table} ADD COLUMNS ({', '.join(to_add)})")


def _has_column(spark: "SparkSession", table: str, column: str) -> bool:
    try:
        cols = {f.name for f in spark.table(table).schema.fields}
    except Exception:
        return False
    return column in cols
