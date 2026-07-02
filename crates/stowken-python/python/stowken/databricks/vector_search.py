"""
Mosaic AI Vector Search helpers for Stowken-on-Databricks.

Two flavors of index:

**Conversation-level** (one embedding per conversation, over `summary_text`):
  * :func:`setup_index`              — create endpoint + Delta-sync index.
  * :func:`search_conversations`     — embed a query, return top-k matches.
  * :func:`ask_about_conversations`  — retrieval + Foundation Model = grounded answer.

**Segment-level** (one embedding per UNIQUE segment — dedup-aware):
  * :func:`materialize_segment_embeddings`   — embed each unique segment once.
  * :func:`setup_segment_index`              — Delta-sync index over the segment embeddings.
  * :func:`find_conversations_via_segments`  — search at segment granularity, return conversations.

The segment-level path is where Stowken's dedup specifically pays off — if
1000 conversations share a system prompt, we make ONE embedding API call
instead of 1000 for the same retrieval quality.

Requires the ``databricks-vectorsearch`` Python package (in the
``[databricks]`` extra). Imports are deferred to function bodies so this
module loads without the package installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from .backend import DatabricksBackend
    from .summarize import Embedder


@dataclass(frozen=True)
class VectorSearchConfig:
    """Connection info for a provisioned Vector Search index."""

    endpoint_name: str
    index_name: str
    embedding_dim: int = 1024


DEFAULT_VS_COLUMNS = [
    # Identifier + vector
    "conversation_id",
    "summary_embedding",
    # Surfaceable metadata (Vector-Search-supported types only)
    "summary_text",
    "summary_model",
    "summary_generated_at",
    "application",
    "model",
    "tokenizer",
    "user_id",
    "ts",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_ms",
    "segment_hashes",
    "segment_types",
    "raw_metadata",
    # NOTE: cost_usd (DECIMAL) and tags (MAP<STRING,STRING>) are deliberately
    # excluded — Vector Search rejects DECIMAL + MAP column types. Filter on
    # cost or tags via a post-search JOIN to the conversations table.
]


def setup_index(
    backend: "DatabricksBackend",
    *,
    endpoint_name: str,
    index_name: Optional[str] = None,
    embedding_dim: int = 1024,
    primary_key: str = "conversation_id",
    embedding_column: str = "summary_embedding",
    pipeline_type: str = "TRIGGERED",
    endpoint_type: str = "STANDARD",
    wait_minutes: int = 20,
    trigger_sync: bool = True,
    columns_to_sync: Optional[List[str]] = None,
) -> "VectorSearchConfig":
    """
    Create (or verify) a Mosaic AI Vector Search endpoint + Delta-sync index
    on ``conversations.summary_embedding``.

    Idempotent. If the endpoint or index already exists, it's reused.

    Args:
        endpoint_name: Vector Search endpoint name (workspace-scoped).
            Free tier supports one ``STANDARD`` endpoint per workspace.
        index_name: Full three-part name (catalog.schema.idx). Defaults
            to ``{catalog}.{schema}.conversations_idx``.
        embedding_dim: Must match the dimensionality of vectors written to
            ``summary_embedding``. ``databricks-bge-large-en`` and
            ``databricks-gte-large-en`` are both 1024.
        pipeline_type: ``"TRIGGERED"`` (manual / scheduled sync) or
            ``"CONTINUOUS"`` (auto-sync as new rows land). Default
            ``TRIGGERED`` so cost is bounded; trigger manually after each
            ``summarize_conversations`` run.
        endpoint_type: ``"STANDARD"`` (covers most workloads) or
            ``"STORAGE_OPTIMIZED"`` (large indexes).
        wait_minutes: How long to wait for the endpoint to come up.
            New endpoints take 5-10 minutes.
        trigger_sync: When True (default), kick off an initial sync after
            creating the index so it's immediately queryable.

    Returns:
        A ``VectorSearchConfig`` to pass to ``search_conversations`` /
        ``ask_about_conversations``.
    """
    from datetime import timedelta

    from databricks.vector_search.client import VectorSearchClient

    if index_name is None:
        index_name = (
            f"{backend.config.catalog}.{backend.config.schema}.conversations_idx"
        )

    vsc = VectorSearchClient()

    # ── Endpoint ────────────────────────────────────────────────────
    try:
        existing = {e["name"] for e in (vsc.list_endpoints().get("endpoints") or [])}
    except Exception:
        existing = set()

    if endpoint_name not in existing:
        print(
            f"Creating Vector Search endpoint {endpoint_name} ({endpoint_type})…",
            flush=True,
        )
        vsc.create_endpoint(name=endpoint_name, endpoint_type=endpoint_type)
    else:
        print(f"Vector Search endpoint already exists: {endpoint_name}", flush=True)

    # Wait regardless — a "previously created" endpoint may still be
    # PROVISIONING from an earlier (failed) run, and creating an index
    # against a non-READY endpoint will error. The wrapper requires a
    # ``timedelta`` for ``timeout``, not an int.
    print(
        f"Waiting up to {wait_minutes} min for endpoint to become READY "
        f"(first-time setup takes 5-10 min)…",
        flush=True,
    )
    vsc.wait_for_endpoint(
        name=endpoint_name,
        timeout=timedelta(minutes=wait_minutes),
    )
    print(f"✓ Endpoint {endpoint_name} is READY", flush=True)

    # ── Index ───────────────────────────────────────────────────────
    try:
        idx_names = {
            idx["name"]
            for idx in (vsc.list_indexes(endpoint_name).get("vector_indexes") or [])
        }
    except Exception:
        idx_names = set()

    if index_name in idx_names:
        print(f"Vector Search index already exists: {index_name}", flush=True)
        index = vsc.get_index(endpoint_name, index_name)
    else:
        cols = (
            list(columns_to_sync)
            if columns_to_sync is not None
            else list(DEFAULT_VS_COLUMNS)
        )
        # Ensure the PK + vector column are always present even if the
        # caller passed a custom list.
        for required in (primary_key, embedding_column):
            if required not in cols:
                cols.append(required)
        print(
            f"Creating Delta-sync index {index_name} "
            f"(source={backend.conversations_table}, pk={primary_key}, "
            f"vec_col={embedding_column}, dim={embedding_dim}, pipeline={pipeline_type}, "
            f"columns_to_sync={cols})…",
            flush=True,
        )
        index = vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=backend.conversations_table,
            index_name=index_name,
            primary_key=primary_key,
            pipeline_type=pipeline_type,
            embedding_dimension=embedding_dim,
            embedding_vector_column=embedding_column,
            columns_to_sync=cols,
        )
        print(f"✓ Index {index_name} created", flush=True)

    if trigger_sync and pipeline_type == "TRIGGERED":
        try:
            print("Triggering initial sync…", flush=True)
            index.sync()
            print(
                "✓ Sync triggered (runs asynchronously — check the index in the UI)",
                flush=True,
            )
        except Exception as e:
            # Some SDK versions surface "already in progress" as an error.
            # Don't fail the setup over it.
            print(f"  sync trigger: {type(e).__name__}: {e}", flush=True)

    return VectorSearchConfig(
        endpoint_name=endpoint_name,
        index_name=index_name,
        embedding_dim=embedding_dim,
    )


def _unpack_search_result(result: Any, columns: List[str]) -> List[Dict[str, Any]]:
    """
    Normalize the Vector Search response into a list of dicts.

    The client returns nested structures whose exact shape varies across
    SDK versions; pull the data array out defensively and zip with the
    requested column list. The last element of each row is the similarity
    score.
    """
    # Try a few known shapes
    data_array = None
    if isinstance(result, dict):
        data_array = (result.get("result") or {}).get("data_array")
    else:
        try:
            data_array = result.result.data_array  # type: ignore[attr-defined]
        except AttributeError:
            pass
    if data_array is None:
        return []

    out: List[Dict[str, Any]] = []
    n_cols = len(columns)
    for row in data_array:
        # Each row has the requested columns plus a trailing score column.
        row_dict = {
            col: (row[i] if i < len(row) else None) for i, col in enumerate(columns)
        }
        if len(row) > n_cols:
            row_dict["_score"] = row[n_cols]
        out.append(row_dict)
    return out


def search_conversations(
    backend: "DatabricksBackend",
    config: "VectorSearchConfig",
    query: str,
    embedder: "Embedder",
    *,
    k: int = 10,
    columns: Optional[List[str]] = None,
    filters: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Embed ``query`` via ``embedder`` and return the top-k most similar
    conversations from the index.

    Args:
        columns: which columns to return per match. Defaults to a sensible
            set for conversation Q&A. To include large fields (full
            decoded text) pass them explicitly — but note that returning
            many large fields per match slows down the call.
        filters: Vector Search filter — either a dict (e.g.
            ``{"application": "support-bot"}``) or a JSON-encoded string of
            the same. ``None`` = no filter.

    Returns:
        A list of dicts with the requested columns + a ``_score`` field
        (cosine similarity if the index was built that way).
    """
    import json as _json

    from databricks.vector_search.client import VectorSearchClient

    if columns is None:
        columns = [
            "conversation_id",
            "summary_text",
            "application",
            "model",
            "ts",
            "total_tokens",
        ]

    query_vec = embedder.embed(query)
    vsc = VectorSearchClient()
    index = vsc.get_index(config.endpoint_name, config.index_name)

    base_kwargs = {
        "query_vector": query_vec,
        "columns": columns,
        "num_results": k,
    }

    if filters is None:
        raw = index.similarity_search(**base_kwargs)
    else:
        # databricks-vectorsearch has renamed the filter kwarg across
        # versions. Try ``filters`` (current), fall back to ``filters_json``
        # (older). Accept dict or JSON string and convert as needed.
        try:
            raw = index.similarity_search(filters=filters, **base_kwargs)
        except TypeError:
            filters_str = filters if isinstance(filters, str) else _json.dumps(filters)
            raw = index.similarity_search(filters_json=filters_str, **base_kwargs)

    return _unpack_search_result(raw, columns)


DEFAULT_SEGMENT_VS_COLUMNS = [
    "segment_hash",
    "segment_type",
    "ref_count",
    "embedding",
]


def _segments_embedded_table(backend: "DatabricksBackend") -> str:
    return f"{backend.config.catalog}.{backend.config.schema}.segments_embedded"


def materialize_segment_embeddings(
    backend: "DatabricksBackend",
    embedder: "Embedder",
    *,
    segment_types: Iterable[str] = ("system_prompt",),
    top_n: Optional[int] = None,
    only_new: bool = True,
    embed_reuse_threshold: Optional[float] = None,
    tokenizer: str = "cl100k_base",
    embed_batch_size: int = 32,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Embed each UNIQUE segment once and persist to
    ``<catalog>.<schema>.segments_embedded``.

    The dedup-aware embedding story: if 1000 conversations share a system
    prompt, we make 1 embedding API call instead of 1000. The savings are
    reported in the return dict so you can show them off.

    **Embedding reuse (``embed_reuse_threshold``).** Off by default (``None``):
    every unique segment — including near-dedup variants — is embedded from its
    own *exact* text, so search quality is identical to embedding each unique
    piece of text once. Set a threshold (e.g. ``0.97``) to let near-dedup
    variants whose Jaccard similarity to their canonical is at/above it **reuse
    the canonical's vector** instead of paying for their own embedding — an
    approximation that's lossless when the variation is search-irrelevant
    boilerplate (an interpolated id/date/name) and degrades ranking when the
    differing tokens carry meaning. Mirrors the vault's ``embed_reuse_threshold``
    (also off by default, stricter than the 0.85 storage-delta bar).

    Defaults to **embedding all unique segments of the requested types**
    (``top_n=None``). Embedding API calls are cheap (~$0.0001 each on
    bge-large-en) so a cap is artificial in the typical case. Set ``top_n``
    only if you're on a workload with millions of unique segments where
    driver-side collection would push memory or you want a strict spend
    bound.

    For each segment of the requested types (top-N most-referenced):
      1. Read the compressed bytes from ``segments``.
      2. Decompress to text via Stowken (driver-side).
      3. Batch-embed via ``embedder.embed_batch``.
      4. MERGE into ``segments_embedded``.

    Returns a dict with:
      - ``segments_embedded``: rows newly written this run
      - ``unique_segments``: total unique segments in scope after this run
      - ``total_references``: sum of ref_count across those segments
        (= number of embedding calls a naive per-conversation pipeline
        would have made for the same retrieval quality)
      - ``api_calls_saved``: ``total_references - unique_segments``
      - ``embedding_model``: model name written to the table
    """
    import time
    from datetime import datetime, timezone

    from pyspark.sql.types import (
        ArrayType,
        LongType,
        FloatType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    import stowken

    spark = backend.spark
    embedded_table = _segments_embedded_table(backend)
    model_name = getattr(embedder, "model_name", "unknown")

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {embedded_table} (
          segment_hash    STRING       NOT NULL,
          segment_type    STRING       NOT NULL,
          ref_count       BIGINT       NOT NULL,
          embedding       ARRAY<FLOAT> NOT NULL,
          embedding_model STRING       NOT NULL,
          embedded_at     TIMESTAMP    NOT NULL
        ) USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        """
    )

    types_list = ", ".join(f"'{t}'" for t in segment_types)
    cols = "segment_hash, segment_type, ref_count, content_zstd, canonical_hash, canonical_similarity"
    if top_n is None:
        ranked_sql = f"""
            SELECT {cols}
            FROM {backend.segments_table}
            WHERE segment_type IN ({types_list})
        """
    else:
        ranked_sql = f"""
            SELECT {cols} FROM (
              SELECT {cols},
                     ROW_NUMBER() OVER (PARTITION BY segment_type ORDER BY ref_count DESC) AS rn
              FROM {backend.segments_table}
              WHERE segment_type IN ({types_list})
            ) WHERE rn <= {int(top_n)}
        """

    # A variant is REUSED (rides its canonical's vector, no API call) only when
    # embedding reuse is enabled AND its similarity clears the threshold.
    # Everything else — canonicals and below-threshold variants — is embedded
    # from its own exact text. With reuse off (default), that's every unique
    # segment, so search quality is identical to "embed each unique text once".
    if embed_reuse_threshold is None:
        reuse_cond = "FALSE"
    else:
        reuse_cond = (
            "(c.canonical_hash IS NOT NULL AND c.canonical_similarity IS NOT NULL "
            f"AND c.canonical_similarity >= {float(embed_reuse_threshold)})"
        )
    embed_cond = f"NOT {reuse_cond}"

    anti = (
        f"LEFT ANTI JOIN {embedded_table} e ON c.segment_hash = e.segment_hash"
        if only_new
        else ""
    )
    # Variants are 0x04 deltas — fetch the canonical's bytes so the driver can
    # decode them before embedding.
    pending_sql = f"""
        WITH cand AS ({ranked_sql})
        SELECT c.segment_hash, c.segment_type, c.ref_count, c.content_zstd,
               canon.content_zstd AS canon_zstd
        FROM cand c
        LEFT JOIN {backend.segments_table} canon ON c.canonical_hash = canon.segment_hash
        {anti}
        WHERE {embed_cond}
    """
    pending = spark.sql(pending_sql).collect()

    # Scope stats (independent of what's already embedded). `to_embed` = segments
    # that need an API call; `to_reuse` = variants riding a canonical's vector.
    scope_stats_sql = f"""
        WITH cand AS ({ranked_sql})
        SELECT COUNT(*) AS unique_segments,
               COUNT_IF({embed_cond}) AS to_embed,
               COUNT_IF({reuse_cond}) AS to_reuse,
               COALESCE(SUM(ref_count), 0) AS total_references
        FROM cand c
    """
    scope = spark.sql(scope_stats_sql).first()
    unique_segments = int(scope["unique_segments"])
    to_embed = int(scope["to_embed"])
    to_reuse = int(scope["to_reuse"])
    total_references = int(scope["total_references"])

    out_schema = StructType(
        [
            StructField("segment_hash", StringType(), False),
            StructField("segment_type", StringType(), False),
            StructField("ref_count", LongType(), False),
            StructField("embedding", ArrayType(FloatType()), False),
            StructField("embedding_model", StringType(), False),
            StructField("embedded_at", TimestampType(), False),
        ]
    )

    total_written = 0
    overall_start = time.time()
    if pending:
        if progress:
            print(
                f"Decoding + embedding {len(pending)} segments "
                f"(scope: {unique_segments} unique, {to_embed} via API, {to_reuse} reuse, "
                f"{total_references} total refs, model={model_name})…",
                flush=True,
            )
        for i in range(0, len(pending), embed_batch_size):
            batch = pending[i : i + embed_batch_size]
            batch_start = time.time()

            # Canonicals are self-contained; below-threshold variants are 0x04
            # deltas — decode_segment_text resolves them via canon_zstd.
            texts: List[str] = []
            kept: List[Any] = []
            for row in batch:
                blob = row.content_zstd
                if blob is None:
                    continue
                canon = getattr(row, "canon_zstd", None)
                try:
                    texts.append(
                        stowken.decode_segment_text(
                            bytes(blob),
                            tokenizer,
                            bytes(canon) if canon is not None else None,
                        )
                    )
                    kept.append(row)
                except Exception as e:
                    if progress:
                        print(
                            f"  skip {row.segment_hash[:12]}…: {type(e).__name__}: {e}",
                            flush=True,
                        )

            if not texts:
                continue

            # Batched embedding API call
            vectors = embedder.embed_batch(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
                )

            now = datetime.now(timezone.utc)
            out_rows = [
                {
                    "segment_hash": row.segment_hash,
                    "segment_type": row.segment_type,
                    "ref_count": int(row.ref_count or 0),
                    "embedding": [float(x) for x in vec],
                    "embedding_model": model_name,
                    "embedded_at": now,
                }
                for row, vec in zip(kept, vectors)
            ]

            out_df = spark.createDataFrame(out_rows, schema=out_schema)
            out_df.createOrReplaceTempView("_stowken_segment_embed_stage")
            spark.sql(
                f"""
                MERGE INTO {embedded_table} t
                USING _stowken_segment_embed_stage s
                ON t.segment_hash = s.segment_hash
                WHEN MATCHED THEN UPDATE SET
                  t.embedding       = s.embedding,
                  t.embedding_model = s.embedding_model,
                  t.embedded_at     = s.embedded_at,
                  t.ref_count       = s.ref_count
                WHEN NOT MATCHED THEN INSERT *
                """
            )
            total_written += len(out_rows)

            if progress:
                elapsed = time.time() - batch_start
                print(
                    f"  batch {(i // embed_batch_size) + 1}: "
                    f"+{len(out_rows)} embedded in {elapsed:.1f}s  "
                    f"({total_written}/{len(pending)} done)",
                    flush=True,
                )
    elif progress:
        print(
            f"No new segments to embed "
            f"(scope: {unique_segments} unique, {to_embed} via API, {to_reuse} reuse).",
            flush=True,
        )

    # ── Embedding reuse (opt-in): high-similarity variants copy their
    # canonical's vector, no API call. Only runs when embed_reuse_threshold is
    # set; the similarity gate keeps it to variants close enough that the
    # canonical's embedding is a faithful stand-in.
    variants_reused = 0
    if embed_reuse_threshold is not None:
        only_new_filter = (
            f"AND NOT EXISTS (SELECT 1 FROM {embedded_table} x WHERE x.segment_hash = v.segment_hash)"
            if only_new
            else ""
        )
        spark.sql(
            f"""
            MERGE INTO {embedded_table} t
            USING (
              SELECT v.segment_hash, v.segment_type, CAST(v.ref_count AS BIGINT) AS ref_count,
                     e.embedding, e.embedding_model, current_timestamp() AS embedded_at
              FROM {backend.segments_table} v
              JOIN {embedded_table} e
                ON v.canonical_hash = e.segment_hash AND e.embedding_model = '{model_name}'
              WHERE v.segment_type IN ({types_list})
                AND v.canonical_hash IS NOT NULL
                AND v.canonical_similarity IS NOT NULL
                AND v.canonical_similarity >= {float(embed_reuse_threshold)}
                {only_new_filter}
            ) s
            ON t.segment_hash = s.segment_hash
            WHEN MATCHED THEN UPDATE SET
              t.embedding = s.embedding, t.embedding_model = s.embedding_model,
              t.embedded_at = s.embedded_at, t.ref_count = s.ref_count
            WHEN NOT MATCHED THEN INSERT *
            """
        )
        hist = spark.sql(f"DESCRIBE HISTORY {embedded_table} LIMIT 1").first()
        cm = (hist["operationMetrics"] or {}) if hist else {}
        variants_reused = int(cm.get("numTargetRowsInserted", 0)) + int(
            cm.get("numTargetRowsUpdated", 0)
        )

    overall_elapsed = time.time() - overall_start
    # API calls we made = `to_embed` (canonicals + below-threshold variants).
    # A naive per-conversation pipeline would have made `total_references`.
    api_calls_saved = max(total_references - to_embed, 0)
    if progress:
        reuse_note = (
            f" (+{variants_reused} variants reused a canonical's vector)"
            if embed_reuse_threshold is not None
            else ""
        )
        print(
            f"\n✓ Embedded {total_written} segments via API{reuse_note} in "
            f"{overall_elapsed:.1f}s. Naive per-conversation embedding would have made "
            f"{total_references} calls; we made {to_embed} — savings: "
            f"{api_calls_saved} ({100.0 * api_calls_saved / max(total_references, 1):.1f}%).",
            flush=True,
        )
    return {
        "segments_embedded": total_written,
        "variants_reused": variants_reused,
        "unique_segments": unique_segments,
        "api_segments": to_embed,
        "total_references": total_references,
        "api_calls_saved": api_calls_saved,
        "embedding_model": model_name,
    }


def setup_segment_index(
    backend: "DatabricksBackend",
    *,
    endpoint_name: str,
    index_name: Optional[str] = None,
    embedding_dim: int = 1024,
    pipeline_type: str = "TRIGGERED",
    endpoint_type: str = "STANDARD",
    wait_minutes: int = 20,
    trigger_sync: bool = True,
    columns_to_sync: Optional[List[str]] = None,
) -> "VectorSearchConfig":
    """
    Create (or verify) a Mosaic AI Vector Search Delta-sync index on the
    ``segments_embedded`` table. Same shape as :func:`setup_index` but
    points at segments rather than conversations.

    Pairs with :func:`materialize_segment_embeddings` — run that first to
    populate the source table.

    Defaults the index name to ``{catalog}.{schema}.segments_idx``.
    """
    from datetime import timedelta

    from databricks.vector_search.client import VectorSearchClient

    embedded_table = _segments_embedded_table(backend)
    if index_name is None:
        index_name = f"{backend.config.catalog}.{backend.config.schema}.segments_idx"

    vsc = VectorSearchClient()

    # Endpoint (reuses the same endpoint pattern as setup_index)
    try:
        existing = {e["name"] for e in (vsc.list_endpoints().get("endpoints") or [])}
    except Exception:
        existing = set()
    if endpoint_name not in existing:
        print(
            f"Creating Vector Search endpoint {endpoint_name} ({endpoint_type})…",
            flush=True,
        )
        vsc.create_endpoint(name=endpoint_name, endpoint_type=endpoint_type)
    else:
        print(f"Vector Search endpoint already exists: {endpoint_name}", flush=True)
    print(f"Waiting up to {wait_minutes} min for endpoint to become READY…", flush=True)
    vsc.wait_for_endpoint(name=endpoint_name, timeout=timedelta(minutes=wait_minutes))
    print(f"✓ Endpoint {endpoint_name} is READY", flush=True)

    # Index
    try:
        idx_names = {
            idx["name"]
            for idx in (vsc.list_indexes(endpoint_name).get("vector_indexes") or [])
        }
    except Exception:
        idx_names = set()

    if index_name in idx_names:
        print(f"Segment-level index already exists: {index_name}", flush=True)
        index = vsc.get_index(endpoint_name, index_name)
    else:
        cols = (
            list(columns_to_sync)
            if columns_to_sync is not None
            else list(DEFAULT_SEGMENT_VS_COLUMNS)
        )
        for required in ("segment_hash", "embedding"):
            if required not in cols:
                cols.append(required)
        print(
            f"Creating Delta-sync index {index_name} "
            f"(source={embedded_table}, pk=segment_hash, dim={embedding_dim}, "
            f"pipeline={pipeline_type}, columns_to_sync={cols})…",
            flush=True,
        )
        index = vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=embedded_table,
            index_name=index_name,
            primary_key="segment_hash",
            pipeline_type=pipeline_type,
            embedding_dimension=embedding_dim,
            embedding_vector_column="embedding",
            columns_to_sync=cols,
        )
        print(f"✓ Index {index_name} created", flush=True)

    if trigger_sync and pipeline_type == "TRIGGERED":
        try:
            print("Triggering initial sync…", flush=True)
            index.sync()
            print("✓ Sync triggered (runs asynchronously)", flush=True)
        except Exception as e:
            print(f"  sync trigger: {type(e).__name__}: {e}", flush=True)

    return VectorSearchConfig(
        endpoint_name=endpoint_name,
        index_name=index_name,
        embedding_dim=embedding_dim,
    )


# ── Index health / self-healing ──────────────────────────────────────────
#
# Mosaic AI Vector Search indexes can fail in two ways the search path must
# distinguish: a fresh index that is still provisioning (queries 404 until
# it's ready), and an index that has fallen into a persistent server-side
# 5xx state (``AISearchException: Something went wrong unexpectedly``). The
# first is a wait; the second is unrecoverable except by delete + recreate.
# Both surface to callers as opaque errors deep inside ``similarity_search``,
# so we probe the index over REST up front and classify it.

_INDEX_REST_PREFIX = "/api/2.0/vector-search/indexes"


def _classify_index_status(body: Optional[Dict[str, Any]]) -> str:
    """Map a successful GET-index response body to ``"ready"`` or ``"syncing"``."""
    status = (body or {}).get("status") or {}
    return "ready" if status.get("ready") else "syncing"


def _classify_index_error(exc: BaseException) -> str:
    """
    Map an exception from the GET-index call to ``"missing"`` or ``"broken"``.

    A 404 / ``RESOURCE_DOES_NOT_EXIST`` means the index simply isn't there
    (recoverable by creating it). Anything else — notably the persistent 5xx
    ``AISearchException`` an index can fall into — is ``"broken"`` and
    recoverable only by delete + recreate.
    """
    name = type(exc).__name__
    code = str(getattr(exc, "error_code", "") or "")
    status_code = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if (
        status_code == 404
        or name in ("NotFound", "ResourceDoesNotExist")
        or "RESOURCE_DOES_NOT_EXIST" in code
        or "does not exist" in msg
        or "not found" in msg
    ):
        return "missing"
    return "broken"


def _probe_index_state(client: Any, index_name: str) -> str:
    """GET the index over REST and classify into ready/syncing/missing/broken."""
    try:
        body = client.api_client.do("GET", f"{_INDEX_REST_PREFIX}/{index_name}")
    except Exception as exc:  # noqa: BLE001 — classify rather than swallow
        return _classify_index_error(exc)
    return _classify_index_status(body)


def _delete_index(client: Any, index_name: str) -> None:
    """DELETE the index over REST; treat an already-gone index as success."""
    try:
        client.api_client.do("DELETE", f"{_INDEX_REST_PREFIX}/{index_name}")
    except Exception as exc:  # noqa: BLE001
        if _classify_index_error(exc) != "missing":
            raise


def ensure_index_healthy(
    backend: "DatabricksBackend",
    endpoint_name: str,
    index_name: str,
    *,
    kind: Optional[str] = None,
    recreate: bool = False,
    require_ready: bool = True,
    wait_minutes: int = 20,
    poll_seconds: int = 30,
    _client: Any = None,
    **setup_kwargs: Any,
) -> str:
    """
    Make sure a Vector Search index is queryable before the search path uses it.

    Probes the index over REST and classifies it as one of ``"ready"``,
    ``"syncing"`` (provisioning — waited out), ``"missing"`` (never created),
    or ``"broken"`` (the persistent server-side 5xx state). Returns the final
    state string.

    Args:
        kind: ``"conversations"`` or ``"segments"`` — selects which setup
            function recreates the index. Inferred from the index name suffix
            (``*segments_idx`` → segments) when ``None``.
        recreate: when True, a ``"missing"`` index is created and a
            ``"broken"`` index is deleted and recreated, then waited until
            ready. When False, the bad state is reported as-is.
        require_ready: when True (default), raise a clear ``RuntimeError`` with
            remediation guidance if the index isn't ``"ready"`` at the end —
            so notebooks fail fast instead of dying inside ``similarity_search``
            on an opaque ``ErrorTraceId``. Set False to get the raw state.
        wait_minutes: overall budget for waiting out a sync / recreate.
        poll_seconds: interval between readiness polls.
        **setup_kwargs: forwarded to :func:`setup_index` /
            :func:`setup_segment_index` on the recreate path (e.g.
            ``embedding_dim``, ``pipeline_type``).

    Returns:
        The final state string (``"ready"`` unless ``require_ready=False`` and
        the index couldn't be made ready).
    """
    import time

    if kind is None:
        kind = "segments" if index_name.endswith("segments_idx") else "conversations"
    if kind not in ("conversations", "segments"):
        raise ValueError(f"kind must be 'conversations' or 'segments', got {kind!r}")

    if _client is None:
        from databricks.sdk import WorkspaceClient

        _client = WorkspaceClient()

    deadline = time.time() + wait_minutes * 60

    def _wait_ready() -> str:
        while True:
            state = _probe_index_state(_client, index_name)
            if state in ("ready", "broken"):
                return state
            if time.time() > deadline:
                return state  # "syncing" or "missing" at timeout
            print(
                f"  index {index_name}: {state} — waiting {poll_seconds}s…", flush=True
            )
            time.sleep(poll_seconds)

    def _recreate() -> str:
        setup = setup_index if kind == "conversations" else setup_segment_index
        print(f"Recreating {kind} index {index_name} via {setup.__name__}…", flush=True)
        setup(
            backend,
            endpoint_name=endpoint_name,
            index_name=index_name,
            wait_minutes=wait_minutes,
            **setup_kwargs,
        )
        return _wait_ready()

    state = _probe_index_state(_client, index_name)
    if state == "ready":
        result = "ready"
    elif state == "syncing":
        result = _wait_ready()
    elif state == "missing":
        result = _recreate() if recreate else "missing"
    elif state == "broken" and recreate:
        print(f"Index {index_name} is broken; deleting before recreate…", flush=True)
        _delete_index(_client, index_name)
        while _probe_index_state(_client, index_name) != "missing":
            if time.time() > deadline:
                break
            time.sleep(poll_seconds)
        result = _recreate()
    else:
        result = state  # broken, recreate=False

    if require_ready and result != "ready":
        raise RuntimeError(
            f"Vector Search index {index_name} is '{result}'. "
            f"Re-run 06_setup_vector_search to (re)create it, or call "
            f"ensure_index_healthy(..., recreate=True) to self-heal. "
            f"('broken' is a known Mosaic AI server-side 5xx state that only "
            f"delete + recreate clears.)"
        )
    return result


def find_conversations_via_segments(
    backend: "DatabricksBackend",
    segment_index: "VectorSearchConfig",
    query: str,
    embedder: "Embedder",
    *,
    k_segments: int = 10,
    k_conversations: int = 20,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search at segment granularity, return distinct conversations that
    contain any of the matched segments.

    Useful for queries that target a *specific reusable element* of
    conversations — e.g. "which conversations used the system prompt about
    refund policies?" — rather than the overall topic.

    Steps:
      1. Embed the query.
      2. Top-k_segments similarity search on the segment index.
      3. ``ANY MATCH`` join: conversations whose ``segment_hashes`` array
         intersects the matched segment hashes.
      4. Aggregate to the top-k_conversations distinct conversations,
         ranked by the best segment match (highest similarity score) that
         touched them.

    Returns a list of dicts with conversation columns + ``_top_segment_score``.
    """

    from databricks.vector_search.client import VectorSearchClient

    if columns is None:
        columns = [
            "conversation_id",
            "summary_text",
            "application",
            "model",
            "ts",
            "total_tokens",
        ]

    query_vec = embedder.embed(query)
    vsc = VectorSearchClient()
    index = vsc.get_index(segment_index.endpoint_name, segment_index.index_name)
    seg_columns = ["segment_hash", "segment_type", "ref_count"]
    raw = index.similarity_search(
        query_vector=query_vec,
        columns=seg_columns,
        num_results=k_segments,
    )
    seg_matches = _unpack_search_result(raw, seg_columns)
    if not seg_matches:
        return []

    # Build a Spark dataframe of matched (hash, score) and join to conversations.
    spark = backend.spark
    hash_scores = [
        (m["segment_hash"], float(m.get("_score") or 0.0)) for m in seg_matches
    ]
    seg_df = spark.createDataFrame(
        hash_scores, schema="segment_hash STRING, seg_score DOUBLE"
    )
    seg_df.createOrReplaceTempView("_stowken_seg_matches")

    col_list = ", ".join(f"c.{c}" for c in columns)
    out = spark.sql(
        f"""
        WITH exploded AS (
          SELECT c.conversation_id, EXPLODE(c.segment_hashes) AS segment_hash
          FROM {backend.conversations_table} c
        ),
        joined AS (
          SELECT e.conversation_id, MAX(m.seg_score) AS top_seg_score
          FROM exploded e
          JOIN _stowken_seg_matches m ON e.segment_hash = m.segment_hash
          GROUP BY e.conversation_id
        )
        SELECT {col_list}, j.top_seg_score AS _top_segment_score
        FROM joined j
        JOIN {backend.conversations_table} c ON c.conversation_id = j.conversation_id
        ORDER BY _top_segment_score DESC
        LIMIT {int(k_conversations)}
        """
    ).collect()

    return [r.asDict(recursive=True) for r in out]


_ROLE_FOR_SEGMENT_TYPE = {
    "system_prompt": "system",
    "context": "context",
    "user_turn": "user",
    "assistant_turn": "assistant",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "continuation": "continuation",
}


def _rehydrate_conversations(
    backend: "DatabricksBackend",
    conversation_ids: List[str],
    *,
    tokenizer: str = "cl100k_base",
    max_chars_per_conv: int = 4000,
) -> Dict[str, str]:
    """
    For each ``conversation_id``, stitch its segments into a single
    ``[role] text\\n[role] text\\n…`` string. Driver-side decompression via
    Stowken; cheap because segments are deduped (we decompress each unique
    blob once even if multiple conversations reference it).

    Returns: ``{conversation_id: full_text}``. Conversations with no
    resolvable segments map to an empty string. ``max_chars_per_conv``
    truncates the per-conversation text to bound LLM context size.
    """
    import stowken
    from pyspark.sql.functions import col

    if not conversation_ids:
        return {}

    spark = backend.spark
    conv_rows = (
        spark.table(backend.conversations_table)
        .filter(col("conversation_id").isin(list(conversation_ids)))
        .select("conversation_id", "segment_hashes", "segment_types", "tokenizer")
        .collect()
    )
    conv_segs = {
        r.conversation_id: (
            r.segment_hashes or [],
            r.segment_types or [],
            r.tokenizer or tokenizer,
        )
        for r in conv_rows
    }

    all_hashes = {h for hashes, _, _ in conv_segs.values() for h in hashes}
    if not all_hashes:
        return {cid: "" for cid in conv_segs}

    seg_rows = backend.get_segment_blobs_with_canonical(all_hashes).collect()
    text_map: Dict[str, str] = {}
    for r in seg_rows:
        blob = r.content_zstd
        if blob is None:
            continue
        canon = r.canon_zstd
        try:
            text_map[r.segment_hash] = stowken.decode_segment_text(
                bytes(blob),
                tokenizer,
                canonical_zstd=bytes(canon) if canon is not None else None,
            )
        except Exception:
            text_map[r.segment_hash] = ""

    out: Dict[str, str] = {}
    for cid, (hashes, types, _tok) in conv_segs.items():
        parts: List[str] = []
        for h, t in zip(hashes, types):
            text = text_map.get(h)
            if not text:
                continue
            role = _ROLE_FOR_SEGMENT_TYPE.get(t, t or "user")
            parts.append(f"[{role}] {text}")
        stitched = "\n".join(parts)
        if max_chars_per_conv and len(stitched) > max_chars_per_conv:
            stitched = stitched[:max_chars_per_conv] + " … [truncated]"
        out[cid] = stitched
    return out


def ask_about_conversations(
    backend: "DatabricksBackend",
    config: "VectorSearchConfig",
    question: str,
    embedder: "Embedder",
    *,
    qa_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    k: int = 10,
    max_tokens: int = 600,
    rehydrate_full_text: bool = True,
    max_chars_per_conv: int = 4000,
    tokenizer: str = "cl100k_base",
) -> Dict[str, Any]:
    """
    Retrieval-augmented Q&A over the conversation history.

    Steps:
      1. Embed the question.
      2. Top-k semantic search on the summary-embedding index.
      3. If ``rehydrate_full_text`` (default): pull each matched conversation's
         segments via Stowken and reconstruct the full text. This is where
         Stowken's storage approach pays off — the compressed segments are
         cheap to decompress on demand, so the LLM gets full grounding
         instead of just the lossy summary.
      4. Pass the chosen context (full text or summary) + question to
         ``qa_endpoint`` (a chat completion endpoint).
      5. Return the LLM's grounded answer plus the source rows.

    ``max_chars_per_conv`` caps per-conversation text size before it hits
    the LLM context window. Default 4000 chars (~1000 tokens) × k=10 ≈
    10k tokens of context, comfortable on a 32k+ context model.

    Set ``rehydrate_full_text=False`` to fall back to summary-only Q&A —
    useful for cost comparisons or when summaries are sufficient.

    Returns: ``{"answer": str, "sources": [match_dict, …]}``. Sources
    include a ``full_text`` field when rehydration was used.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    matches = search_conversations(backend, config, question, embedder, k=k)
    if not matches:
        return {
            "answer": "No matching conversations found in the index.",
            "sources": [],
        }

    # Rehydrate full text via Stowken if requested.
    if rehydrate_full_text:
        full_map = _rehydrate_conversations(
            backend,
            [m["conversation_id"] for m in matches if m.get("conversation_id")],
            tokenizer=tokenizer,
            max_chars_per_conv=max_chars_per_conv,
        )
        for m in matches:
            m["full_text"] = full_map.get(m.get("conversation_id"), "")

    context_label = "Full conversation" if rehydrate_full_text else "Summary"
    context_lines = []
    for i, m in enumerate(matches, start=1):
        text = (
            (m.get("full_text") if rehydrate_full_text else m.get("summary_text"))
            or m.get("summary_text")
            or "<no text>"
        )
        context_lines.append(
            f"[{i}] conversation_id={m.get('conversation_id', '?')} "
            f"(app={m.get('application', '?')}, model={m.get('model', '?')}, "
            f"score={m.get('_score', 0):.3f}) — {context_label}:\n"
            f"  {text}"
        )
    context = "\n\n".join(context_lines)

    system_prompt = (
        "You are an analyst answering questions about an LLM-application's "
        "production conversation history. Answer the user's question using ONLY "
        "the conversation context provided. Cite specific conversations with "
        "their [number] when relevant. If the context doesn't contain enough "
        "information to answer, say so plainly — don't speculate."
    )

    w = WorkspaceClient()
    resp = w.serving_endpoints.query(
        name=qa_endpoint,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(
                role=ChatMessageRole.USER,
                content=f"Conversation context:\n{context}\n\nQuestion: {question}",
            ),
        ],
        max_tokens=max_tokens,
    )
    answer = resp.choices[0].message.content.strip()
    return {"answer": answer, "sources": matches}
