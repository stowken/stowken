"""
Stowken on Databricks.

Stores Stowken segments and conversation manifests as native Delta tables in
Unity Catalog. Analytics queries run against the manifest table without
rehydrating compressed segment blobs.

Available only with the ``databricks`` extra::

    pip install 'stowken[databricks]'

Importing this module on a host without PySpark / delta-spark / databricks-sdk
will raise ``ImportError`` at import time, by design — the rest of Stowken
remains usable without those dependencies.
"""

from .analytics import (
    conversation_length_distribution,
    cost_by_application_and_model,
    dedup_ratio,
    find_outlier_conversations,
    materialize_decoded_segments,
    token_spend_over_time,
    top_system_prompts,
)
from .backend import BackendConfig, DatabricksBackend
from .ingest import IngestResult, SourceColumnMapping, ingest_from_delta_table
from .near_dedup import NearDedupResult, near_dedup_segments
from .schema import (
    CONVERSATIONS_DDL,
    SEGMENTS_DDL,
    create_tables,
    migrate_segments_add_near_dedup,
)
from .summarize import (
    DatabricksFoundationModelEmbedder,
    DatabricksFoundationModelSummarizer,
    Embedder,
    Summarizer,
    summarize_conversations,
)
from .vector_search import (
    VectorSearchConfig,
    ask_about_conversations,
    ensure_index_healthy,
    find_conversations_via_segments,
    materialize_segment_embeddings,
    search_conversations,
    setup_index,
    setup_segment_index,
)

__all__ = [
    # backend
    "BackendConfig",
    "DatabricksBackend",
    # ingest
    "IngestResult",
    "SourceColumnMapping",
    "ingest_from_delta_table",
    # schema
    "CONVERSATIONS_DDL",
    "SEGMENTS_DDL",
    "create_tables",
    "migrate_segments_add_near_dedup",
    # near-dedup (post-ingest delta pass)
    "NearDedupResult",
    "near_dedup_segments",
    # summarize
    "DatabricksFoundationModelEmbedder",
    "DatabricksFoundationModelSummarizer",
    "Embedder",
    "Summarizer",
    "summarize_conversations",
    # analytics
    "conversation_length_distribution",
    "cost_by_application_and_model",
    "dedup_ratio",
    "find_outlier_conversations",
    "materialize_decoded_segments",
    "token_spend_over_time",
    "top_system_prompts",
    # vector search — conversation level
    "VectorSearchConfig",
    "ask_about_conversations",
    "ensure_index_healthy",
    "search_conversations",
    "setup_index",
    # vector search — segment level (dedup-aware embeddings)
    "find_conversations_via_segments",
    "materialize_segment_embeddings",
    "setup_segment_index",
]
