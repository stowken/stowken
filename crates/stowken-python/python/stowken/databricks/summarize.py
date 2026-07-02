"""
Summarization + embedding for Stowken-on-Databricks.

The Databricks ingest path uses ``chunk_only``, which is synchronous and
stateless — it cannot make LLM calls. Summarization runs as a separate
post-ingest step that backfills the ``summary_text``, ``summary_embedding``,
``summary_model``, and ``summary_generated_at`` columns on rows where they
are missing.

This module defines the ``Summarizer`` and ``Embedder`` protocols so
customers can plug in any LLM. The default implementations
(``DatabricksFoundationModelSummarizer``, ``DatabricksFoundationModelEmbedder``)
call Databricks Foundation Model endpoints via ``databricks.sdk``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .backend import DatabricksBackend


@runtime_checkable
class Summarizer(Protocol):
    """Generates a natural-language summary of a conversation."""

    @property
    def model_name(self) -> str:
        """Identifier persisted to ``conversations.summary_model``."""
        ...

    def summarize(self, conversation_text: str) -> str:
        """Return a summary for one conversation's full text."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Generates an embedding vector from text."""

    @property
    def model_name(self) -> str:
        """Identifier useful when joining downstream Vector Search indexes."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings; preferred for throughput."""
        ...


@dataclass
class DatabricksFoundationModelSummarizer:
    """
    Default ``Summarizer`` that calls a Databricks Foundation Model chat
    endpoint. Defaults to ``databricks-meta-llama-3-3-70b-instruct``; override
    by constructing with a different ``endpoint_name``.

    The caller's principal must have ``CAN_QUERY`` on the endpoint and a
    workspace authentication context that ``databricks.sdk.WorkspaceClient``
    can resolve (notebook execution context, OAuth, or PAT).
    """

    endpoint_name: str = "databricks-meta-llama-3-3-70b-instruct"
    system_prompt: str = (
        "You are a summarizer. Produce a concise 1-3 sentence summary of the "
        "following LLM conversation, focusing on the task the user was trying "
        "to accomplish and whether the assistant completed it successfully."
    )
    max_tokens: int = 200

    @property
    def model_name(self) -> str:
        return self.endpoint_name

    def summarize(self, conversation_text: str) -> str:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=self.endpoint_name,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(role=ChatMessageRole.USER,   content=conversation_text),
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()


@dataclass
class DatabricksFoundationModelEmbedder:
    """
    Default ``Embedder`` that calls a Databricks Foundation Model embedding
    endpoint. Defaults to ``databricks-bge-large-en`` (1024-dim).
    """

    endpoint_name: str = "databricks-bge-large-en"

    @property
    def model_name(self) -> str:
        return self.endpoint_name

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=self.endpoint_name,
            input=texts,
        )
        return [item.embedding for item in response.data]


def summarize_conversations(
    backend: "DatabricksBackend",
    summarizer: Summarizer,
    embedder: Embedder,
    *,
    batch_size: int = 256,
    only_missing: bool = True,
    max_text_chars: int = 20000,
    max_workers: int = 8,
    progress: bool = True,
) -> int:
    """
    Backfill ``summary_text``, ``summary_embedding``, ``summary_model``, and
    ``summary_generated_at`` for conversations.

    Walks ``conversations`` filtered to ``summary_text IS NULL`` (when
    ``only_missing`` is True), rehydrates each conversation by joining
    through ``segments``, runs the rehydrated text through ``summarizer``
    and the resulting summary through ``embedder``, then ``MERGE`` s the
    summary columns back into the conversations table.

    ``max_text_chars`` truncates very long rehydrated conversations before
    they hit the summarizer so we don't blow LLM context windows on outlier
    rows. The truncation marker is appended.

    ``max_workers`` controls the number of concurrent summarizer calls per
    batch. Shared Foundation Model endpoints are typically throughput-limited
    per workspace rather than per-request, so 4-16 concurrent calls usually
    yields a 4-8x wall-clock speedup over sequential. Set to 1 to disable
    parallelism (useful when debugging an adapter implementation).

    ``progress`` toggles per-batch stdout logging with throughput + ETA.

    Returns the number of rows updated.

    Note: rehydration runs on the driver (single Stowken vault per call) for
    v1 simplicity. For very large backfills (>10k pending rows), partition
    the work and call this function per partition.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone

    from pyspark.sql.functions import col
    from pyspark.sql.types import (
        ArrayType,
        FloatType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    import stowken

    spark = backend.spark

    # ── 1. Find conversations needing summaries ────────────────────────────
    conv_df = spark.table(backend.conversations_table)
    if only_missing:
        conv_df = conv_df.filter(col("summary_text").isNull())

    pending = conv_df.select(
        "conversation_id", "segment_hashes", "segment_types", "tokenizer"
    ).collect()
    if not pending:
        return 0

    # ── 2. Fetch every referenced segment blob (+ canonical, for deltas) ──
    all_hashes = {h for row in pending for h in (row.segment_hashes or [])}
    seg_df = backend.get_segment_blobs_with_canonical(all_hashes)
    seg_map = {
        row.segment_hash: (row.content_zstd, row.canon_zstd)
        for row in seg_df.collect()
    }

    # ── 3. Rehydrate each conversation to text ─────────────────────────────
    role_for_type = {
        "system_prompt":   "system",
        "context":         "context",
        "user_turn":       "user",
        "assistant_turn":  "assistant",
        "tool_call":       "tool_call",
        "tool_result":     "tool_result",
        "continuation":    "continuation",
    }
    rehydrated: list[dict] = []
    for row in pending:
        tokenizer = row.tokenizer or "cl100k_base"
        parts: list[str] = []
        for hash_, seg_type in zip(row.segment_hashes or [], row.segment_types or []):
            blob, canon = seg_map.get(hash_, (None, None))
            if blob is None:
                continue
            text = stowken.decode_segment_text(
                bytes(blob),
                tokenizer,
                canonical_zstd=bytes(canon) if canon is not None else None,
            )
            role = role_for_type.get(seg_type, seg_type or "user")
            parts.append(f"[{role}] {text}")
        full = "\n".join(parts)
        if len(full) > max_text_chars:
            full = full[:max_text_chars] + " ... [truncated]"
        rehydrated.append({"conversation_id": row.conversation_id, "text": full})

    # ── 4. Summarize + embed in batches, then MERGE back into conversations
    update_schema = StructType(
        [
            StructField("conversation_id",      StringType(),    False),
            StructField("summary_text",         StringType(),    True),
            StructField("summary_embedding",    ArrayType(FloatType()), True),
            StructField("summary_model",        StringType(),    True),
            StructField("summary_generated_at", TimestampType(), True),
        ]
    )

    updated_total = 0
    summary_model = getattr(embedder, "model_name", None) or getattr(summarizer, "model_name", "unknown")
    n_batches = (len(rehydrated) + batch_size - 1) // batch_size
    overall_start = time.time()

    def _summarize_one(item):
        return summarizer.summarize(item["text"])

    if progress:
        print(
            f"summarize_conversations: {len(rehydrated):,} pending rows, "
            f"{n_batches} batches of {batch_size}, "
            f"max_workers={max_workers}",
            flush=True,
        )

    for batch_idx, i in enumerate(range(0, len(rehydrated), batch_size), start=1):
        batch = rehydrated[i : i + batch_size]
        batch_start = time.time()

        # Concurrent summarization. Each call is one LLM round-trip; shared
        # endpoints handle parallel requests well so we get near-linear
        # speedup up to ~16 workers on typical workspaces.
        if max_workers <= 1 or len(batch) <= 1:
            summaries = [_summarize_one(item) for item in batch]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                summaries = list(pool.map(_summarize_one, batch))

        embeddings = embedder.embed_batch(summaries)
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(embeddings)} vectors for {len(batch)} summaries"
            )

        ts = datetime.now(timezone.utc)
        update_rows = [
            {
                "conversation_id":      item["conversation_id"],
                "summary_text":         summary,
                "summary_embedding":    [float(x) for x in emb],
                "summary_model":        summary_model,
                "summary_generated_at": ts,
            }
            for item, summary, emb in zip(batch, summaries, embeddings)
        ]
        update_df = spark.createDataFrame(update_rows, schema=update_schema)
        update_df.createOrReplaceTempView("_stowken_summary_updates")
        spark.sql(
            f"""
            MERGE INTO {backend.conversations_table} t
            USING _stowken_summary_updates s
            ON t.conversation_id = s.conversation_id
            WHEN MATCHED THEN UPDATE SET
                t.summary_text         = s.summary_text,
                t.summary_embedding    = s.summary_embedding,
                t.summary_model        = s.summary_model,
                t.summary_generated_at = s.summary_generated_at
            """
        )
        updated_total += len(batch)

        if progress:
            elapsed = time.time() - batch_start
            overall = time.time() - overall_start
            rate_per_min = (updated_total / overall * 60) if overall > 0 else 0.0
            remaining = len(rehydrated) - updated_total
            eta_min = (remaining / rate_per_min) if rate_per_min > 0 else float("inf")
            print(
                f"  [{batch_idx:3d}/{n_batches}] +{len(batch)} in {elapsed:5.1f}s  "
                f"— {updated_total:,}/{len(rehydrated):,} done  "
                f"({rate_per_min:5.0f} rows/min, ETA {eta_min:5.1f} min)",
                flush=True,
            )

    return updated_total
