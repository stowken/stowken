//! Semantic similarity search & analytics.
//!
//! This module is gated behind the `semantic-search` feature. It adds:
//!   * Per-unique-segment embeddings (deduplication-friendly).
//!   * Per-conversation summary embeddings (context-aware).
//!   * `semantic_search` with optional structural filters and a pluggable
//!     `EmbeddingAdapter`.
//!   * Typed search wrappers (`search_prompts`, `search_responses`, ...).
//!   * `cluster_conversations` and `find_outliers` for analytical workflows.
//!
//! Embeddings are stored L2-normalized so cosine similarity reduces to a
//! dot product at query time. Brute-force in-process scan is used; ANN is
//! deferred until the unique-segment count grows beyond ~1M.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::task;
use tracing::info;

use crate::clustering::{kmeans, KMeansConfig};
use crate::compression::FrameVersion;
use crate::index::metadata::BatchSegmentOp;
use crate::storage::backend::StorageBackend;
use crate::types::{
    ConversationCluster, EmbedStats, EmbeddingAdapter, MatchedVia, OutlierConversation,
    RetrievedConversation, SearchGranularity, SegmentHash, SegmentType, SemanticSearchHit,
    SemanticSearchQuery, StoredSegment, SummaryStrategy, Token, TokenizerAdapter,
};
use crate::vault::{Stowken, StowkenError};

const EMBED_BATCH_SIZE: usize = 64;
/// Max concurrent summary-generation calls within a batch. Keeps LLM API
/// traffic well under typical 500 RPM limits (8 concurrent × ~2 req/s ≈ 960 RPM
/// headroom, but retries handle occasional bursts).
const SUMMARY_CONCURRENCY: usize = 8;

/// L2-normalize a vector. Returns the input unchanged if its norm is below
/// `f32::EPSILON` (avoids dividing by zero for empty/degenerate vectors).
pub(crate) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Build the canonical "[role] text" representation used for both concat-and-
/// truncate summaries and the input to LLM-generated summaries. Skips
/// continuation segments — they're folded into their parents implicitly by
/// being ordered after.
fn detokenize_with_role_markers(
    conv: &RetrievedConversation,
    tokenizer: Option<&Arc<dyn TokenizerAdapter>>,
    fallback_tokenizer_name: &str,
) -> String {
    let tok: Option<Arc<dyn TokenizerAdapter>> = tokenizer
        .cloned()
        .or_else(|| crate::tokenizer::get_tokenizer(fallback_tokenizer_name).map(Arc::from));

    let mut out = String::new();
    for seg in &conv.segments {
        let role = match seg.segment_type {
            SegmentType::SystemPrompt => "system",
            SegmentType::Context => "context",
            SegmentType::UserTurn => "user",
            SegmentType::AssistantTurn => "assistant",
            SegmentType::ToolCall => "tool_call",
            SegmentType::ToolResult => "tool_result",
            SegmentType::Continuation => "continuation",
        };
        let text = match &tok {
            Some(t) => t.detokenize(&seg.tokens),
            None => seg
                .tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push('[');
        out.push_str(role);
        out.push_str("] ");
        out.push_str(&text);
    }
    out
}

/// Truncate a string to `max_chars` characters (UTF-8 boundary safe).
fn truncate_safely(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_owned();
    }
    s.chars().take(max_chars).collect()
}

/// Sort `(key, embedding)` pairs by dot-product against `query_vec` and
/// return the top-K with `score >= min_score`. Pre-condition: every
/// embedding is L2-normalized and has the same length as `query_vec`.
fn scan_top_k<K: Clone>(
    query_vec: &[f32],
    embeddings: &[(K, Vec<f32>)],
    k: usize,
    min_score: f32,
) -> Vec<(K, f32)> {
    let dim = query_vec.len();
    let mut scored: Vec<(K, f32)> = embeddings
        .iter()
        .filter_map(|(key, emb)| {
            if emb.len() != dim {
                return None;
            }
            let score: f32 = query_vec.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            if score >= min_score {
                Some((key.clone(), score))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    scored
}

/// Merge a hit into the per-conversation map, keeping the best-scoring
/// match per conversation_id.
fn upsert_best(
    hits: &mut HashMap<String, SemanticSearchHit>,
    candidate: SemanticSearchHit,
) {
    use std::collections::hash_map::Entry;
    match hits.entry(candidate.conversation_id.clone()) {
        Entry::Vacant(v) => {
            v.insert(candidate);
        }
        Entry::Occupied(mut o) => {
            if candidate.score > o.get().score {
                o.insert(candidate);
            }
        }
    }
}

impl<B: StorageBackend + 'static> Stowken<B> {
    // ── Embedding core ────────────────────────────────────────────────────

    /// Embed a batch of unique segment hashes. Skips delta frames (the
    /// canonical captures the meaning) and segments that fail to decompress
    /// or detokenize. Persists L2-normalized embeddings via
    /// `MetadataIndex::upsert_segment_embedding`.
    pub(crate) async fn embed_segment_hashes(
        &self,
        hashes: &[SegmentHash],
        tokenizer_name: &str,
        adapter: Arc<dyn EmbeddingAdapter>,
    ) -> Result<u64, StowkenError> {
        if hashes.is_empty() {
            return Ok(0);
        }
        let total = hashes.len();
        let mut embedded: u64 = 0;
        for chunk in hashes.chunks(EMBED_BATCH_SIZE) {
            let mut texts: Vec<String> = Vec::with_capacity(chunk.len());
            let mut valid_hashes: Vec<SegmentHash> = Vec::with_capacity(chunk.len());

            for hash in chunk {
                let stored: StoredSegment = match self.backend_get_segment(hash).await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if stored.compressed_data.first() == Some(&(FrameVersion::Delta as u8)) {
                    continue;
                }
                let tokens = match self
                    .decompress_segment_async(&stored.compressed_data)
                    .await
                {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if tokens.is_empty() {
                    continue;
                }
                let text = self.detokenize_segment_tokens(&tokens, tokenizer_name);
                if text.trim().is_empty() {
                    continue;
                }
                texts.push(text);
                valid_hashes.push(hash.clone());
            }

            if texts.is_empty() {
                continue;
            }

            let adapter_clone = Arc::clone(&adapter);
            let owned_texts: Vec<String> = texts;
            let embeddings = task::spawn_blocking(move || {
                let refs: Vec<&str> = owned_texts.iter().map(String::as_str).collect();
                adapter_clone.embed_batch(&refs)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(StowkenError::Embedding)?;

            if embeddings.len() != valid_hashes.len() {
                return Err(StowkenError::Embedding(format!(
                    "embedder returned {} vectors for {} inputs",
                    embeddings.len(),
                    valid_hashes.len()
                )));
            }

            let model_name = adapter.model_name().to_owned();
            for (hash, raw) in valid_hashes.iter().zip(embeddings.iter()) {
                let normalized = l2_normalize(raw);
                let index = self.index_clone();
                let h = hash.clone();
                let mn = model_name.clone();
                task::spawn_blocking(move || {
                    index.upsert_segment_embedding(&h, &mn, &normalized)
                })
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;
                embedded += 1;
            }
            eprintln!("  segments: {}/{total}", embedded);
        }
        Ok(embedded)
    }

    /// Build a text summary of a conversation according to the active
    /// strategy and embed it. The summary text is persisted alongside the
    /// embedding for transparency.
    pub(crate) async fn embed_conversation(
        &self,
        conv_id: &str,
        adapter: Arc<dyn EmbeddingAdapter>,
        strategy: &SummaryStrategy,
        tokenizer_name: &str,
    ) -> Result<bool, StowkenError> {
        let conv = self.retrieve(conv_id).await?;
        let summary = self
            .build_conversation_summary(&conv, strategy, tokenizer_name)
            .await?;
        if summary.trim().is_empty() {
            return Ok(false);
        }

        let adapter_clone = Arc::clone(&adapter);
        let summary_for_embed = summary.clone();
        let embeddings = task::spawn_blocking(move || {
            adapter_clone.embed_batch(&[summary_for_embed.as_str()])
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))?
        .map_err(StowkenError::Embedding)?;

        if embeddings.is_empty() {
            return Err(StowkenError::Embedding(
                "embedder returned no vectors for conversation summary".into(),
            ));
        }

        let normalized = l2_normalize(&embeddings[0]);
        let model_name = adapter.model_name().to_owned();
        let strategy_id = strategy.id();
        let conv_id = conv_id.to_owned();
        let summary_clone = summary;
        let index = self.index_clone();
        task::spawn_blocking(move || {
            index.upsert_conversation_embedding(
                &conv_id,
                &model_name,
                &strategy_id,
                &normalized,
                Some(summary_clone.as_str()),
            )
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;
        Ok(true)
    }

    /// Build the summary text for a retrieved conversation per the active
    /// strategy. `ConcatTruncate` is sync; `LlmGenerated` calls the
    /// summarizer adapter via `spawn_blocking`.
    pub(crate) async fn build_conversation_summary(
        &self,
        conv: &RetrievedConversation,
        strategy: &SummaryStrategy,
        tokenizer_name: &str,
    ) -> Result<String, StowkenError> {
        let full_text = detokenize_with_role_markers(
            conv,
            self.tokenizer_arc().as_ref(),
            tokenizer_name,
        );

        match strategy {
            SummaryStrategy::ConcatTruncate { max_chars } => {
                Ok(truncate_safely(&full_text, *max_chars))
            }
            SummaryStrategy::LlmGenerated(summarizer) => {
                let s = Arc::clone(summarizer);
                let text = full_text;
                task::spawn_blocking(move || s.summarize(&text))
                    .await
                    .map_err(|e| StowkenError::Internal(e.to_string()))?
                    .map_err(StowkenError::Summarization)
            }
        }
    }

    // ── Embed-on-store hook ───────────────────────────────────────────────

    /// Called from `store()` after the manifest is committed. Idempotent:
    /// segments that already have an embedding for the active model are
    /// skipped, so a partial run followed by a retry won't double-bill the
    /// embedding API.
    pub(crate) async fn run_embed_on_store(
        &self,
        new_segment_hashes: Vec<SegmentHash>,
        conversation_id: &str,
        tokenizer_name: &str,
    ) -> Result<(), StowkenError> {
        if !self.embed_on_store_enabled() {
            return Ok(());
        }
        let adapter = match self.embedding_adapter_clone() {
            Some(a) => a,
            None => return Ok(()),
        };

        if !new_segment_hashes.is_empty() {
            let model = adapter.model_name().to_owned();
            let index = self.index_clone();
            let hashes_for_check = new_segment_hashes.clone();
            let already_embedded = task::spawn_blocking(move || {
                index.embedded_segment_hashes(&hashes_for_check, &model)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

            let to_embed: Vec<SegmentHash> = new_segment_hashes
                .into_iter()
                .filter(|h| !already_embedded.contains(h))
                .collect();
            if !to_embed.is_empty() {
                self.embed_segment_hashes(&to_embed, tokenizer_name, Arc::clone(&adapter))
                    .await?;
            }
        }

        let strategy = self.summary_strategy_clone();
        self.embed_conversation(conversation_id, adapter, &strategy, tokenizer_name)
            .await?;
        Ok(())
    }

    // ── Bulk embedding ────────────────────────────────────────────────────

    /// Embed all unique segments that lack an embedding for the configured
    /// model. Uses the vault's tokenizer if set; otherwise falls back to
    /// `cl100k_base` via the bundled `tiktoken` integration.
    pub async fn embed_all_segments(&self) -> Result<u64, StowkenError> {
        let adapter = self
            .embedding_adapter_clone()
            .ok_or_else(|| StowkenError::Internal("no embedding adapter set".into()))?;
        let model = adapter.model_name().to_owned();
        let index = self.index_clone();
        let unembedded = task::spawn_blocking(move || {
            index.unembedded_segment_hashes(&model, u64::MAX)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;
        let tokenizer_name = self.default_tokenizer_name();
        self.embed_segment_hashes(&unembedded, &tokenizer_name, adapter)
            .await
    }

    /// Embed a summary for every conversation that lacks one for the active
    /// (model, strategy) pair.
    pub async fn embed_all_conversations(&self) -> Result<u64, StowkenError> {
        let adapter = self
            .embedding_adapter_clone()
            .ok_or_else(|| StowkenError::Internal("no embedding adapter set".into()))?;
        let strategy = self.summary_strategy_clone();
        let model = adapter.model_name().to_owned();
        let strategy_id = strategy.id();
        let index = self.index_clone();
        let ids = task::spawn_blocking(move || {
            index.unembedded_conversation_ids(&model, &strategy_id, u64::MAX)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;

        let tokenizer_name = self.default_tokenizer_name();
        let total = ids.len();
        let mut embedded: u64 = 0;

        // `&self` is Copy, so it can be captured by multiple async blocks at once.
        // `join_all` polls them all concurrently in this task — no spawning needed.
        // For LLM strategy each block fires a spawn_blocking (GPT call) that runs
        // on Tokio's thread pool, giving real parallelism across the batch.
        use futures::StreamExt as _;
        for chunk in ids.chunks(EMBED_BATCH_SIZE) {
            // Drive at most SUMMARY_CONCURRENCY summary calls in flight at once
            // to stay within typical LLM API rate limits.
            let results: Vec<(String, String)> = futures::stream::iter(chunk.iter().cloned())
                .map(|id| {
                    let strategy = strategy.clone();
                    let tokenizer_name = tokenizer_name.clone();
                    async move {
                        let conv = self.retrieve(&id).await?;
                        let summary = self
                            .build_conversation_summary(&conv, &strategy, &tokenizer_name)
                            .await?;
                        Ok::<(String, String), StowkenError>((id, summary))
                    }
                })
                .buffer_unordered(SUMMARY_CONCURRENCY)
                .collect::<Vec<Result<_, _>>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .filter(|(_, s)| !s.trim().is_empty())
                .collect();

            if results.is_empty() {
                continue;
            }

            let valid_ids: Vec<String> = results.iter().map(|(id, _)| id.clone()).collect();
            let summaries: Vec<String> = results.iter().map(|(_, s)| s.clone()).collect();

            // One embedding API call for the whole batch.
            let adapter_clone = Arc::clone(&adapter);
            let owned_summaries = summaries.clone();
            let embeddings = task::spawn_blocking(move || {
                let refs: Vec<&str> = owned_summaries.iter().map(String::as_str).collect();
                adapter_clone.embed_batch(&refs)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(StowkenError::Embedding)?;

            if embeddings.len() != valid_ids.len() {
                return Err(StowkenError::Embedding(format!(
                    "embedder returned {} vectors for {} conversation summaries",
                    embeddings.len(),
                    valid_ids.len()
                )));
            }

            let model_name = adapter.model_name().to_owned();
            let strategy_id = strategy.id();
            for ((conv_id, summary), raw) in
                valid_ids.iter().zip(summaries.iter()).zip(embeddings.iter())
            {
                let normalized = l2_normalize(raw);
                let index = self.index_clone();
                let cid = conv_id.clone();
                let mn = model_name.clone();
                let sid = strategy_id.clone();
                let sum = summary.clone();
                task::spawn_blocking(move || {
                    index.upsert_conversation_embedding(&cid, &mn, &sid, &normalized, Some(&sum))
                })
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;
                embedded += 1;
            }

            eprintln!("  conversations: {embedded}/{total}");
        }
        Ok(embedded)
    }

    /// Return (unembedded_segments, unembedded_conversations) counts for the
    /// active model and strategy without doing any work. Useful for progress
    /// reporting before calling `embed_all`.
    pub async fn embed_all_pending_counts(&self) -> Result<(u64, u64), StowkenError> {
        let adapter = self
            .embedding_adapter_clone()
            .ok_or_else(|| StowkenError::Internal("no embedding adapter set".into()))?;
        let strategy = self.summary_strategy_clone();
        let model = adapter.model_name().to_owned();
        let strategy_id = strategy.id();
        let seg_count = {
            let index = self.index_clone();
            let m = model.clone();
            task::spawn_blocking(move || index.count_unembedded_segments(&m))
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??
        };
        let conv_count = {
            let index = self.index_clone();
            let m = model.clone();
            let s = strategy_id.clone();
            task::spawn_blocking(move || {
                index.unembedded_conversation_ids(&m, &s, u64::MAX).map(|v| v.len() as u64)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??
        };
        Ok((seg_count, conv_count))
    }

    /// One-shot retroactive index: embed all unembedded segments and all
    /// unembedded conversation summaries, returning combined statistics.
    pub async fn embed_all(&self) -> Result<EmbedStats, StowkenError> {
        let adapter = self
            .embedding_adapter_clone()
            .ok_or_else(|| StowkenError::Internal("no embedding adapter set".into()))?;
        let strategy = self.summary_strategy_clone();
        let model = adapter.model_name().to_owned();
        let strategy_id = strategy.id();

        // Snapshot counts before — anything not unembedded is already done.
        let total_segments = {
            let index = self.index_clone();
            let m = model.clone();
            task::spawn_blocking(move || index.count_unembedded_segments(&m))
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??
        };
        let total_conversations_unembedded = {
            let index = self.index_clone();
            let m = model.clone();
            let s = strategy_id.clone();
            task::spawn_blocking(move || {
                index.unembedded_conversation_ids(&m, &s, u64::MAX).map(|v| v.len() as u64)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??
        };

        info!(
            target: "stowken::semantic",
            unembedded_segments = total_segments,
            unembedded_conversations = total_conversations_unembedded,
            "embed_all: starting retroactive index"
        );

        let segments_embedded = self.embed_all_segments().await?;
        if total_conversations_unembedded > 0 {
            eprintln!("Embedding conversations…");
        }
        let conversations_embedded = self.embed_all_conversations().await?;

        Ok(EmbedStats {
            segments_embedded,
            segments_skipped: total_segments.saturating_sub(segments_embedded),
            segments_already_done: 0,
            conversations_embedded,
            conversations_already_done: 0,
            embedding_model: model,
            summary_strategy: strategy_id,
        })
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Run a semantic search. See `SemanticSearchQuery` for filter options.
    pub async fn semantic_search(
        &self,
        query: SemanticSearchQuery,
    ) -> Result<Vec<SemanticSearchHit>, StowkenError> {
        let adapter = self.embedding_adapter_clone().ok_or_else(|| {
            StowkenError::Internal(
                "semantic_search requires an embedding adapter — call set_embedding_adapter() first".into(),
            )
        })?;
        let strategy_id = self.summary_strategy_clone().id();

        // 1. Embed query.
        let adapter_clone = Arc::clone(&adapter);
        let q_text = query.text.clone();
        let raw_q = task::spawn_blocking(move || adapter_clone.embed_batch(&[q_text.as_str()]))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(StowkenError::Embedding)?;
        if raw_q.is_empty() {
            return Ok(vec![]);
        }
        let query_vec = l2_normalize(&raw_q[0]);

        let limit = query.limit.max(1);
        let mut hits: HashMap<String, SemanticSearchHit> = HashMap::new();

        // 2. Segment-level scan.
        if matches!(
            query.granularity,
            SearchGranularity::Segment | SearchGranularity::Both
        ) {
            let model = adapter.model_name().to_owned();
            let index = self.index_clone();
            let seg_embs = task::spawn_blocking(move || index.load_all_segment_embeddings(&model))
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;
            let scored: Vec<(SegmentHash, f32)> =
                scan_top_k(&query_vec, &seg_embs, limit * 10, query.min_score);
            if !scored.is_empty() {
                let top_hashes: Vec<SegmentHash> = scored.iter().map(|(h, _)| h.clone()).collect();
                let scores: HashMap<SegmentHash, f32> = scored.into_iter().collect();
                let index = self.index_clone();
                let filter_model = query.model.clone();
                let filter_app = query.application.clone();
                let filter_type = query.segment_type.clone();
                let from = query.date_from;
                let to = query.date_to;
                let rows = task::spawn_blocking(move || {
                    index.conversations_for_segments_filtered(
                        &top_hashes,
                        filter_model.as_deref(),
                        filter_app.as_deref(),
                        filter_type.as_ref(),
                        from,
                        to,
                    )
                })
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;

                for row in rows {
                    let score = scores.get(&row.segment_hash).copied().unwrap_or(0.0);
                    upsert_best(
                        &mut hits,
                        SemanticSearchHit {
                            conversation_id: row.conversation_id,
                            score,
                            matched_via: MatchedVia::Segment {
                                hash: row.segment_hash,
                                segment_type: row.segment_type,
                            },
                            application: row.application,
                            model: row.model,
                            created_at: row.created_at,
                        },
                    );
                }
            }
        }

        // 3. Conversation-level scan.
        if matches!(
            query.granularity,
            SearchGranularity::Conversation | SearchGranularity::Both
        ) {
            let model = adapter.model_name().to_owned();
            let strategy = strategy_id.clone();
            let index = self.index_clone();
            let conv_embs = task::spawn_blocking(move || {
                index.load_all_conversation_embeddings(&model, &strategy)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;
            let scored: Vec<(String, f32)> =
                scan_top_k(&query_vec, &conv_embs, limit * 5, query.min_score);
            if !scored.is_empty() {
                let ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
                let scores: HashMap<String, f32> = scored.into_iter().collect();
                let index = self.index_clone();
                let filter_model = query.model.clone();
                let filter_app = query.application.clone();
                let from = query.date_from;
                let to = query.date_to;
                let rows = task::spawn_blocking(move || {
                    index.conversation_meta_filtered(
                        &ids,
                        filter_model.as_deref(),
                        filter_app.as_deref(),
                        from,
                        to,
                    )
                })
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;

                for row in rows {
                    let score = scores.get(&row.conversation_id).copied().unwrap_or(0.0);
                    upsert_best(
                        &mut hits,
                        SemanticSearchHit {
                            conversation_id: row.conversation_id.clone(),
                            score,
                            matched_via: MatchedVia::Conversation,
                            application: row.application,
                            model: row.model,
                            created_at: row.created_at,
                        },
                    );
                }
            }
        }

        let mut results: Vec<SemanticSearchHit> = hits.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    // ── Typed search wrappers ─────────────────────────────────────────────

    /// Search across system prompts only.
    pub async fn search_prompts(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>, StowkenError> {
        let q = SemanticSearchQuery {
            granularity: SearchGranularity::Segment,
            segment_type: Some(SegmentType::SystemPrompt),
            limit,
            ..SemanticSearchQuery::new(text)
        };
        self.semantic_search(q).await
    }

    /// Search across assistant responses only.
    pub async fn search_responses(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>, StowkenError> {
        let q = SemanticSearchQuery {
            granularity: SearchGranularity::Segment,
            segment_type: Some(SegmentType::AssistantTurn),
            limit,
            ..SemanticSearchQuery::new(text)
        };
        self.semantic_search(q).await
    }

    /// Search across user turns only.
    pub async fn search_user_questions(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>, StowkenError> {
        let q = SemanticSearchQuery {
            granularity: SearchGranularity::Segment,
            segment_type: Some(SegmentType::UserTurn),
            limit,
            ..SemanticSearchQuery::new(text)
        };
        self.semantic_search(q).await
    }

    /// Search across tool calls only.
    pub async fn search_tool_uses(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>, StowkenError> {
        let q = SemanticSearchQuery {
            granularity: SearchGranularity::Segment,
            segment_type: Some(SegmentType::ToolCall),
            limit,
            ..SemanticSearchQuery::new(text)
        };
        self.semantic_search(q).await
    }

    /// Search across conversation summaries only.
    pub async fn search_conversations(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchHit>, StowkenError> {
        let q = SemanticSearchQuery {
            granularity: SearchGranularity::Conversation,
            limit,
            ..SemanticSearchQuery::new(text)
        };
        self.semantic_search(q).await
    }

    // ── Clustering & outliers ─────────────────────────────────────────────

    /// Cluster all conversation summaries with k-means++. Returns one
    /// `ConversationCluster` per cluster, with `representative_ids` set to
    /// the top-K conversations closest to the cluster centroid.
    pub async fn cluster_conversations(
        &self,
        k: usize,
        representative_count: usize,
    ) -> Result<Vec<ConversationCluster>, StowkenError> {
        let adapter = self
            .embedding_adapter_clone()
            .ok_or_else(|| StowkenError::Internal("no embedding adapter set".into()))?;
        let strategy_id = self.summary_strategy_clone().id();
        let model = adapter.model_name().to_owned();
        let index = self.index_clone();
        let conv_embs = task::spawn_blocking(move || {
            index.load_all_conversation_embeddings(&model, &strategy_id)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;

        if conv_embs.is_empty() || k == 0 {
            return Ok(vec![]);
        }
        let effective_k = k.min(conv_embs.len());

        let points: Vec<Vec<f32>> = conv_embs.iter().map(|(_, v)| v.clone()).collect();
        let result = task::spawn_blocking(move || {
            kmeans(
                &points,
                KMeansConfig {
                    k: effective_k,
                    max_iterations: 100,
                    seed: 0xC0FFEE,
                },
            )
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))?;

        // Build cluster groups: cluster_id → (members, member_index_in_input)
        let mut by_cluster: Vec<Vec<usize>> = vec![Vec::new(); effective_k];
        for (idx, &c) in result.assignments.iter().enumerate() {
            by_cluster[c as usize].push(idx);
        }

        let mut out = Vec::with_capacity(effective_k);
        for (cluster_id, member_indices) in by_cluster.into_iter().enumerate() {
            if member_indices.is_empty() {
                continue;
            }
            let centroid = &result.centroids[cluster_id];
            let mut scored: Vec<(String, f32)> = member_indices
                .iter()
                .map(|&i| {
                    let (id, emb) = &conv_embs[i];
                    let score: f32 = centroid.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                    (id.clone(), score)
                })
                .collect();
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let representatives: Vec<String> = scored
                .iter()
                .take(representative_count)
                .map(|(id, _)| id.clone())
                .collect();
            let members: Vec<String> = scored.into_iter().map(|(id, _)| id).collect();

            out.push(ConversationCluster {
                cluster_id: cluster_id as u32,
                size: member_indices.len(),
                representative_ids: representatives,
                members,
            });
        }
        Ok(out)
    }

    /// Identify conversations that are far from any cluster centroid —
    /// candidates for "this conversation is unusual" review.
    ///
    /// Uses k-means with `k` clusters; conversations whose distance
    /// (1 - cosine similarity) to the nearest centroid exceeds `threshold`
    /// are returned, sorted by isolation score descending.
    pub async fn find_outliers(
        &self,
        k: usize,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<OutlierConversation>, StowkenError> {
        let adapter = self
            .embedding_adapter_clone()
            .ok_or_else(|| StowkenError::Internal("no embedding adapter set".into()))?;
        let strategy_id = self.summary_strategy_clone().id();
        let model = adapter.model_name().to_owned();
        let index = self.index_clone();
        let conv_embs = task::spawn_blocking(move || {
            index.load_all_conversation_embeddings(&model, &strategy_id)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;
        if conv_embs.is_empty() || k == 0 {
            return Ok(vec![]);
        }
        let effective_k = k.min(conv_embs.len());

        let points: Vec<Vec<f32>> = conv_embs.iter().map(|(_, v)| v.clone()).collect();
        let kmeans_points = points.clone();
        let result = task::spawn_blocking(move || {
            kmeans(
                &kmeans_points,
                KMeansConfig {
                    k: effective_k,
                    max_iterations: 100,
                    seed: 0xC0FFEE,
                },
            )
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))?;

        // Compute distance to nearest centroid for each point.
        let mut scored: Vec<(usize, f32)> = points
            .iter()
            .enumerate()
            .map(|(idx, p)| {
                let best_sim: f32 = result
                    .centroids
                    .iter()
                    .map(|c| c.iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::MIN, f32::max);
                (idx, 1.0 - best_sim)
            })
            .filter(|(_, d)| *d >= threshold)
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);

        if scored.is_empty() {
            return Ok(vec![]);
        }

        // Resolve metadata for the selected conversation IDs.
        let ids: Vec<String> = scored
            .iter()
            .map(|(idx, _)| conv_embs[*idx].0.clone())
            .collect();
        let id_to_score: HashMap<String, f32> = scored
            .into_iter()
            .map(|(idx, d)| (conv_embs[idx].0.clone(), d))
            .collect();
        let index = self.index_clone();
        let rows = task::spawn_blocking(move || {
            index.conversation_meta_filtered(&ids, None, None, None, None)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;

        let mut out: Vec<OutlierConversation> = rows
            .into_iter()
            .map(|row| {
                let score = id_to_score
                    .get(&row.conversation_id)
                    .copied()
                    .unwrap_or(0.0);
                OutlierConversation {
                    conversation_id: row.conversation_id,
                    isolation_score: score,
                    application: row.application,
                    model: row.model,
                    created_at: row.created_at,
                }
            })
            .collect();
        out.sort_by(|a, b| {
            b.isolation_score
                .partial_cmp(&a.isolation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(out)
    }
}

/// Helpers shared across `Stowken<B>` semantic methods. These are exposed
/// as `pub(crate)` accessors on `Stowken<B>` (defined in `vault.rs`) so
/// this module can read the private adapter/strategy/embed-on-store fields
/// without leaking them through the public API.
#[doc(hidden)]
pub(crate) mod plumbing {
    use super::*;
    use crate::index::metadata::MetadataIndex;

    impl<B: StorageBackend + 'static> Stowken<B> {
        pub(crate) fn embedding_adapter_clone(&self) -> Option<Arc<dyn EmbeddingAdapter>> {
            self.embedding_adapter_internal()
        }
        pub(crate) fn summary_strategy_clone(&self) -> SummaryStrategy {
            self.summary_strategy_internal()
        }
        pub(crate) fn embed_on_store_enabled(&self) -> bool {
            self.embed_on_store_internal()
        }
        pub(crate) fn index_clone(&self) -> MetadataIndex {
            self.index_internal()
        }
        pub(crate) fn tokenizer_arc(&self) -> Option<Arc<dyn TokenizerAdapter>> {
            self.tokenizer_internal()
        }
        pub(crate) fn default_tokenizer_name(&self) -> String {
            self.default_tokenizer_name_internal()
        }
        pub(crate) fn detokenize_segment_tokens(
            &self,
            tokens: &[Token],
            tokenizer_name: &str,
        ) -> String {
            self.detokenize_segment_tokens_internal(tokens, tokenizer_name)
        }
        pub(crate) async fn backend_get_segment(
            &self,
            hash: &SegmentHash,
        ) -> Result<StoredSegment, StowkenError> {
            self.backend_get_segment_internal(hash).await
        }
    }
}

// `BatchSegmentOp` re-export used by the embed-on-store hook in vault.rs.
pub(crate) fn collect_new_hashes(ops: &[BatchSegmentOp]) -> Vec<SegmentHash> {
    ops.iter()
        .filter_map(|op| match op {
            BatchSegmentOp::Upsert(s) => Some(s.hash.clone()),
            BatchSegmentOp::IncrementRef(_) => None,
        })
        .collect()
}

