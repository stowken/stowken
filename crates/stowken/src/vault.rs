//! `Stowken` — the main orchestrator.
//!
//! Coordinates segmentation, dedup, compression, storage, and indexing.

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use tokio::task;
use tracing::{debug, info, instrument};

use crate::{
    compression::{CompressionPipeline, FrameVersion},
    dedup::exact,
    dict_registry::{DictError, DictId, DictInfo, DictRegistry},
    index::metadata::{BatchSegmentOp, MetadataError, MetadataIndex},
    near_dedup::{self, MinHashSignature},
    near_dedup_index::NearDupIndex,
    segmenter::{Segmenter, SegmenterConfig, SegmenterError},
    storage::backend::{StorageBackend, StorageError},
    substring_registry::{
        hash_window, SubstringError, SubstringInfo, SubstringRegistry, MIN_LENGTH as SUBSTR_MIN_LEN,
    },
    types::{
        AnalyticsQuery, Conversation, ConversationManifest, ConversationText, ConversationTurn,
        ExportConfig, ExportStats, NearDuplicateCluster, NearDuplicateVariant,
        RetrievedConversation, RetrievedSegment, SegmentRef, SegmentType, SegmentTypeStats,
        StoreResult, StoredSegment, StowkenConfig, SystemPromptInfo, Token, TokenUsageStats,
        TokenizerAdapter,
    },
};

use crate::types::SegmentHash;

// Near-dedup is now two decoupled decisions, each with its own threshold:
//   * `find_best_canonical` surfaces the closest prior segment + similarity.
//   * Storage-delta (`build_delta_frame`) fires at `near_dedup_threshold`.
//   * Embedding-reuse (an alias in the NearDupIndex) fires at the stricter
//     `embed_reuse_threshold`.
// See `StowkenConfig` for why the two thresholds must differ.

/// Substring discovery on a sample of segments. For each MIN_LENGTH-token
/// window, count distinct segments observed it (the same window appearing
/// twice in one segment counts once). Promote windows that occurred in
/// `min_occurrences` or more distinct segments.
///
/// Greedy left-to-right per segment so we don't double-count overlapping
/// matches, and so the resulting promotions favour the highest-frequency
/// canonical position. Emits each promoted window as a `MIN_LENGTH`-token
/// substring; v0.5.x will extend promising windows outward.
///
/// Sync function (called from `tokio::task::spawn_blocking`) so it doesn't
/// hold a runtime thread during the linear scan.
fn discover_substrings(
    samples: &[Vec<Token>],
    min_occurrences: u32,
    registry: &SubstringRegistry,
) -> Result<Vec<SubstringInfo>, StowkenError> {
    use std::collections::HashMap;

    // First pass: count distinct segments per window hash.
    // hash → (occurrences, prototype_tokens)
    let mut counts: HashMap<u64, (u32, Vec<Token>)> = HashMap::new();
    for sample in samples {
        if sample.len() < SUBSTR_MIN_LEN {
            continue;
        }
        // Per-segment dedup: a window seen multiple times in one segment
        // shouldn't inflate the count.
        let mut seen_this_segment: std::collections::HashSet<u64> =
            std::collections::HashSet::new();
        for start in 0..=sample.len() - SUBSTR_MIN_LEN {
            let window = &sample[start..start + SUBSTR_MIN_LEN];
            let h = hash_window(window);
            if !seen_this_segment.insert(h) {
                continue;
            }
            let entry = counts.entry(h).or_insert_with(|| (0, window.to_vec()));
            entry.0 += 1;
        }
    }

    // Second pass: promote windows clearing the threshold. Skip ones that
    // are already in the registry (e.g. from a previous train pass) — the
    // registry's `register` would error, but we'd rather just no-op.
    let mut promoted: Vec<SubstringInfo> = Vec::new();
    let mut sorted: Vec<(u64, u32, Vec<Token>)> = counts
        .into_iter()
        .filter(|(_, (count, _))| *count >= min_occurrences)
        .map(|(h, (count, tokens))| (h, count, tokens))
        .collect();
    // Highest-occurrence first so the most valuable substrings get
    // registered first (and survive if we ever cap the registry).
    sorted.sort_by_key(|t| std::cmp::Reverse(t.1));

    for (_h, count, tokens) in sorted {
        match registry.register(tokens, count) {
            Ok(info) => promoted.push(info),
            // A duplicate window registration shouldn't happen unless two
            // distinct token sequences hash-collide — extremely rare with
            // a 64-bit FNV-style hash on a real corpus, but safe to skip.
            Err(SubstringError::TooShort(_)) => continue,
            Err(SubstringError::Serialization(_)) => continue,
            Err(other) => return Err(StowkenError::Substring(other)),
        }
    }

    Ok(promoted)
}

/// Errors from vault operations.
#[derive(Debug, Error)]
pub enum StowkenError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("metadata index error: {0}")]
    Metadata(#[from] MetadataError),
    #[error("segmentation error: {0}")]
    Segmentation(#[from] SegmenterError),
    #[error("compression error: {0}")]
    Compression(String),
    #[error("dictionary error: {0}")]
    Dict(#[from] DictError),
    #[error("substring registry error: {0}")]
    Substring(#[from] SubstringError),
    #[error("not enough segments to train a dictionary ({0} < {1} required)")]
    InsufficientTrainingSamples(usize, usize),
    #[error("export error: {0}")]
    Export(String),
    #[error("conversation not found: {0}")]
    NotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[cfg(feature = "semantic-search")]
    #[error("embedding error: {0}")]
    Embedding(String),
    #[cfg(feature = "semantic-search")]
    #[error("summarization error: {0}")]
    Summarization(String),
}

/// `Stowken<B>` — the main entry point for all storage and retrieval operations.
///
/// `B` is the storage backend (e.g., `MemoryBackend`, `FilesystemBackend`).
pub struct Stowken<B: StorageBackend> {
    backend: Arc<B>,
    compressor: CompressionPipeline,
    index: MetadataIndex,
    /// Registry of zstd compression dictionaries. In-memory for `Stowken::new`;
    /// filesystem-backed (under `<dirname(db_path)>/dictionaries/`) for `Stowken::open`.
    dict_registry: Arc<DictRegistry>,
    /// Registry of token-level substrings for v0.5 substring dedup.
    /// In-memory for `Stowken::new`, file-backed under
    /// `<dirname(db_path)>/substrings/` for `Stowken::open`.
    substring_registry: Arc<SubstringRegistry>,
    /// MinHash/LSH near-duplicate index + variant→canonical embedding-alias
    /// map. In-memory for `Stowken::new`; append-only logs under
    /// `<dirname(db_path)>/near_dedup/` for `Stowken::open`. Replaces the
    /// former SQLite LSH tables.
    near_dup_index: Arc<NearDupIndex>,
    segmenter_config: SegmenterConfig,
    tokenizer: Option<Arc<dyn TokenizerAdapter>>,
    /// Cached from StowkenConfig — threshold for storing near-duplicate
    /// variants as deltas. None disables delta storage.
    near_dedup_threshold: Option<f64>,
    /// Cached from StowkenConfig — threshold for reusing a canonical's
    /// embedding on a near-duplicate variant. None disables reuse.
    embed_reuse_threshold: Option<f64>,
    #[cfg(feature = "semantic-search")]
    embedding_adapter: Arc<std::sync::RwLock<Option<Arc<dyn crate::types::EmbeddingAdapter>>>>,
    #[cfg(feature = "semantic-search")]
    summary_strategy: Arc<std::sync::RwLock<crate::types::SummaryStrategy>>,
    #[cfg(feature = "semantic-search")]
    embed_on_store: Arc<std::sync::atomic::AtomicBool>,
}

impl<B: StorageBackend + 'static> Stowken<B> {
    /// Create a new vault with an in-memory metadata index and an in-memory
    /// dictionary registry. Dictionaries trained on this vault are NOT
    /// persisted — use `open` for that.
    pub async fn new(backend: B, config: StowkenConfig) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open_in_memory()?;
        let dict_registry = Arc::new(DictRegistry::in_memory());
        let substring_registry = Arc::new(SubstringRegistry::in_memory());
        let near_dup_index = Arc::new(NearDupIndex::in_memory());
        let compressor = CompressionPipeline::new(config.enable_compression, 3)
            .with_registry(Arc::clone(&dict_registry))
            .with_substring_registry(Arc::clone(&substring_registry));
        let near_dedup_threshold = config.near_dedup_threshold;
        let embed_reuse_threshold = config.embed_reuse_threshold;
        Ok(Self {
            backend: Arc::new(backend),
            compressor,
            index,
            dict_registry,
            substring_registry,
            near_dup_index,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
            near_dedup_threshold,
            embed_reuse_threshold,
            #[cfg(feature = "semantic-search")]
            embedding_adapter: Arc::new(std::sync::RwLock::new(None)),
            #[cfg(feature = "semantic-search")]
            summary_strategy: Arc::new(std::sync::RwLock::new(
                crate::types::SummaryStrategy::default(),
            )),
            #[cfg(feature = "semantic-search")]
            embed_on_store: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Open a vault with a persistent metadata database at `db_path`. The
    /// dictionary registry is rooted at `<dirname(db_path)>/dictionaries/`.
    pub async fn open(
        backend: B,
        config: StowkenConfig,
        db_path: &str,
    ) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open(db_path)?;

        let parent = std::path::Path::new(db_path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let dict_registry = Arc::new(DictRegistry::open(parent.join("dictionaries"))?);
        let substring_registry = Arc::new(SubstringRegistry::open(parent.join("substrings"))?);
        let near_dup_index = Arc::new(
            NearDupIndex::open(parent.join("near_dedup"))
                .map_err(|e| StowkenError::Internal(e.to_string()))?,
        );

        let compressor = CompressionPipeline::new(config.enable_compression, 3)
            .with_registry(Arc::clone(&dict_registry))
            .with_substring_registry(Arc::clone(&substring_registry));
        let near_dedup_threshold = config.near_dedup_threshold;
        let embed_reuse_threshold = config.embed_reuse_threshold;
        Ok(Self {
            backend: Arc::new(backend),
            compressor,
            index,
            dict_registry,
            substring_registry,
            near_dup_index,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
            near_dedup_threshold,
            embed_reuse_threshold,
            #[cfg(feature = "semantic-search")]
            embedding_adapter: Arc::new(std::sync::RwLock::new(None)),
            #[cfg(feature = "semantic-search")]
            summary_strategy: Arc::new(std::sync::RwLock::new(
                crate::types::SummaryStrategy::default(),
            )),
            #[cfg(feature = "semantic-search")]
            embed_on_store: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Set a tokenizer adapter for text → token conversion.
    pub fn set_tokenizer(&mut self, tokenizer: impl TokenizerAdapter + 'static) {
        self.tokenizer = Some(Arc::new(tokenizer));
    }

    /// Set the segmenter configuration (e.g., context delimiter).
    pub fn set_segmenter_config(&mut self, cfg: SegmenterConfig) {
        self.segmenter_config = cfg;
    }

    // ── Semantic search configuration ─────────────────────────────────────

    #[cfg(feature = "semantic-search")]
    /// Set the embedding adapter used by `semantic_search`, embed-on-store,
    /// and `embed_all`. Takes `&self` (interior mutability via `RwLock`).
    pub fn set_embedding_adapter(&self, adapter: impl crate::types::EmbeddingAdapter + 'static) {
        *self.embedding_adapter.write().unwrap() = Some(Arc::new(adapter));
    }

    #[cfg(feature = "semantic-search")]
    /// Switch the conversation summary strategy (concat-truncate or LLM-generated).
    pub fn set_summary_strategy(&self, strategy: crate::types::SummaryStrategy) {
        *self.summary_strategy.write().unwrap() = strategy;
    }

    #[cfg(feature = "semantic-search")]
    /// Sugar: install an LLM summarizer. Equivalent to
    /// `set_summary_strategy(SummaryStrategy::LlmGenerated(Arc::new(s)))`.
    pub fn set_summarizer(&self, summarizer: impl crate::types::SummarizerAdapter + 'static) {
        self.set_summary_strategy(crate::types::SummaryStrategy::LlmGenerated(Arc::new(
            summarizer,
        )));
    }

    #[cfg(feature = "semantic-search")]
    /// Sugar: use concat-and-truncate summaries with the given character limit.
    pub fn use_concat_summary(&self, max_chars: usize) {
        self.set_summary_strategy(crate::types::SummaryStrategy::ConcatTruncate { max_chars });
    }

    #[cfg(feature = "semantic-search")]
    /// Toggle whether `store()` automatically embeds new segments and a
    /// conversation summary. Default: true (when an adapter is configured).
    pub fn set_embed_on_store(&self, enabled: bool) {
        self.embed_on_store
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    // ── Internal accessors used by the `semantic` module ──────────────────
    //
    // These exist purely so `crate::semantic` can read private fields
    // without exposing them through the public API. Each is a one-liner.

    #[cfg(feature = "semantic-search")]
    pub(crate) fn embedding_adapter_internal(
        &self,
    ) -> Option<Arc<dyn crate::types::EmbeddingAdapter>> {
        self.embedding_adapter.read().unwrap().clone()
    }
    #[cfg(feature = "semantic-search")]
    pub(crate) fn summary_strategy_internal(&self) -> crate::types::SummaryStrategy {
        self.summary_strategy.read().unwrap().clone()
    }
    #[cfg(feature = "semantic-search")]
    pub(crate) fn embed_on_store_internal(&self) -> bool {
        self.embed_on_store
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(feature = "semantic-search")]
    pub(crate) fn index_internal(&self) -> MetadataIndex {
        self.index.clone()
    }
    #[cfg(feature = "semantic-search")]
    pub(crate) fn near_dup_index_internal(&self) -> Arc<NearDupIndex> {
        Arc::clone(&self.near_dup_index)
    }
    #[cfg(feature = "semantic-search")]
    pub(crate) fn tokenizer_internal(&self) -> Option<Arc<dyn TokenizerAdapter>> {
        self.tokenizer.clone()
    }
    pub(crate) fn default_tokenizer_name_internal(&self) -> String {
        // used by retrieve_batch + semantic
        self.tokenizer
            .as_ref()
            .map(|t| t.name().to_owned())
            .unwrap_or_else(|| "cl100k_base".to_owned())
    }

    /// Detokenize a token slice using the vault's tokenizer or a built-in
    /// fallback for the given tokenizer name.
    pub(crate) fn detokenize_segment_tokens_internal(
        &self,
        tokens: &[Token],
        tokenizer_name: &str,
    ) -> String {
        if let Some(t) = &self.tokenizer {
            t.detokenize(tokens)
        } else if let Some(t) = crate::tokenizer::get_tokenizer(tokenizer_name) {
            t.detokenize(tokens)
        } else {
            tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
    #[cfg(feature = "semantic-search")]
    pub(crate) async fn backend_get_segment_internal(
        &self,
        hash: &SegmentHash,
    ) -> Result<StoredSegment, StowkenError> {
        Ok(self.backend.get_segment(hash).await?)
    }

    // ── Core Operations ───────────────────────────────────────────────────

    /// Store a conversation. Returns the conversation ID and storage metrics.
    #[instrument(skip(self, conversation), fields(model = %conversation.model))]
    pub async fn store(&self, conversation: Conversation) -> Result<StoreResult, StowkenError> {
        let id = conversation
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let segmenter = self.make_segmenter(&conversation.tokenizer);
        let segments = segmenter.segment(&conversation)?;

        let mut segment_refs: Vec<SegmentRef> = Vec::with_capacity(segments.len());
        let mut index_ops: Vec<BatchSegmentOp> = Vec::with_capacity(segments.len());
        let mut new_segments: u64 = 0;
        let mut deduped_segments: u64 = 0;
        let mut bytes_saved: u64 = 0;
        let mut total_compressed: u64 = 0;
        let mut total_raw: u64 = 0;

        // Captured up-front so the per-segment loop can branch without
        // an additional async boundary; cheap copies.
        let near_dedup_threshold = self.near_dedup_threshold;
        let embed_reuse_threshold = self.embed_reuse_threshold;
        // We probe the LSH index when EITHER feature is on. Candidates are
        // surfaced down to the lower of the two thresholds; each feature then
        // applies its own bar to the resulting similarity.
        let probe_min = match (near_dedup_threshold, embed_reuse_threshold) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };

        for (position, segment) in segments.iter().enumerate() {
            let hash = exact::hash_segment(&segment.tokens);

            if self.backend.has_segment(&hash).await? {
                // Exact dedup hit — most efficient path.
                self.backend.increment_ref(&hash).await?;
                index_ops.push(BatchSegmentOp::IncrementRef(hash.clone()));
                deduped_segments += 1;
                bytes_saved += (segment.tokens.len() as u64) * 4;
            } else if let Some(min_threshold) = probe_min {
                // Near-dedup probe: find the closest prior segment, then make
                // two independent decisions — store as a delta, and/or reuse
                // its embedding — each against its own threshold.
                let signature = MinHashSignature::compute(&segment.tokens);
                let best = self
                    .find_best_canonical(&signature, &hash, min_threshold)
                    .await?;

                let mut stored_as_delta = false;
                if let Some((canonical_hash, sim)) = &best {
                    if near_dedup_threshold.is_some_and(|t| *sim >= t) {
                        if let Some(delta_frame) = self
                            .build_delta_frame(canonical_hash, &segment.tokens)
                            .await?
                        {
                            let raw_size = (segment.tokens.len() as u32) * 4;
                            let compressed_size = delta_frame.len() as u32;
                            total_compressed += compressed_size as u64;
                            total_raw += raw_size as u64;

                            let stored = StoredSegment {
                                hash: hash.clone(),
                                segment_type: segment.segment_type.clone(),
                                tokenizer: conversation.tokenizer.clone(),
                                token_count: segment.tokens.len() as u32,
                                compressed_data: delta_frame,
                                raw_size,
                                compressed_size,
                                ref_count: 1,
                                created_at: chrono::Utc::now(),
                            };
                            self.backend.put_segment(&stored).await?;
                            // The variant keeps the canonical alive — bump its
                            // ref so GC won't collect it out from under the delta.
                            self.backend.increment_ref(canonical_hash).await?;
                            index_ops.push(BatchSegmentOp::Upsert(stored));
                            index_ops.push(BatchSegmentOp::IncrementRef(canonical_hash.clone()));
                            new_segments += 1;
                            stored_as_delta = true;
                        }
                    }
                }

                if !stored_as_delta {
                    self.store_new_segment(
                        &hash,
                        segment,
                        &conversation.tokenizer,
                        &mut index_ops,
                        &mut total_compressed,
                        &mut total_raw,
                    )
                    .await?;
                    new_segments += 1;
                }

                // Index this segment so future segments can match it. Reuse
                // the signature we already computed.
                self.index_signature(&hash, &signature).await?;

                // Embedding reuse — a stricter, independent bar. When the
                // variant is close enough to the canonical, alias its
                // embedding so the embed path copies the canonical's vector
                // instead of paying for a fresh call (and the variant stays
                // searchable even when stored as a delta).
                if let Some((canonical_hash, sim)) = &best {
                    if embed_reuse_threshold.is_some_and(|t| *sim >= t) {
                        self.set_embedding_alias(&hash, canonical_hash).await?;
                    }
                }
            } else {
                self.store_new_segment(
                    &hash,
                    segment,
                    &conversation.tokenizer,
                    &mut index_ops,
                    &mut total_compressed,
                    &mut total_raw,
                )
                .await?;
                new_segments += 1;
            }

            segment_refs.push(SegmentRef {
                segment_type: segment.segment_type.clone(),
                hash,
                token_count: segment.tokens.len() as u32,
                position: position as u32,
            });
        }

        let total_tokens: u64 = segment_refs.iter().map(|s| s.token_count as u64).sum();
        let manifest = ConversationManifest {
            schema_version: crate::types::MANIFEST_SCHEMA_VERSION,
            id: id.clone(),
            application: conversation.application.clone(),
            model: conversation.model.clone(),
            tokenizer: conversation.tokenizer.clone(),
            total_tokens,
            segments: segment_refs,
            created_at: chrono::Utc::now(),
            metadata: conversation.metadata.clone(),
        };

        self.backend.put_manifest(&manifest).await?;

        // Snapshot new-segment hashes for the embed-on-store hook (below)
        // before `index_ops` is consumed by the SQLite batch writer.
        #[cfg(feature = "semantic-search")]
        let new_segment_hashes = crate::semantic::collect_new_hashes(&index_ops);

        let index = self.index.clone();
        let manifest_clone = manifest.clone();
        task::spawn_blocking(move || index.store_conversation_batch(&index_ops, &manifest_clone))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

        // Embed-on-store: when an `EmbeddingAdapter` is configured and
        // `embed_on_store` is true (default), embed both the new segments
        // and a per-conversation summary. Idempotent — already-embedded
        // segments are skipped.
        #[cfg(feature = "semantic-search")]
        if let Err(e) = self
            .run_embed_on_store(new_segment_hashes, &id, &conversation.tokenizer)
            .await
        {
            // Don't fail the store on embedding errors — the conversation
            // is already persisted. Log and continue. Callers needing
            // exact-once semantics should set_embed_on_store(false) and
            // run embed_all() explicitly.
            tracing::warn!(id = %id, error = %e, "embed-on-store failed; segment is stored, embedding will be retried by embed_all");
        }

        let compression_ratio = if total_raw == 0 {
            1.0
        } else {
            total_compressed as f64 / total_raw as f64
        };

        let total_segments = new_segments + deduped_segments;
        info!(
            id,
            total_segments, new_segments, deduped_segments, "stored conversation"
        );

        Ok(StoreResult {
            id,
            total_segments,
            new_segments,
            deduped_segments,
            bytes_saved,
            compression_ratio,
        })
    }

    /// Chunk, hash, and compress a conversation without persisting anything.
    ///
    /// Returns segment-level results so the caller can write them to an
    /// external system (e.g. Delta MERGE, an object store batch, a custom
    /// table). Unlike `store`, this method touches neither the storage
    /// backend nor the metadata index, never performs near-dedup (the
    /// caller owns dedup at write time against whatever target it controls),
    /// and is synchronous — all work is CPU-bound.
    ///
    /// The active compression pipeline state (dict + substring registries)
    /// still applies, so `compressed_data` is decompressible only by a
    /// Stowken instance with access to the same registry contents. For
    /// stateless deployments (e.g. ephemeral Spark workers writing to a
    /// shared Delta table), leave the pipeline default-configured so the
    /// output is plain varint+zstd and any reader can decompress it.
    pub fn chunk_only(
        &self,
        conversation: Conversation,
    ) -> Result<crate::types::ChunkResult, StowkenError> {
        let id = conversation
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let segmenter = self.make_segmenter(&conversation.tokenizer);
        let segments = segmenter.segment(&conversation)?;

        let mut out: Vec<crate::types::ChunkedSegment> = Vec::with_capacity(segments.len());
        let mut total_tokens: u64 = 0;

        for (position, segment) in segments.iter().enumerate() {
            let hash = exact::hash_segment(&segment.tokens);
            let compressed_data = self
                .compressor
                .compress(&segment.tokens)
                .map_err(|e| StowkenError::Compression(e.to_string()))?;
            let token_count = segment.tokens.len() as u32;
            total_tokens += token_count as u64;
            out.push(crate::types::ChunkedSegment {
                segment_type: segment.segment_type.clone(),
                hash,
                compressed_data,
                token_count,
                raw_size: token_count * 4,
                position: position as u32,
            });
        }

        Ok(crate::types::ChunkResult {
            id,
            application: conversation.application.clone(),
            model: conversation.model.clone(),
            tokenizer: conversation.tokenizer.clone(),
            total_tokens,
            segments: out,
            metadata: conversation.metadata.clone(),
        })
    }

    /// Retrieve a full conversation with all segments decompressed.
    #[instrument(skip(self))]
    pub async fn retrieve(&self, id: &str) -> Result<RetrievedConversation, StowkenError> {
        let manifest = self.backend.get_manifest(id).await.map_err(|e| match e {
            StorageError::ConversationNotFound(_) => StowkenError::NotFound(id.to_owned()),
            other => StowkenError::Storage(other),
        })?;

        if manifest.schema_version > crate::types::MANIFEST_SCHEMA_VERSION {
            return Err(StowkenError::Internal(format!(
                "manifest {id} schema version {} is newer than supported {}",
                manifest.schema_version,
                crate::types::MANIFEST_SCHEMA_VERSION
            )));
        }

        let mut retrieved_segments = Vec::with_capacity(manifest.segments.len());
        for seg_ref in &manifest.segments {
            let stored = self.backend.get_segment(&seg_ref.hash).await?;
            let tokens = self
                .decompress_segment_async(&stored.compressed_data)
                .await?;
            retrieved_segments.push(RetrievedSegment {
                segment_type: seg_ref.segment_type.clone(),
                hash: seg_ref.hash.clone(),
                tokens,
                token_count: seg_ref.token_count,
                position: seg_ref.position,
            });
        }

        debug!(
            id,
            segments = retrieved_segments.len(),
            "retrieved conversation"
        );
        Ok(RetrievedConversation {
            manifest,
            segments: retrieved_segments,
        })
    }

    /// Retrieve multiple conversations by ID and return them as flat,
    /// human-readable `ConversationText` objects — tokens detokenized and
    /// joined with role markers. Designed to close the search → read loop:
    /// pass the `conversation_id` values from `semantic_search` hits directly.
    ///
    /// IDs that are not found are silently skipped; check the returned length
    /// if you need to detect missing conversations.
    pub async fn retrieve_batch(
        &self,
        ids: &[&str],
    ) -> Result<Vec<ConversationText>, StowkenError> {
        let tokenizer_name = self.default_tokenizer_name_internal();
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let conv = match self.retrieve(id).await {
                Ok(c) => c,
                Err(StowkenError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            let role_for = |seg_type: &SegmentType| -> &'static str {
                match seg_type {
                    SegmentType::SystemPrompt => "system",
                    SegmentType::UserTurn => "user",
                    SegmentType::AssistantTurn => "assistant",
                    SegmentType::ToolCall => "tool_call",
                    SegmentType::ToolResult => "tool_result",
                    SegmentType::Context => "context",
                    SegmentType::Continuation => "continuation",
                }
            };
            let mut turns: Vec<ConversationTurn> = Vec::new();
            for seg in &conv.segments {
                let text = self.detokenize_segment_tokens_internal(&seg.tokens, &tokenizer_name);
                let role = role_for(&seg.segment_type).to_string();
                // Merge consecutive continuation segments into the preceding turn.
                if seg.segment_type == SegmentType::Continuation {
                    if let Some(last) = turns.last_mut() {
                        last.text.push(' ');
                        last.text.push_str(&text);
                        continue;
                    }
                }
                turns.push(ConversationTurn { role, text });
            }
            let full_text = turns
                .iter()
                .map(|t| format!("[{}] {}", t.role, t.text))
                .collect::<Vec<_>>()
                .join("\n");
            results.push(ConversationText {
                conversation_id: conv.manifest.id.clone(),
                model: conv.manifest.model.clone(),
                application: conv.manifest.application.clone(),
                text: full_text,
                turns,
                created_at: conv.manifest.created_at,
            });
        }
        Ok(results)
    }

    /// Retrieve a filtered subset of segments from a conversation.
    pub async fn retrieve_segments(
        &self,
        id: &str,
        segment_type: Option<SegmentType>,
    ) -> Result<Vec<RetrievedSegment>, StowkenError> {
        let conv = self.retrieve(id).await?;
        Ok(conv
            .segments
            .into_iter()
            .filter(|s| segment_type.as_ref().is_none_or(|t| &s.segment_type == t))
            .collect())
    }

    /// Retrieve a single segment by its content hash.
    pub async fn retrieve_segment_by_hash(
        &self,
        hash: &SegmentHash,
    ) -> Result<RetrievedSegment, StowkenError> {
        let stored = self.backend.get_segment(hash).await?;
        let tokens = self
            .decompress_segment_async(&stored.compressed_data)
            .await?;
        Ok(RetrievedSegment {
            segment_type: stored.segment_type,
            hash: hash.clone(),
            token_count: stored.token_count,
            tokens,
            position: 0,
        })
    }

    /// Delete a conversation. Decrements segment ref-counts but does not GC immediately.
    #[instrument(skip(self))]
    pub async fn delete(&self, id: &str) -> Result<(), StowkenError> {
        let manifest = self.backend.get_manifest(id).await.map_err(|e| match e {
            StorageError::ConversationNotFound(_) => StowkenError::NotFound(id.to_owned()),
            other => StowkenError::Storage(other),
        })?;

        for seg_ref in &manifest.segments {
            self.backend.decrement_ref(&seg_ref.hash).await?;
        }
        self.backend.delete_manifest(id).await?;

        let index = self.index.clone();
        let id_owned = id.to_owned();
        task::spawn_blocking(move || index.remove_conversation(&id_owned))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

        info!(id, "deleted conversation");
        Ok(())
    }

    // ── Analytics ─────────────────────────────────────────────────────────

    /// Get overall storage statistics.
    pub async fn stats(&self) -> Result<TokenUsageStats, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.get_stats())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// Get per-segment-type statistics.
    pub async fn segment_stats(&self) -> Result<Vec<SegmentTypeStats>, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.get_segment_type_stats())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// Run a filtered analytics query.
    pub async fn query(
        &self,
        query: AnalyticsQuery,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.query_analytics(&query))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// List all unique system prompts.
    pub async fn list_system_prompts(&self) -> Result<Vec<SystemPromptInfo>, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.list_unique_system_prompts())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// Find conversations that contain a specific segment.
    pub async fn find_by_segment(&self, hash: &SegmentHash) -> Result<Vec<String>, StowkenError> {
        let index = self.index.clone();
        let h = hash.clone();
        task::spawn_blocking(move || index.find_conversations_by_segment(&h))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// Bytes a corpus would occupy if stored as one raw copy per unique
    /// segment (no compression). Used to separate dedup savings from zstd
    /// savings in benchmarks.
    pub async fn dedup_only_bytes(&self) -> Result<u64, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.dedup_only_bytes())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// Find segments referenced at least `min_refs` times — the high-traffic
    /// dedup candidates worth examining for prompt refactoring.
    pub async fn find_duplicates(
        &self,
        min_refs: u64,
        limit: u64,
    ) -> Result<Vec<crate::types::DuplicateSegment>, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.find_duplicates(min_refs, limit))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    // ── Export ────────────────────────────────────────────────────────────

    /// Export training data pairs (user/assistant turns) in the requested format.
    pub async fn export_training_data(
        &self,
        config: ExportConfig,
        writer: &mut dyn std::io::Write,
    ) -> Result<ExportStats, StowkenError> {
        crate::export::training::export_jsonl(self, &config, writer)
            .await
            .map_err(|e| StowkenError::Export(e.to_string()))
    }

    // ── Maintenance ───────────────────────────────────────────────────────

    /// Run garbage collection on orphaned segments (ref_count == 0).
    ///
    /// Also prunes the collected segments from the near-dup index: a stale
    /// signature for a deleted canonical would otherwise be surfaced as a
    /// delta target and fail when `build_delta_frame` fetches it, breaking a
    /// later ingest. Canonicals with live variants keep a non-zero ref count
    /// (each variant bumps it), so they aren't collected out from under a
    /// delta in the first place.
    pub async fn gc(&self) -> Result<u64, StowkenError> {
        // Snapshot the eligible hashes before collection — `garbage_collect`
        // returns only a count. In a single-writer vault these are exactly
        // the segments about to be removed.
        let collected = self.backend.list_garbage().await?;
        let deleted = self.backend.garbage_collect().await?;

        if !collected.is_empty() {
            let index = Arc::clone(&self.near_dup_index);
            let to_prune = collected.clone();
            task::spawn_blocking(move || {
                for hash in &to_prune {
                    index.remove(hash)?;
                }
                Ok::<(), crate::near_dedup_index::NearDupIndexError>(())
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(|e| StowkenError::Internal(e.to_string()))?;
        }

        info!(
            deleted,
            pruned = collected.len(),
            "garbage collected segments"
        );
        Ok(deleted)
    }

    /// List the segment hashes currently eligible for GC, without deleting.
    pub async fn gc_candidates(&self) -> Result<Vec<SegmentHash>, StowkenError> {
        Ok(self.backend.list_garbage().await?)
    }

    // ── Near-duplicate analytics ──────────────────────────────────────────

    /// Cluster heavily-referenced segments together with their nearest
    /// duplicates. For each segment with `ref_count >= min_refs`, scans
    /// the LSH index for candidates and returns those whose Jaccard
    /// similarity to the canonical exceeds `threshold`.
    ///
    /// Useful for prompt-drift audits: surfaces "you have 47 system
    /// prompts that are 90%+ similar to this one — should they be one?".
    /// Read-only; does not change storage.
    pub async fn find_near_duplicates(
        &self,
        threshold: f64,
        min_refs: u64,
        max_clusters: u64,
    ) -> Result<Vec<NearDuplicateCluster>, StowkenError> {
        let candidates = self.find_duplicates(min_refs, max_clusters).await?;
        let mut clusters = Vec::new();

        for canonical in candidates {
            // Need the canonical's tokens to compute a signature if it isn't
            // already in the index.
            let sig = match self.near_dup_index.signature(&canonical.hash) {
                Some(s) => s,
                None => {
                    let canonical_stored = self.backend.get_segment(&canonical.hash).await?;
                    let canonical_tokens = self
                        .decompress_segment_async(&canonical_stored.compressed_data)
                        .await?;
                    MinHashSignature::compute(&canonical_tokens)
                }
            };
            let band_hashes: Vec<u64> = (0..near_dedup::BANDS).map(|b| sig.band_hash(b)).collect();

            let cand_hashes = self.near_dup_index.candidates(&band_hashes);
            let sig_map = self.near_dup_index.signatures_batch(&cand_hashes);

            let mut variants = Vec::new();
            for c in cand_hashes {
                if c == canonical.hash {
                    continue;
                }
                let Some(other_sig) = sig_map.get(&c) else {
                    continue;
                };
                let similarity = sig.jaccard(other_sig);
                if similarity >= threshold {
                    let stored = self.backend.get_segment(&c).await?;
                    variants.push(NearDuplicateVariant {
                        hash: c,
                        similarity,
                        token_count: stored.token_count,
                        ref_count: stored.ref_count,
                    });
                }
            }

            if !variants.is_empty() {
                variants.sort_by(|a, b| {
                    b.similarity
                        .partial_cmp(&a.similarity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                clusters.push(NearDuplicateCluster {
                    canonical: canonical.hash,
                    canonical_token_count: canonical.token_count,
                    canonical_ref_count: canonical.ref_count,
                    variants,
                });
            }
        }

        Ok(clusters)
    }

    /// Number of segments tracked in the near-dup index. Useful for sanity
    /// checks and post-reindex confirmation.
    pub async fn near_dedup_index_size(&self) -> Result<u64, StowkenError> {
        Ok(self.near_dup_index.len())
    }

    // ── Substring management (v0.5) ────────────────────────────────────────

    /// Retroactively rewrite pre-train segments to use the substring
    /// registry. Walks every unique segment in the index, decompresses
    /// it, tries `0x05` encoding, and replaces the stored bytes if the
    /// substring frame is smaller. Skips `0x04` (delta) and `0x05`
    /// (already-substring) frames.
    ///
    /// Atomic per-segment via the backend's `replace_segment_data`. A
    /// crash mid-compaction leaves each segment in either old or new
    /// form — both decompressible. The metadata index's
    /// `compressed_size` is updated after each replace, so stats
    /// remain consistent.
    pub async fn compact_substrings(
        &self,
    ) -> Result<crate::types::SubstringCompactStats, StowkenError> {
        if self.substring_registry.is_empty() {
            return Ok(crate::types::SubstringCompactStats {
                segments_examined: 0,
                segments_rewritten: 0,
                bytes_saved: 0,
                segments_skipped: 0,
            });
        }

        let hashes = self.list_all_segment_hashes().await?;

        let mut examined = 0u64;
        let mut rewritten = 0u64;
        let mut skipped = 0u64;
        let mut bytes_saved: u64 = 0;

        for hash in hashes {
            examined += 1;
            let stored = match self.backend.get_segment(&hash).await {
                Ok(s) => s,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };

            // Skip 0x04 (delta) and 0x05 (already substring-encoded)
            // frames. 0x04 frames hold edit scripts against another
            // canonical, not native tokens — re-encoding them as 0x05
            // would be conceptually wrong. 0x05 frames are already in
            // their final form.
            let first_byte = match stored.compressed_data.first() {
                Some(&b) => b,
                None => {
                    skipped += 1;
                    continue;
                }
            };
            if first_byte == FrameVersion::Delta as u8
                || first_byte == FrameVersion::Substring as u8
            {
                skipped += 1;
                continue;
            }

            // Decompress to native tokens and try the substring path.
            let tokens = self
                .decompress_segment_async(&stored.compressed_data)
                .await?;
            let baseline_len = stored.compressed_data.len();
            let new_frame = match self
                .compressor
                .try_compress_substring_frame(&tokens, baseline_len)
                .map_err(|e| StowkenError::Compression(e.to_string()))?
            {
                Some(frame) => frame,
                None => continue, // not a win — leave the segment alone
            };

            let saved = (baseline_len - new_frame.len()) as u64;
            let new_size = new_frame.len() as u32;
            self.backend.replace_segment_data(&hash, new_frame).await?;

            // Keep the metadata index in sync with the on-disk size.
            let index = self.index.clone();
            let h = hash.clone();
            task::spawn_blocking(move || index.update_segment_compressed_size(&h, new_size))
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;

            rewritten += 1;
            bytes_saved += saved;
        }

        info!(
            examined,
            rewritten, skipped, bytes_saved, "compacted segments to use substring registry"
        );
        Ok(crate::types::SubstringCompactStats {
            segments_examined: examined,
            segments_rewritten: rewritten,
            bytes_saved,
            segments_skipped: skipped,
        })
    }

    /// Drop substrings from the registry that are no longer referenced
    /// by any `0x05` frame in the backend. Walks every segment, decodes
    /// any `0x05` frame to collect referenced substring ids, then
    /// retains only those in the registry.
    pub async fn gc_substrings(&self) -> Result<crate::types::SubstringGcStats, StowkenError> {
        let before = self.substring_registry.len() as u64;
        if before == 0 {
            return Ok(crate::types::SubstringGcStats {
                registry_size_before: 0,
                registry_size_after: 0,
                substrings_dropped: 0,
            });
        }

        // Collect every substring id referenced by a 0x05 frame.
        let hashes = self.list_all_segment_hashes().await?;
        let mut referenced: std::collections::HashSet<crate::substring_registry::SubstringId> =
            std::collections::HashSet::new();
        for hash in hashes {
            let stored = match self.backend.get_segment(&hash).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            if stored.compressed_data.first() != Some(&(FrameVersion::Substring as u8)) {
                continue;
            }
            // Body starts at byte 1 (skip version tag).
            let ops = match crate::compression::decode_substring_frame(&stored.compressed_data[1..])
            {
                Ok(o) => o,
                Err(_) => continue,
            };
            for op in ops {
                if let crate::compression::SubstringOp::Ref(id) = op {
                    referenced.insert(id);
                }
            }
        }

        let dropped = self.substring_registry.retain(&referenced)?;
        let after = self.substring_registry.len() as u64;

        info!(
            before,
            after, dropped, "garbage-collected substring registry"
        );
        Ok(crate::types::SubstringGcStats {
            registry_size_before: before,
            registry_size_after: after,
            substrings_dropped: dropped,
        })
    }

    /// Helper: every distinct segment hash known to the metadata index.
    async fn list_all_segment_hashes(&self) -> Result<Vec<SegmentHash>, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.all_segment_hashes())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
    }

    /// Result of a `train_substrings` pass.
    pub fn substring_registry(&self) -> &Arc<SubstringRegistry> {
        &self.substring_registry
    }

    /// All known substrings, in registration order.
    pub fn list_substrings(&self) -> Vec<SubstringInfo> {
        self.substring_registry.list()
    }

    /// Discover candidate substrings in `sample_count` recent segments,
    /// filter by occurrence threshold, and register the survivors. Returns
    /// the list of newly-registered substrings.
    ///
    /// Discovery is a fixed-window scan: every `MIN_LENGTH`-token window
    /// across the sample is hashed; windows occurring in at least
    /// `min_occurrences` distinct segments are promoted as substrings.
    ///
    /// New segments stored after this call automatically use the
    /// registered substrings if the resulting `0x05` frame is smaller
    /// than the baseline.
    pub async fn train_substrings(
        &self,
        sample_count: usize,
        min_occurrences: u32,
    ) -> Result<Vec<SubstringInfo>, StowkenError> {
        let samples = self.collect_substring_samples(sample_count).await?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // For each MIN_LENGTH-token window, count distinct segments
        // observed it. We dedup within a single segment (a window seen
        // 5x in one segment counts as 1 occurrence) by collecting per-
        // segment hashes into a set first.
        let registry = Arc::clone(&self.substring_registry);
        let report =
            task::spawn_blocking(move || discover_substrings(&samples, min_occurrences, &registry))
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;

        info!(promoted = report.len(), "trained substring registry");
        Ok(report)
    }

    /// Pull at most `count` recent segments (decompressed) for substring
    /// discovery. Skips delta segments — their tokens are derivable but
    /// they're already a near-dedup variant, so promoting substrings from
    /// them would double-count.
    async fn collect_substring_samples(
        &self,
        count: usize,
    ) -> Result<Vec<Vec<Token>>, StowkenError> {
        let query = AnalyticsQuery::default();
        let ids = self
            .backend
            .list_conversations(&query, count as u64, 0)
            .await?;

        let mut samples: Vec<Vec<Token>> = Vec::with_capacity(count);
        for id in &ids {
            if samples.len() >= count {
                break;
            }
            let manifest = match self.backend.get_manifest(id).await {
                Ok(m) => m,
                Err(_) => continue,
            };
            for seg_ref in &manifest.segments {
                if samples.len() >= count {
                    break;
                }
                let stored = match self.backend.get_segment(&seg_ref.hash).await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if stored.compressed_data.first() == Some(&(FrameVersion::Delta as u8)) {
                    continue;
                }
                let tokens = match self.decompress_segment_async(&stored.compressed_data).await {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if tokens.len() >= SUBSTR_MIN_LEN {
                    samples.push(tokens);
                }
            }
        }
        Ok(samples)
    }

    // ── Dictionary management ──────────────────────────────────────────────

    /// Minimum samples required to train a dictionary. The zstd builder
    /// itself wants at least a handful; below this we'd produce a dict that
    /// generalises poorly and isn't worth the frame overhead.
    pub const MIN_TRAINING_SAMPLES: usize = 10;

    /// Train a new zstd dictionary from `sample_count` segments pulled from
    /// the backend (most-recent conversations first), register it, and return
    /// the new dict's metadata. The dict is NOT activated — call
    /// `activate_dictionary` to have new compressions use it.
    ///
    /// Use `train_and_activate_dictionary` for the common case.
    pub async fn train_dictionary(&self, sample_count: usize) -> Result<DictInfo, StowkenError> {
        let samples = self.collect_training_samples(sample_count).await?;
        if samples.len() < Self::MIN_TRAINING_SAMPLES {
            return Err(StowkenError::InsufficientTrainingSamples(
                samples.len(),
                Self::MIN_TRAINING_SAMPLES,
            ));
        }

        // 110 KiB is the zstd-recommended dict size for token-shaped data.
        let dict_bytes = task::spawn_blocking(move || {
            let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
            zstd::dict::from_samples(&refs, 112_640)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))?
        .map_err(|e| StowkenError::Compression(format!("dict training: {e}")))?;

        let actual_samples = sample_count as u32;
        let info = self.dict_registry.register(dict_bytes, actual_samples)?;
        info!(
            dict_id = info.id,
            samples = actual_samples,
            "trained dictionary"
        );
        Ok(info)
    }

    /// Activate `dict_id` so new compressions use it. Old frames remain
    /// readable as long as their dictionary stays in the registry.
    pub async fn activate_dictionary(&self, dict_id: DictId) -> Result<(), StowkenError> {
        self.dict_registry.activate(dict_id)?;
        info!(dict_id, "activated dictionary");
        Ok(())
    }

    /// Convenience: train a dictionary and immediately activate it.
    pub async fn train_and_activate_dictionary(
        &self,
        sample_count: usize,
    ) -> Result<DictInfo, StowkenError> {
        let info = self.train_dictionary(sample_count).await?;
        self.activate_dictionary(info.id).await?;
        Ok(DictInfo {
            is_active: true,
            ..info
        })
    }

    /// All known dictionaries, sorted by creation time. Each entry's
    /// `is_active` flag indicates which is the writer.
    pub fn list_dictionaries(&self) -> Vec<DictInfo> {
        self.dict_registry.list()
    }

    /// Read access to the dictionary registry — exposed for advanced use
    /// (e.g. the CLI inspector). Most callers should use the methods above.
    pub fn dict_registry(&self) -> &Arc<DictRegistry> {
        &self.dict_registry
    }

    /// Pull at most `count` varint-encoded segment payloads from the backend
    /// for use as dictionary training samples. We decompress each segment
    /// and re-encode just the varint stream — the zstd dict-builder works
    /// best on the data that will *actually* be compressed in production,
    /// which is the varint output (not the zstd output, not the raw tokens).
    async fn collect_training_samples(&self, count: usize) -> Result<Vec<Vec<u8>>, StowkenError> {
        let query = AnalyticsQuery::default();
        let ids = self
            .backend
            .list_conversations(&query, count as u64, 0)
            .await?;

        let mut samples: Vec<Vec<u8>> = Vec::with_capacity(count);
        for id in &ids {
            if samples.len() >= count {
                break;
            }
            let manifest = match self.backend.get_manifest(id).await {
                Ok(m) => m,
                Err(_) => continue,
            };
            for seg_ref in &manifest.segments {
                if samples.len() >= count {
                    break;
                }
                let stored = match self.backend.get_segment(&seg_ref.hash).await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let tokens = match self.decompress_data(&stored.compressed_data) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                samples.push(crate::compression::varint::encode_tokens(&tokens));
            }
        }
        Ok(samples)
    }

    /// Rebuild the SQLite metadata index from scratch by walking every
    /// manifest in the backend. Manifests are the source of truth; the
    /// index is a derived cache.
    ///
    /// Use this after a corrupted or deleted `metadata.db`, after restoring
    /// a vault from a backup that didn't include the index, or to reconcile
    /// the index when you suspect drift.
    ///
    /// Does not touch backend ref-counts. Reindex makes the SQL state match
    /// what the manifests actually reference; running `gc` afterwards will
    /// clean up any orphaned segments in the backend whose ref-counts
    /// drifted independently.
    pub async fn reindex(&self) -> Result<crate::types::ReindexStats, StowkenError> {
        use std::collections::HashSet;

        // Wipe the existing index in one transaction.
        let index = self.index.clone();
        task::spawn_blocking(move || index.clear())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

        // Replay every manifest through the same batch path that store() uses.
        let query = AnalyticsQuery::default();
        let ids = self.backend.list_conversations(&query, u64::MAX, 0).await?;

        let mut seen: HashSet<SegmentHash> = HashSet::new();
        let mut conversations_indexed = 0u64;
        let mut unique_segments_indexed = 0u64;
        let mut segments_missing = 0u64;
        // Track unique non-delta segments (LSH signatures) and delta variants
        // (embedding aliases) so we can rebuild the near-dup index after the
        // SQL-state pass — kept separate so batch commits aren't interleaved
        // with index writes.
        let mut lsh_targets: Vec<SegmentHash> = Vec::new();
        let mut delta_variants: Vec<SegmentHash> = Vec::new();

        for id in &ids {
            let manifest = match self.backend.get_manifest(id).await {
                Ok(m) => m,
                Err(_) => continue,
            };

            let mut index_ops: Vec<BatchSegmentOp> = Vec::with_capacity(manifest.segments.len());

            for seg_ref in &manifest.segments {
                if seen.insert(seg_ref.hash.clone()) {
                    // First time seeing this hash this run — read its meta
                    // from the backend so the segments_meta row gets the
                    // correct compressed_size, raw_size, etc.
                    match self.backend.get_segment(&seg_ref.hash).await {
                        Ok(stored) => {
                            // Schedule for LSH reindex if this is a real
                            // (non-delta) frame — only canonicals get
                            // signatures.
                            let is_delta = stored.compressed_data.first()
                                == Some(&(crate::compression::FrameVersion::Delta as u8));
                            if is_delta {
                                delta_variants.push(seg_ref.hash.clone());
                            } else {
                                lsh_targets.push(seg_ref.hash.clone());
                            }
                            // Force ref_count=1 in the BatchSegmentOp::Upsert
                            // payload; subsequent IncrementRef ops will bump
                            // it as we encounter further references.
                            let mut s = stored;
                            s.ref_count = 1;
                            index_ops.push(BatchSegmentOp::Upsert(s));
                            unique_segments_indexed += 1;
                        }
                        Err(_) => {
                            segments_missing += 1;
                            continue;
                        }
                    }
                } else {
                    index_ops.push(BatchSegmentOp::IncrementRef(seg_ref.hash.clone()));
                }
            }

            let index = self.index.clone();
            let manifest_clone = manifest.clone();
            task::spawn_blocking(move || {
                index.store_conversation_batch(&index_ops, &manifest_clone)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

            conversations_indexed += 1;
        }

        // Phase 2: rebuild the near-dup signature index from scratch. The
        // packed logs are derived state, so we clear and replay them from the
        // manifests (the source of truth). Done unconditionally so the index
        // is correct even when near-dedup is currently disabled —
        // re-enabling later won't require another reindex.
        self.near_dup_index
            .clear()
            .map_err(|e| StowkenError::Internal(e.to_string()))?;
        for hash in &lsh_targets {
            let stored = match self.backend.get_segment(hash).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let tokens = match self.decompress_segment_async(&stored.compressed_data).await {
                Ok(t) => t,
                Err(_) => continue,
            };
            let sig = MinHashSignature::compute(&tokens);
            self.index_signature(hash, &sig).await?;
        }

        // Phase 3: rebuild embedding aliases for delta variants. Only when
        // embedding reuse is configured — otherwise aliases stay empty,
        // matching "feature off". A delta frame names its canonical directly;
        // we recompute the similarity to apply the (stricter) reuse bar. This
        // does not reconstruct aliases for full-frame variants stored under
        // embed-reuse-without-near-dedup; those are rebuilt on next ingest.
        let mut aliases_rebuilt = 0u64;
        if let Some(reuse_threshold) = self.embed_reuse_threshold {
            for variant in &delta_variants {
                let stored = match self.backend.get_segment(variant).await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let canonical = match self
                    .compressor
                    .frame_canonical_hash(&stored.compressed_data)
                {
                    Ok(Some(c)) => c,
                    _ => continue,
                };
                let variant_tokens =
                    match self.decompress_segment_async(&stored.compressed_data).await {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                let variant_sig = MinHashSignature::compute(&variant_tokens);
                let Some(canonical_sig) = self.near_dup_index.signature(&canonical) else {
                    continue;
                };
                if variant_sig.jaccard(&canonical_sig) >= reuse_threshold {
                    self.set_embedding_alias(variant, &canonical).await?;
                    aliases_rebuilt += 1;
                }
            }
        }

        info!(
            conversations = conversations_indexed,
            segments = unique_segments_indexed,
            missing = segments_missing,
            lsh = lsh_targets.len(),
            aliases = aliases_rebuilt,
            "reindexed metadata from manifests"
        );

        Ok(crate::types::ReindexStats {
            conversations_indexed,
            unique_segments_indexed,
            segments_missing,
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Compress `segment` and store it as a fresh blob (no near-dedup).
    /// Tracking counters are updated in place; the BatchSegmentOp is pushed
    /// into `index_ops`. Caller increments `new_segments`.
    async fn store_new_segment(
        &self,
        hash: &SegmentHash,
        segment: &crate::types::Segment,
        tokenizer: &str,
        index_ops: &mut Vec<BatchSegmentOp>,
        total_compressed: &mut u64,
        total_raw: &mut u64,
    ) -> Result<(), StowkenError> {
        // Always start with the baseline frame (0x02 / 0x03 / 0x01).
        let baseline = self.compress_tokens(&segment.tokens)?;
        // Try the substring-frame path; keep it only if the result beats
        // the baseline.
        let compressed = match self
            .compressor
            .try_compress_substring_frame(&segment.tokens, baseline.len())
            .map_err(|e| StowkenError::Compression(e.to_string()))?
        {
            Some(substring_frame) => substring_frame,
            None => baseline,
        };

        let raw_size = (segment.tokens.len() as u32) * 4;
        let compressed_size = compressed.len() as u32;
        *total_compressed += compressed_size as u64;
        *total_raw += raw_size as u64;

        let stored = StoredSegment {
            hash: hash.clone(),
            segment_type: segment.segment_type.clone(),
            tokenizer: tokenizer.to_owned(),
            token_count: segment.tokens.len() as u32,
            compressed_data: compressed,
            raw_size,
            compressed_size,
            ref_count: 1,
            created_at: chrono::Utc::now(),
        };
        self.backend.put_segment(&stored).await?;
        index_ops.push(BatchSegmentOp::Upsert(stored));
        Ok(())
    }

    /// Find the closest prior segment to `signature` in the near-dup index,
    /// returning `(canonical_hash, similarity)` for the best candidate whose
    /// estimated Jaccard meets `min_threshold`. Candidate lookup and scoring
    /// are pure in-memory operations against the [`NearDupIndex`].
    async fn find_best_canonical(
        &self,
        signature: &MinHashSignature,
        variant_hash: &SegmentHash,
        min_threshold: f64,
    ) -> Result<Option<(SegmentHash, f64)>, StowkenError> {
        let band_hashes: Vec<u64> = (0..near_dedup::BANDS)
            .map(|b| signature.band_hash(b))
            .collect();

        let candidates: Vec<SegmentHash> = self
            .near_dup_index
            .candidates(&band_hashes)
            .into_iter()
            .filter(|h| h != variant_hash)
            .collect();
        if candidates.is_empty() {
            return Ok(None);
        }

        let sig_map = self.near_dup_index.signatures_batch(&candidates);
        let mut best: Option<(SegmentHash, f64)> = None;
        for candidate_hash in candidates {
            let Some(candidate_sig) = sig_map.get(&candidate_hash) else {
                continue;
            };
            let sim = signature.jaccard(candidate_sig);
            if sim >= min_threshold && best.as_ref().is_none_or(|(_, prev)| sim > *prev) {
                best = Some((candidate_hash, sim));
            }
        }
        Ok(best)
    }

    /// Build a `0x04` delta frame for `tokens` against `canonical_hash`.
    /// Returns `None` when the canonical is itself a delta (we never chain
    /// deltas) or when the delta isn't smaller than a fresh full compression.
    async fn build_delta_frame(
        &self,
        canonical_hash: &SegmentHash,
        tokens: &[Token],
    ) -> Result<Option<Vec<u8>>, StowkenError> {
        let canonical_stored = self.backend.get_segment(canonical_hash).await?;
        if canonical_stored.compressed_data.first() == Some(&(FrameVersion::Delta as u8)) {
            return Ok(None);
        }
        let canonical_tokens = self
            .compressor
            .decompress(&canonical_stored.compressed_data)
            .map_err(|e| StowkenError::Compression(e.to_string()))?;

        let ops = near_dedup::compute_delta(&canonical_tokens, tokens);
        let frame = self
            .compressor
            .compress_delta(canonical_hash, &ops)
            .map_err(|e| StowkenError::Compression(e.to_string()))?;

        // Only accept the delta if it's actually smaller than a fresh full
        // compression. Otherwise we'd pay for the cross-segment ref (extra
        // ref bump, can't GC canonical) for negative gain.
        let full_size = self.compress_tokens(tokens)?.len();
        if frame.len() >= full_size {
            return Ok(None);
        }
        Ok(Some(frame))
    }

    /// Persist a segment's signature into the near-dup index so future
    /// incoming segments can match it. The signature is precomputed by the
    /// caller (the store path already has it).
    async fn index_signature(
        &self,
        hash: &SegmentHash,
        signature: &MinHashSignature,
    ) -> Result<(), StowkenError> {
        let index = Arc::clone(&self.near_dup_index);
        let h = hash.clone();
        let sig = signature.clone();
        task::spawn_blocking(move || index.insert(&h, &sig))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(|e| StowkenError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Record that `variant` should reuse `canonical`'s embedding.
    async fn set_embedding_alias(
        &self,
        variant: &SegmentHash,
        canonical: &SegmentHash,
    ) -> Result<(), StowkenError> {
        let index = Arc::clone(&self.near_dup_index);
        let v = variant.clone();
        let c = canonical.clone();
        task::spawn_blocking(move || index.set_embedding_alias(&v, &c))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(|e| StowkenError::Internal(e.to_string()))?;
        Ok(())
    }

    fn make_segmenter(&self, tokenizer_name: &str) -> Segmenter {
        let seg = Segmenter::new(self.segmenter_config.clone());
        if let Some(ref adapter) = self.tokenizer {
            seg.with_tokenizer(Arc::clone(adapter))
        } else if let Some(adapter) = crate::tokenizer::get_tokenizer(tokenizer_name) {
            seg.with_tokenizer(Arc::from(adapter))
        } else {
            seg
        }
    }

    /// Compress a token sequence using the active pipeline.
    pub fn compress_tokens(&self, tokens: &[Token]) -> Result<Vec<u8>, StowkenError> {
        self.compressor
            .compress(tokens)
            .map_err(|e| StowkenError::Compression(e.to_string()))
    }

    /// Decompress stored bytes back to a token sequence using the active
    /// pipeline. Synchronous: cannot resolve `0x04` (delta) frames since
    /// resolving the canonical requires backend I/O. Use
    /// `decompress_segment_async` for retrieval paths.
    pub fn decompress_data(&self, data: &[u8]) -> Result<Vec<Token>, StowkenError> {
        self.compressor
            .decompress(data)
            .map_err(|e| StowkenError::Compression(e.to_string()))
    }

    /// Decompress a segment blob back to text using the named tokenizer.
    ///
    /// Convenience wrapper for callers that hold compressed bytes (e.g.
    /// rows from an external `segments` table) and want readable text
    /// without composing `decompress_data` + a tokenizer adapter manually.
    /// Synchronous; cannot resolve `0x04` delta frames (same limitation as
    /// `decompress_data`).
    pub fn decompress_to_text(
        &self,
        data: &[u8],
        tokenizer_name: &str,
    ) -> Result<String, StowkenError> {
        let tokens = self.decompress_data(data)?;
        let adapter = crate::tokenizer::get_tokenizer(tokenizer_name).ok_or_else(|| {
            StowkenError::Internal(format!("unknown tokenizer: {tokenizer_name}"))
        })?;
        Ok(adapter.detokenize(&tokens))
    }

    /// Decompress any frame, including `0x04` deltas. For deltas, the
    /// canonical is fetched from the backend and the delta applied.
    /// Used by `retrieve` and the LSH reindex path.
    ///
    /// Box the future because decoding a delta against a canonical that is
    /// itself a delta would otherwise create an infinitely-sized async
    /// state machine. (We currently refuse to chain deltas, but the
    /// boxing keeps the code-path safe even if that policy ever changes.)
    pub fn decompress_segment_async<'a>(
        &'a self,
        data: &'a [u8],
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<Token>, StowkenError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let canonical_hash = self
                .compressor
                .frame_canonical_hash(data)
                .map_err(|e| StowkenError::Compression(e.to_string()))?;

            match canonical_hash {
                None => self.decompress_data(data),
                Some(canonical_hash) => {
                    let canonical_stored = self.backend.get_segment(&canonical_hash).await?;
                    let canonical_tokens = self
                        .decompress_segment_async(&canonical_stored.compressed_data)
                        .await?;

                    // Frame body = 1 byte version + 32 byte hash + delta ops.
                    let delta_bytes = &data[33..];
                    let ops = near_dedup::decode_delta(delta_bytes)
                        .map_err(|e| StowkenError::Compression(e.to_string()))?;
                    near_dedup::apply_delta(&canonical_tokens, &ops)
                        .map_err(|e| StowkenError::Compression(e.to_string()))
                }
            }
        })
    }

    /// Read access to the storage backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Read access to the metadata index.
    pub fn index(&self) -> &MetadataIndex {
        &self.index
    }
}
