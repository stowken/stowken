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
    segmenter::{Segmenter, SegmenterConfig, SegmenterError},
    storage::backend::{StorageBackend, StorageError},
    types::{
        AnalyticsQuery, Conversation, ConversationManifest, ExportConfig, ExportStats,
        NearDuplicateCluster, NearDuplicateVariant, RetrievedConversation, RetrievedSegment,
        SegmentRef, SegmentType, SegmentTypeStats, StoreResult, StoredSegment, SystemPromptInfo,
        Token, TokenUsageStats, TokenizerAdapter, StowkenConfig,
    },
};

use crate::types::SegmentHash;

/// Internal: output of a successful near-dedup probe.
struct NearDedupOutcome {
    canonical_hash: SegmentHash,
    delta_frame: Vec<u8>,
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
    #[error("not enough segments to train a dictionary ({0} < {1} required)")]
    InsufficientTrainingSamples(usize, usize),
    #[error("export error: {0}")]
    Export(String),
    #[error("conversation not found: {0}")]
    NotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
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
    segmenter_config: SegmenterConfig,
    tokenizer: Option<Arc<dyn TokenizerAdapter>>,
    /// Cached from StowkenConfig — threshold for near-dedup. None disables.
    near_dedup_threshold: Option<f64>,
}

impl<B: StorageBackend + 'static> Stowken<B> {
    /// Create a new vault with an in-memory metadata index and an in-memory
    /// dictionary registry. Dictionaries trained on this vault are NOT
    /// persisted — use `open` for that.
    pub async fn new(backend: B, config: StowkenConfig) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open_in_memory()?;
        let dict_registry = Arc::new(DictRegistry::in_memory());
        let compressor = CompressionPipeline::new(config.enable_compression, 3)
            .with_registry(Arc::clone(&dict_registry));
        let near_dedup_threshold = config.near_dedup_threshold;
        Ok(Self {
            backend: Arc::new(backend),
            compressor,
            index,
            dict_registry,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
            near_dedup_threshold,
        })
    }

    /// Open a vault with a persistent metadata database at `db_path`. The
    /// dictionary registry is rooted at `<dirname(db_path)>/dictionaries/`.
    pub async fn open(backend: B, config: StowkenConfig, db_path: &str) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open(db_path)?;

        let dict_root = std::path::Path::new(db_path)
            .parent()
            .map(|p| p.join("dictionaries"))
            .unwrap_or_else(|| std::path::PathBuf::from("dictionaries"));
        let dict_registry = Arc::new(DictRegistry::open(&dict_root)?);

        let compressor = CompressionPipeline::new(config.enable_compression, 3)
            .with_registry(Arc::clone(&dict_registry));
        let near_dedup_threshold = config.near_dedup_threshold;
        Ok(Self {
            backend: Arc::new(backend),
            compressor,
            index,
            dict_registry,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
            near_dedup_threshold,
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
        // an additional async boundary; cheap copy.
        let near_dedup_threshold = self.near_dedup_threshold;

        for (position, segment) in segments.iter().enumerate() {
            let hash = exact::hash_segment(&segment.tokens);

            if self.backend.has_segment(&hash).await? {
                // Exact dedup hit — most efficient path.
                self.backend.increment_ref(&hash).await?;
                index_ops.push(BatchSegmentOp::IncrementRef(hash.clone()));
                deduped_segments += 1;
                bytes_saved += (segment.tokens.len() as u64) * 4;
            } else if let Some(threshold) = near_dedup_threshold {
                // Near-dedup probe. If a matching canonical exists, store
                // the variant as a 0x04 delta frame.
                match self
                    .try_near_dedup(&hash, &segment.tokens, threshold)
                    .await?
                {
                    Some(NearDedupOutcome { canonical_hash, delta_frame }) => {
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
                        // The variant's existence keeps the canonical alive
                        // — bump the canonical's ref so GC doesn't collect
                        // it out from under the delta.
                        self.backend.increment_ref(&canonical_hash).await?;
                        index_ops.push(BatchSegmentOp::Upsert(stored));
                        index_ops.push(BatchSegmentOp::IncrementRef(canonical_hash));
                        new_segments += 1;

                        // Index this variant in the LSH tables too — variants
                        // can themselves be canonicals for future variants.
                        self.upsert_lsh_entry(&hash, &segment.tokens).await?;
                    }
                    None => {
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
                        // Add to LSH so future segments can find this one.
                        self.upsert_lsh_entry(&hash, &segment.tokens).await?;
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

        let index = self.index.clone();
        let manifest_clone = manifest.clone();
        task::spawn_blocking(move || index.store_conversation_batch(&index_ops, &manifest_clone))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

        let compression_ratio = if total_raw == 0 {
            1.0
        } else {
            total_compressed as f64 / total_raw as f64
        };

        let total_segments = new_segments + deduped_segments;
        info!(id, total_segments, new_segments, deduped_segments, "stored conversation");

        Ok(StoreResult {
            id,
            total_segments,
            new_segments,
            deduped_segments,
            bytes_saved,
            compression_ratio,
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
            let tokens = self.decompress_segment_async(&stored.compressed_data).await?;
            retrieved_segments.push(RetrievedSegment {
                segment_type: seg_ref.segment_type.clone(),
                hash: seg_ref.hash.clone(),
                tokens,
                token_count: seg_ref.token_count,
                position: seg_ref.position,
            });
        }

        debug!(id, segments = retrieved_segments.len(), "retrieved conversation");
        Ok(RetrievedConversation {
            manifest,
            segments: retrieved_segments,
        })
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
        let tokens = self.decompress_segment_async(&stored.compressed_data).await?;
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
    pub async fn gc(&self) -> Result<u64, StowkenError> {
        let deleted = self.backend.garbage_collect().await?;
        info!(deleted, "garbage collected segments");
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
            // Need the canonical's tokens to compute a signature. Avoid
            // doing this inside the SQL critical path.
            let canonical_stored = self.backend.get_segment(&canonical.hash).await?;
            let canonical_tokens = self
                .decompress_segment_async(&canonical_stored.compressed_data)
                .await?;
            let sig = MinHashSignature::compute(&canonical_tokens);
            let band_hashes: Vec<u64> =
                (0..near_dedup::BANDS).map(|b| sig.band_hash(b)).collect();

            let index = self.index.clone();
            let bh = band_hashes.clone();
            let cand_hashes = task::spawn_blocking(move || {
                index.find_near_dedup_candidates(&bh)
            })
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))??;

            let mut variants = Vec::new();
            for c in cand_hashes {
                if c == canonical.hash {
                    continue;
                }
                let index = self.index.clone();
                let h = c.clone();
                let sig_bytes_opt =
                    task::spawn_blocking(move || index.get_near_dedup_signature(&h))
                        .await
                        .map_err(|e| StowkenError::Internal(e.to_string()))??;
                let other_sig = match sig_bytes_opt {
                    Some(b) => MinHashSignature::from_bytes(&b)
                        .map_err(|e| StowkenError::Internal(format!("LSH signature corrupt: {e}")))?,
                    None => continue,
                };
                let similarity = sig.jaccard(&other_sig);
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
                variants.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
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

    /// Number of segments tracked in the LSH index. Useful for sanity
    /// checks and post-reindex confirmation.
    pub async fn near_dedup_index_size(&self) -> Result<u64, StowkenError> {
        let index = self.index.clone();
        task::spawn_blocking(move || index.near_dedup_size())
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(Into::into)
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
        info!(dict_id = info.id, samples = actual_samples, "trained dictionary");
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
        Ok(DictInfo { is_active: true, ..info })
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
        let ids = self.backend.list_conversations(&query, count as u64, 0).await?;

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
        let ids = self
            .backend
            .list_conversations(&query, u64::MAX, 0)
            .await?;

        let mut seen: HashSet<SegmentHash> = HashSet::new();
        let mut conversations_indexed = 0u64;
        let mut unique_segments_indexed = 0u64;
        let mut segments_missing = 0u64;
        // Track all unique non-delta segments so we can replay the LSH index
        // pass after the SQL-state pass. Done in two phases because the
        // batch-insert path commits per-conversation and we don't want LSH
        // upserts interleaved with that.
        let mut lsh_targets: Vec<SegmentHash> = Vec::new();

        for id in &ids {
            let manifest = match self.backend.get_manifest(id).await {
                Ok(m) => m,
                Err(_) => continue,
            };

            let mut index_ops: Vec<BatchSegmentOp> =
                Vec::with_capacity(manifest.segments.len());

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
                            if !is_delta {
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

        // Phase 2: rebuild the LSH index for non-delta segments. Done
        // unconditionally so the index is correct even when near-dedup
        // is currently disabled — re-enabling later won't require another
        // reindex.
        for hash in &lsh_targets {
            let stored = match self.backend.get_segment(hash).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let tokens = match self.decompress_segment_async(&stored.compressed_data).await {
                Ok(t) => t,
                Err(_) => continue,
            };
            self.upsert_lsh_entry(hash, &tokens).await?;
        }

        info!(
            conversations = conversations_indexed,
            segments = unique_segments_indexed,
            missing = segments_missing,
            lsh = lsh_targets.len(),
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
        let compressed = self.compress_tokens(&segment.tokens)?;
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

    /// Probe the LSH index for a near-duplicate canonical and, if found,
    /// build a 0x04 delta frame for `tokens`. Returns `None` if no
    /// candidate clears the threshold.
    async fn try_near_dedup(
        &self,
        variant_hash: &SegmentHash,
        tokens: &[Token],
        threshold: f64,
    ) -> Result<Option<NearDedupOutcome>, StowkenError> {
        let signature = MinHashSignature::compute(tokens);
        let band_hashes: Vec<u64> =
            (0..near_dedup::BANDS).map(|b| signature.band_hash(b)).collect();

        // Pull LSH candidates from SQLite.
        let index = self.index.clone();
        let band_hashes_clone = band_hashes.clone();
        let candidates = task::spawn_blocking(move || {
            index.find_near_dedup_candidates(&band_hashes_clone)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;

        // Verify each candidate's actual Jaccard similarity.
        let mut best: Option<(SegmentHash, f64)> = None;
        for candidate_hash in candidates {
            // A candidate matching its own hash is a no-op (shouldn't
            // happen since variant_hash is by definition not yet stored).
            if &candidate_hash == variant_hash {
                continue;
            }
            let index = self.index.clone();
            let h = candidate_hash.clone();
            let sig_bytes_opt = task::spawn_blocking(move || index.get_near_dedup_signature(&h))
                .await
                .map_err(|e| StowkenError::Internal(e.to_string()))??;
            let candidate_sig = match sig_bytes_opt {
                Some(bytes) => MinHashSignature::from_bytes(&bytes)
                    .map_err(|e| StowkenError::Internal(format!("LSH signature corrupt: {e}")))?,
                None => continue,
            };
            let sim = signature.jaccard(&candidate_sig);
            if sim >= threshold && best.as_ref().is_none_or(|(_, prev)| sim > *prev) {
                best = Some((candidate_hash, sim));
            }
        }

        let Some((canonical_hash, _sim)) = best else { return Ok(None) };

        // Materialise the canonical's tokens to compute a delta against.
        // Bypass our own decompress path's 0x04 handler — we only delta
        // against true canonicals, never against existing deltas.
        let canonical_stored = self.backend.get_segment(&canonical_hash).await?;
        if canonical_stored.compressed_data.first() == Some(&(FrameVersion::Delta as u8)) {
            // The "canonical" we matched is itself a delta. Skip — don't
            // chain deltas. (Future v0.4.x optimization: hop through the
            // chain to find the root canonical.)
            return Ok(None);
        }
        let canonical_tokens = self.compressor.decompress(&canonical_stored.compressed_data)
            .map_err(|e| StowkenError::Compression(e.to_string()))?;

        let ops = near_dedup::compute_delta(&canonical_tokens, tokens);
        let frame = self.compressor.compress_delta(&canonical_hash, &ops)
            .map_err(|e| StowkenError::Compression(e.to_string()))?;

        // Only accept the delta if it's actually smaller than a fresh full
        // compression. Otherwise we'd be paying for the cross-segment ref
        // (extra ref bump, can't GC canonical) for negative gain.
        let full_size = self.compress_tokens(tokens)?.len();
        if frame.len() >= full_size {
            return Ok(None);
        }

        Ok(Some(NearDedupOutcome { canonical_hash, delta_frame: frame }))
    }

    /// Persist a segment's signature + LSH band hashes so future incoming
    /// segments can find this one as a candidate.
    async fn upsert_lsh_entry(
        &self,
        hash: &SegmentHash,
        tokens: &[Token],
    ) -> Result<(), StowkenError> {
        let signature = MinHashSignature::compute(tokens);
        let sig_bytes = signature.to_bytes();
        let band_hashes: Vec<u64> =
            (0..near_dedup::BANDS).map(|b| signature.band_hash(b)).collect();

        let index = self.index.clone();
        let h = hash.clone();
        task::spawn_blocking(move || {
            index.upsert_near_dedup_entry(&h, &sig_bytes, &band_hashes)
        })
        .await
        .map_err(|e| StowkenError::Internal(e.to_string()))??;
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
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Token>, StowkenError>> + Send + 'a>>
    {
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
