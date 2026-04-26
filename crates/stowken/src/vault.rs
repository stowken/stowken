//! `Stowken` — the main orchestrator.
//!
//! Coordinates segmentation, dedup, compression, storage, and indexing.

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use tokio::task;
use tracing::{debug, info, instrument};

use crate::{
    compression::CompressionPipeline,
    dedup::exact,
    index::metadata::{BatchSegmentOp, MetadataError, MetadataIndex},
    segmenter::{Segmenter, SegmenterConfig, SegmenterError},
    storage::backend::{StorageBackend, StorageError},
    types::{
        AnalyticsQuery, Conversation, ConversationManifest, ExportConfig, ExportStats,
        RetrievedConversation, RetrievedSegment, SegmentRef, SegmentType, SegmentTypeStats,
        StoreResult, StoredSegment, SystemPromptInfo, Token, TokenUsageStats, TokenizerAdapter,
        StowkenConfig,
    },
};

use crate::types::SegmentHash;

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
    segmenter_config: SegmenterConfig,
    tokenizer: Option<Arc<dyn TokenizerAdapter>>,
}

impl<B: StorageBackend + 'static> Stowken<B> {
    /// Create a new vault with an in-memory metadata index.
    pub async fn new(backend: B, config: StowkenConfig) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open_in_memory()?;
        let compressor = CompressionPipeline::new(config.enable_compression, 3);
        Ok(Self {
            backend: Arc::new(backend),
            compressor,
            index,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
        })
    }

    /// Open a vault, loading or creating a persistent metadata database at `db_path`.
    pub async fn open(backend: B, config: StowkenConfig, db_path: &str) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open(db_path)?;
        let compressor = CompressionPipeline::new(config.enable_compression, 3);
        Ok(Self {
            backend: Arc::new(backend),
            compressor,
            index,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
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

        for (position, segment) in segments.iter().enumerate() {
            let hash = exact::hash_segment(&segment.tokens);

            if self.backend.has_segment(&hash).await? {
                self.backend.increment_ref(&hash).await?;
                index_ops.push(BatchSegmentOp::IncrementRef(hash.clone()));
                deduped_segments += 1;
                bytes_saved += (segment.tokens.len() as u64) * 4;
            } else {
                let compressed = self.compress_tokens(&segment.tokens)?;

                let raw_size = (segment.tokens.len() as u32) * 4;
                let compressed_size = compressed.len() as u32;
                total_compressed += compressed_size as u64;
                total_raw += raw_size as u64;

                let stored = StoredSegment {
                    hash: hash.clone(),
                    segment_type: segment.segment_type.clone(),
                    tokenizer: conversation.tokenizer.clone(),
                    token_count: segment.tokens.len() as u32,
                    compressed_data: compressed,
                    raw_size,
                    compressed_size,
                    ref_count: 1,
                    created_at: chrono::Utc::now(),
                };
                self.backend.put_segment(&stored).await?;
                index_ops.push(BatchSegmentOp::Upsert(stored));
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
            let tokens = self.decompress_data(&stored.compressed_data)?;
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
        let tokens = self.decompress_data(&stored.compressed_data)?;
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

        info!(
            conversations = conversations_indexed,
            segments = unique_segments_indexed,
            missing = segments_missing,
            "reindexed metadata from manifests"
        );

        Ok(crate::types::ReindexStats {
            conversations_indexed,
            unique_segments_indexed,
            segments_missing,
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────

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

    /// Decompress stored bytes back to a token sequence using the active pipeline.
    pub fn decompress_data(&self, data: &[u8]) -> Result<Vec<Token>, StowkenError> {
        self.compressor
            .decompress(data)
            .map_err(|e| StowkenError::Compression(e.to_string()))
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
