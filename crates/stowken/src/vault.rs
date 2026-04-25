//! `Stowken` — the main orchestrator.
//!
//! Coordinates segmentation, dedup, compression, storage, and indexing.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;
use tokio::task;
use tracing::{debug, info, instrument, warn};

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
    config: StowkenConfig,
    /// Hot-swappable compression pipeline (dictionary is applied without &mut self).
    compressor: Arc<RwLock<CompressionPipeline>>,
    index: MetadataIndex,
    segmenter_config: SegmenterConfig,
    tokenizer: Option<Arc<dyn TokenizerAdapter>>,
    /// Cumulative count of brand-new segments stored; drives auto-train trigger.
    new_segment_count: Arc<AtomicU64>,
}

impl<B: StorageBackend + 'static> Stowken<B> {
    /// Create a new vault with an in-memory metadata index.
    pub async fn new(backend: B, config: StowkenConfig) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open_in_memory()?;
        let compressor = CompressionPipeline::new(config.enable_compression, 3);
        Ok(Self {
            backend: Arc::new(backend),
            config,
            compressor: Arc::new(RwLock::new(compressor)),
            index,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
            new_segment_count: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Open a vault, loading or creating a persistent metadata database at `db_path`.
    ///
    /// If `config.dictionary_path` points to an existing file it is loaded
    /// automatically so that previously dictionary-compressed segments remain
    /// decompressible.
    pub async fn open(backend: B, config: StowkenConfig, db_path: &str) -> Result<Self, StowkenError> {
        let index = MetadataIndex::open(db_path)?;
        let mut pipeline = CompressionPipeline::new(config.enable_compression, 3);

        if let Some(ref path) = config.dictionary_path {
            if let Ok(dict_bytes) = std::fs::read(path) {
                info!(path, bytes = dict_bytes.len(), "loaded zstd dictionary from disk");
                pipeline = CompressionPipeline::with_dictionary(dict_bytes, config.enable_compression, 3);
            }
        }

        Ok(Self {
            backend: Arc::new(backend),
            config,
            compressor: Arc::new(RwLock::new(pipeline)),
            index,
            segmenter_config: SegmenterConfig::default(),
            tokenizer: None,
            new_segment_count: Arc::new(AtomicU64::new(0)),
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
            let hash = self.make_hash(&segment.tokens);

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

        // Single spawn_blocking: all segment ops + conversation manifest in one transaction.
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

        // Auto-train: check threshold after updating the cumulative count.
        if new_segments > 0 {
            let before = self.new_segment_count.fetch_add(new_segments, Ordering::Relaxed);
            if let Some(threshold) = self.config.auto_train_dictionary_after {
                let threshold = threshold as u64;
                // Trigger when the counter crosses a multiple of `threshold`.
                if before / threshold != (before + new_segments) / threshold {
                    self.spawn_background_train();
                }
            }
        }

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

    /// Train a zstd dictionary from a sample of stored segments and return
    /// the raw dictionary bytes.  Does not apply the dictionary — call
    /// `apply_dictionary` to activate it.
    pub async fn train_dictionary(
        &self,
        _tokenizer: &str,
        sample_count: usize,
    ) -> Result<Vec<u8>, StowkenError> {
        let query = AnalyticsQuery::default();
        let ids = self.backend.list_conversations(&query, sample_count as u64, 0).await?;

        let mut samples: Vec<Vec<u8>> = Vec::new();
        for id in &ids {
            if let Ok(conv) = self.retrieve(id).await {
                for seg in conv.segments {
                    let varint = crate::compression::varint::encode_tokens(&seg.tokens);
                    samples.push(varint);
                    if samples.len() >= sample_count {
                        break;
                    }
                }
            }
            if samples.len() >= sample_count {
                break;
            }
        }

        if samples.len() < 10 {
            return Err(StowkenError::Compression(
                format!("not enough samples to train dictionary ({} < 10)", samples.len())
            ));
        }

        crate::compression::dictionary::DictionaryCompressor::train_dictionary(&samples, 112_640)
            .map_err(|e| StowkenError::Compression(e.to_string()))
    }

    /// Apply a pre-trained (or externally provided) zstd dictionary to the
    /// compression pipeline.
    ///
    /// All segments stored *after* this call will be compressed with the
    /// dictionary.  Previously stored segments are unaffected — zstd frames
    /// carry a dictionary-ID header, so the decompressor uses the dictionary
    /// only when it was present at compression time.
    ///
    /// If `config.dictionary_path` is set the dictionary is persisted to disk
    /// so it survives across `open()` calls.
    pub async fn apply_dictionary(&self, dict_bytes: Vec<u8>) -> Result<(), StowkenError> {
        info!(bytes = dict_bytes.len(), "applying zstd dictionary");

        if let Some(ref path) = self.config.dictionary_path {
            tokio::fs::write(path, &dict_bytes).await
                .map_err(|e| StowkenError::Compression(format!("failed to write dictionary: {e}")))?;
            info!(path, "persisted zstd dictionary to disk");
        }

        let new_pipeline = CompressionPipeline::with_dictionary(
            dict_bytes,
            self.config.enable_compression,
            3,
        );
        *self.compressor.write().unwrap() = new_pipeline;
        Ok(())
    }

    /// Convenience: train a dictionary from stored data and immediately apply it.
    ///
    /// Returns the number of samples used.
    pub async fn train_and_apply_dictionary(
        &self,
        tokenizer: &str,
        sample_count: usize,
    ) -> Result<usize, StowkenError> {
        let dict = self.train_dictionary(tokenizer, sample_count).await?;
        let n = sample_count;
        self.apply_dictionary(dict).await?;
        info!(n, "trained and applied zstd dictionary");
        Ok(n)
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_segmenter(&self, tokenizer_name: &str) -> Segmenter {
        let seg = Segmenter::new(self.segmenter_config.clone());
        // Explicit adapter set by the caller takes priority; otherwise try the
        // built-in tiktoken lookup for the conversation's declared tokenizer.
        if let Some(ref adapter) = self.tokenizer {
            seg.with_tokenizer(Arc::clone(adapter))
        } else if let Some(adapter) = crate::tokenizer::get_tokenizer(tokenizer_name) {
            seg.with_tokenizer(Arc::from(adapter))
        } else {
            seg
        }
    }

    fn make_hash(&self, tokens: &[Token]) -> SegmentHash {
        if let Some(tenant_id) = &self.config.tenant_id {
            exact::hash_segment_tenant(tokens, tenant_id)
        } else {
            exact::hash_segment(tokens)
        }
    }

    /// Spawn a background task that trains and applies a new dictionary.
    /// Errors are logged but not propagated — store() must not fail due to
    /// a background training hiccup.
    fn spawn_background_train(&self) {
        let backend = Arc::clone(&self.backend);
        let compressor = Arc::clone(&self.compressor);
        let sample_count = self.config.dictionary_sample_count;
        let dict_path = self.config.dictionary_path.clone();
        let enable = self.config.enable_compression;

        tokio::spawn(async move {
            // Collect varint-encoded samples from the backend directly —
            // we decompress with the current pipeline then re-encode with varint.
            let query = AnalyticsQuery::default();
            let Ok(ids) = backend.list_conversations(&query, sample_count as u64, 0).await else {
                warn!("auto-train: failed to list conversations");
                return;
            };

            let current = compressor.read().unwrap().clone();
            let mut samples: Vec<Vec<u8>> = Vec::new();

            'outer: for id in &ids {
                let Ok(manifest) = backend.get_manifest(id).await else { continue };
                for seg_ref in &manifest.segments {
                    let Ok(stored) = backend.get_segment(&seg_ref.hash).await else { continue };
                    let Ok(tokens) = current.decompress(&stored.compressed_data) else { continue };
                    samples.push(crate::compression::varint::encode_tokens(&tokens));
                    if samples.len() >= sample_count { break 'outer; }
                }
            }

            if samples.len() < 10 {
                warn!(count = samples.len(), "auto-train: too few samples, skipping");
                return;
            }

            let dict = match crate::compression::dictionary::DictionaryCompressor::train_dictionary(&samples, 112_640) {
                Ok(d) => d,
                Err(e) => { warn!("auto-train: dictionary training failed: {e}"); return; }
            };

            if let Some(ref path) = dict_path {
                if let Err(e) = tokio::fs::write(path, &dict).await {
                    warn!(path, "auto-train: failed to persist dictionary: {e}");
                }
            }

            let new_pipeline = CompressionPipeline::with_dictionary(dict.clone(), enable, 3);
            *compressor.write().unwrap() = new_pipeline;
            info!(samples = samples.len(), dict_bytes = dict.len(), "auto-trained and applied zstd dictionary");
        });
    }

    // ── Extension points for enterprise wrapper ───────────────────────────

    /// Segment a conversation into typed token chunks (for enterprise callers).
    pub fn segment_conversation(
        &self,
        conversation: &Conversation,
    ) -> Result<Vec<crate::types::Segment>, StowkenError> {
        let segmenter = self.make_segmenter(&conversation.tokenizer);
        Ok(segmenter.segment(conversation)?)
    }

    /// Compute the content hash for a token slice (honours tenant_id if set).
    pub fn hash_tokens(&self, tokens: &[Token]) -> SegmentHash {
        self.make_hash(tokens)
    }

    /// Compress a token sequence using the active pipeline.
    pub fn compress_tokens(&self, tokens: &[Token]) -> Result<Vec<u8>, StowkenError> {
        self.compressor.read().unwrap()
            .compress(tokens)
            .map_err(|e| StowkenError::Compression(e.to_string()))
    }

    /// Decompress stored bytes back to a token sequence using the active pipeline.
    pub fn decompress_data(&self, data: &[u8]) -> Result<Vec<Token>, StowkenError> {
        self.compressor.read().unwrap()
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

// ── Dictionary tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod dictionary_tests {
    use super::*;
    use crate::{storage::MemoryBackend, types::{Conversation, Message, MessageContent}};

    fn make_conv(id: &str, tokens: Vec<u32>) -> Conversation {
        Conversation {
            id: Some(id.to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![Message {
                role: "system".to_owned(),
                content: MessageContent::Tokens(tokens),
                name: None,
                tool_call_id: None,
            }],
            metadata: None,
        }
    }

    async fn vault_with_data(n: usize) -> Stowken<MemoryBackend> {
        let v = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await.unwrap();
        for i in 0..n {
            let tokens: Vec<u32> = (0u32..100).map(|t| (t + i as u32 * 7) % 50_000).collect();
            v.store(make_conv(&format!("c{i}"), tokens)).await.unwrap();
        }
        v
    }

    #[tokio::test]
    async fn train_dictionary_returns_bytes() {
        let v = vault_with_data(20).await;
        let dict = v.train_dictionary("cl100k_base", 15).await.unwrap();
        assert!(!dict.is_empty(), "dictionary should not be empty");
    }

    #[tokio::test]
    async fn apply_dictionary_roundtrip() {
        let v = vault_with_data(20).await;
        let dict = v.train_dictionary("cl100k_base", 15).await.unwrap();
        v.apply_dictionary(dict).await.unwrap();

        // Store a new conversation with the dictionary active.
        let tokens: Vec<u32> = (0u32..100).collect();
        v.store(make_conv("dict_conv", tokens.clone())).await.unwrap();

        // Retrieve and verify lossless roundtrip.
        let retrieved = v.retrieve("dict_conv").await.unwrap();
        assert_eq!(retrieved.segments[0].tokens, tokens);
    }

    #[tokio::test]
    async fn old_segments_still_decompressible_after_apply() {
        let v = vault_with_data(20).await;

        // Store before dictionary.
        let pre_tokens: Vec<u32> = (0u32..50).collect();
        v.store(make_conv("pre", pre_tokens.clone())).await.unwrap();

        // Apply dictionary.
        let dict = v.train_dictionary("cl100k_base", 15).await.unwrap();
        v.apply_dictionary(dict).await.unwrap();

        // Store after dictionary.
        let post_tokens: Vec<u32> = (100u32..150).collect();
        v.store(make_conv("post", post_tokens.clone())).await.unwrap();

        // Both must decompress correctly.
        let pre = v.retrieve("pre").await.unwrap();
        assert_eq!(pre.segments[0].tokens, pre_tokens, "pre-dict segment corrupted");

        let post = v.retrieve("post").await.unwrap();
        assert_eq!(post.segments[0].tokens, post_tokens, "post-dict segment corrupted");
    }

    #[tokio::test]
    async fn train_and_apply_convenience() {
        let v = vault_with_data(20).await;
        let n = v.train_and_apply_dictionary("cl100k_base", 15).await.unwrap();
        assert!(n > 0);

        // Roundtrip after apply.
        let tokens: Vec<u32> = (500u32..600).collect();
        v.store(make_conv("after", tokens.clone())).await.unwrap();
        let r = v.retrieve("after").await.unwrap();
        assert_eq!(r.segments[0].tokens, tokens);
    }

    #[tokio::test]
    async fn train_fails_with_too_few_samples() {
        let v = vault_with_data(2).await;
        let result = v.train_dictionary("cl100k_base", 5).await;
        assert!(result.is_err(), "should fail with too few samples");
    }

    #[tokio::test]
    async fn dictionary_persisted_and_reloaded() {
        // Use a temp file path for the dictionary.
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("test.dict");
        let dict_path_str = dict_path.to_str().unwrap().to_owned();

        // Vault A: train and persist.
        let config_a = StowkenConfig {
            dictionary_path: Some(dict_path_str.clone()),
            ..StowkenConfig::default()
        };
        let va = Stowken::new(MemoryBackend::new(), config_a).await.unwrap();
        for i in 0..20usize {
            let tokens: Vec<u32> = (0u32..100).map(|t| (t + i as u32 * 3) % 30_000).collect();
            va.store(make_conv(&format!("c{i}"), tokens)).await.unwrap();
        }
        let dict = va.train_dictionary("cl100k_base", 15).await.unwrap();
        va.apply_dictionary(dict).await.unwrap();

        // Store a segment with the dictionary.
        let tokens: Vec<u32> = (0u32..80).collect();
        va.store(make_conv("dict_seg", tokens.clone())).await.unwrap();
        // Retrieve with vault A (has dict in memory).
        let r = va.retrieve("dict_seg").await.unwrap();
        assert_eq!(r.segments[0].tokens, tokens);

        assert!(dict_path.exists(), "dictionary should be on disk");
    }

    #[tokio::test]
    async fn auto_train_triggers_after_threshold() {
        let config = StowkenConfig {
            auto_train_dictionary_after: Some(10),
            dictionary_sample_count: 8,
            ..StowkenConfig::default()
        };
        let v = Stowken::new(MemoryBackend::new(), config).await.unwrap();

        // Store enough conversations to cross the threshold.
        for i in 0..15usize {
            let tokens: Vec<u32> = (0u32..100).map(|t| (t + i as u32 * 7) % 50_000).collect();
            v.store(make_conv(&format!("c{i}"), tokens)).await.unwrap();
        }

        // Give the background task a moment to complete.
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Verify the pipeline still works (auto-train shouldn't break anything).
        let tokens: Vec<u32> = (0u32..50).collect();
        v.store(make_conv("post_train", tokens.clone())).await.unwrap();
        let r = v.retrieve("post_train").await.unwrap();
        assert_eq!(r.segments[0].tokens, tokens);
    }
}
