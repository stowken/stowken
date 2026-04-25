//! `StorageBackend` — the pluggable persistence abstraction.
//!
//! Stowken handles compression, dedup, and segmentation.
//! Backends only store and retrieve bytes + manifests.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::types::{
    AnalyticsQuery, ConversationManifest, SegmentHash, SegmentTypeStats, StoredSegment,
    TokenUsageStats,
};

/// Errors that any storage backend can produce.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("segment not found: {0}")]
    SegmentNotFound(String),
    #[error("conversation not found: {0}")]
    ConversationNotFound(String),
    #[error("backend error: {0}")]
    BackendError(String),
    #[error("serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type StorageResult<T> = Result<T, StorageError>;

/// Pluggable storage backend trait.
///
/// All segment content is content-addressed by SHA-256 hash.
/// Backends are responsible for ref-counting; Stowken is responsible for
/// deciding when to increment or decrement refs.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    // ── Segment operations ────────────────────────────────────────────────

    /// Store a compressed segment blob.
    ///
    /// Idempotent: if the hash already exists, increment `ref_count` and return `Ok`.
    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()>;

    /// Retrieve a compressed segment by hash.
    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment>;

    /// Check segment existence without fetching data.
    async fn has_segment(&self, hash: &SegmentHash) -> StorageResult<bool>;

    /// Batch existence check. Returns the subset of `hashes` that already exist.
    async fn has_segments(&self, hashes: &[SegmentHash]) -> StorageResult<Vec<SegmentHash>>;

    /// Increment the reference count of an existing segment.
    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()>;

    /// Decrement the reference count. Returns `true` if `ref_count` reaches 0
    /// (segment is eligible for garbage collection).
    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool>;

    /// Delete a segment. Only called during GC when `ref_count == 0`.
    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()>;

    // ── Manifest operations ───────────────────────────────────────────────

    /// Store a conversation manifest.
    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()>;

    /// Retrieve a conversation manifest by ID.
    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest>;

    /// Delete a conversation manifest.
    ///
    /// Does NOT delete segments; the caller is responsible for ref-counting.
    async fn delete_manifest(&self, id: &str) -> StorageResult<()>;

    /// List conversation IDs with optional filters and pagination.
    async fn list_conversations(
        &self,
        query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> StorageResult<Vec<String>>;

    // ── Analytics ─────────────────────────────────────────────────────────

    /// Get aggregate storage statistics.
    async fn get_stats(&self) -> StorageResult<TokenUsageStats>;

    /// Get per-segment-type statistics.
    async fn get_segment_type_stats(&self) -> StorageResult<Vec<SegmentTypeStats>>;

    /// Run a filtered analytics query.
    async fn query_analytics(
        &self,
        query: &AnalyticsQuery,
    ) -> StorageResult<Vec<HashMap<String, serde_json::Value>>>;

    /// List unique system-prompt segments, ordered by `ref_count` descending.
    ///
    /// Returns `(hash, token_count, ref_count)` tuples.
    async fn list_unique_system_prompts(&self) -> StorageResult<Vec<(SegmentHash, u32, u64)>>;

    /// Return IDs of all conversations that reference a given segment.
    async fn find_conversations_by_segment(
        &self,
        hash: &SegmentHash,
    ) -> StorageResult<Vec<String>>;

    // ── Maintenance ───────────────────────────────────────────────────────

    /// Delete all segments with `ref_count == 0`.
    ///
    /// Returns the number of segments deleted.
    async fn garbage_collect(&self) -> StorageResult<u64>;

    /// Return the total bytes stored in the backend.
    async fn storage_size_bytes(&self) -> StorageResult<u64>;
}
