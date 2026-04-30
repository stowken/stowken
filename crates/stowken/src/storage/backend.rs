//! `StorageBackend` — the pluggable persistence abstraction.
//!
//! Stowken handles compression, dedup, and segmentation.
//! Backends only store and retrieve bytes + manifests.

use async_trait::async_trait;

use crate::types::{AnalyticsQuery, ConversationManifest, SegmentHash, StoredSegment};

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
///
/// Aggregate analytics (stats, system-prompt audits, query) live on
/// `MetadataIndex`, not here. Backends are pure storage.
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

    /// Increment the reference count of an existing segment.
    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()>;

    /// Replace the compressed bytes of an existing segment without
    /// touching its `ref_count`. Used by `compact_substrings` to
    /// rewrite pre-train segments to use the substring registry.
    ///
    /// Atomicity: implementations must update the data and any
    /// metadata sidecar consistently. A crash mid-replace must leave
    /// the segment in either the old or new form — both decompressible
    /// since the frame version byte dispatches per-frame.
    async fn replace_segment_data(
        &self,
        hash: &SegmentHash,
        new_data: Vec<u8>,
    ) -> StorageResult<()>;

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

    // ── Maintenance ───────────────────────────────────────────────────────

    /// List the hashes of segments currently eligible for GC (`ref_count == 0`).
    ///
    /// Used by `gc --dry-run` to report what *would* be deleted. The default
    /// implementation returns an empty list; backends are encouraged to
    /// override it. Returning an empty list is acceptable but means dry-run
    /// will report nothing on that backend.
    async fn list_garbage(&self) -> StorageResult<Vec<SegmentHash>> {
        Ok(Vec::new())
    }

    /// Delete all segments with `ref_count == 0`.
    ///
    /// Returns the number of segments deleted.
    async fn garbage_collect(&self) -> StorageResult<u64>;

    /// Return the total bytes stored in the backend.
    async fn storage_size_bytes(&self) -> StorageResult<u64>;
}
