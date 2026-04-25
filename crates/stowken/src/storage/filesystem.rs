//! Filesystem storage backend.
//!
//! Segments are stored in a content-addressed directory tree:
//!   `{root}/segments/{hash[0..2]}/{hash[2..4]}/{full_hash}`
//!
//! Manifests are stored at:
//!   `{root}/manifests/{conversation_id}.json`
//!
//! The metadata SQLite database lives at:
//!   `{root}/metadata.db`

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::fs;
use tokio::sync::Mutex;

use crate::types::{AnalyticsQuery, ConversationManifest, SegmentHash, StoredSegment};

use super::backend::{StorageBackend, StorageError, StorageResult};

/// A header stored alongside each segment file (JSON sidecar `{hash}.meta`).
#[derive(serde::Serialize, serde::Deserialize)]
struct SegmentMeta {
    segment_type: String,
    tokenizer: String,
    token_count: u32,
    raw_size: u32,
    compressed_size: u32,
    ref_count: u64,
    created_at: String,
}

/// Filesystem-backed storage backend.
///
/// Thread-safe ref-count updates are serialised through a `Mutex<()>` to avoid
/// TOCTOU races when incrementing/decrementing the metadata sidecar.
pub struct FilesystemBackend {
    root: PathBuf,
    /// Serialises metadata-sidecar mutations.
    mu: Arc<Mutex<()>>,
}

impl FilesystemBackend {
    /// Create a new filesystem backend rooted at `path`.
    ///
    /// Creates the directory structure if it does not exist.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(root.join("segments")).await?;
        fs::create_dir_all(root.join("manifests")).await?;
        Ok(Self {
            root,
            mu: Arc::new(Mutex::new(())),
        })
    }

    fn segment_dir(&self, hash: &SegmentHash) -> PathBuf {
        let h = &hash.0;
        self.root.join("segments").join(&h[..2]).join(&h[2..4])
    }

    fn segment_path(&self, hash: &SegmentHash) -> PathBuf {
        self.segment_dir(hash).join(&hash.0)
    }

    fn segment_meta_path(&self, hash: &SegmentHash) -> PathBuf {
        self.segment_dir(hash).join(format!("{}.meta", hash.0))
    }

    fn manifest_path(&self, id: &str) -> PathBuf {
        self.root.join("manifests").join(format!("{id}.json"))
    }

    async fn read_meta(&self, hash: &SegmentHash) -> StorageResult<SegmentMeta> {
        let data = fs::read(self.segment_meta_path(hash)).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::SegmentNotFound(hash.0.clone())
            } else {
                StorageError::IoError(e)
            }
        })?;
        serde_json::from_slice(&data)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    async fn write_meta(&self, hash: &SegmentHash, meta: &SegmentMeta) -> StorageResult<()> {
        let data = serde_json::to_vec(meta)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        fs::write(self.segment_meta_path(hash), data).await?;
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for FilesystemBackend {
    // ── Segment operations ────────────────────────────────────────────────

    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()> {
        let _guard = self.mu.lock().await;
        let dir = self.segment_dir(&segment.hash);
        fs::create_dir_all(&dir).await?;

        let seg_path = self.segment_path(&segment.hash);
        if seg_path.exists() {
            // Already stored — increment ref count
            let mut meta = self.read_meta(&segment.hash).await?;
            meta.ref_count += 1;
            self.write_meta(&segment.hash, &meta).await?;
            return Ok(());
        }

        fs::write(&seg_path, &segment.compressed_data).await?;

        let meta = SegmentMeta {
            segment_type: segment.segment_type.to_string(),
            tokenizer: segment.tokenizer.clone(),
            token_count: segment.token_count,
            raw_size: segment.raw_size,
            compressed_size: segment.compressed_size,
            ref_count: segment.ref_count,
            created_at: segment.created_at.to_rfc3339(),
        };
        self.write_meta(&segment.hash, &meta).await?;
        Ok(())
    }

    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment> {
        let seg_path = self.segment_path(hash);
        let compressed_data = fs::read(&seg_path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::SegmentNotFound(hash.0.clone())
            } else {
                StorageError::IoError(e)
            }
        })?;
        let meta = self.read_meta(hash).await?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&meta.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let seg_type: crate::types::SegmentType = meta
            .segment_type
            .parse()
            .unwrap_or(crate::types::SegmentType::UserTurn);

        Ok(StoredSegment {
            hash: hash.clone(),
            segment_type: seg_type,
            tokenizer: meta.tokenizer,
            token_count: meta.token_count,
            compressed_data,
            raw_size: meta.raw_size,
            compressed_size: meta.compressed_size,
            ref_count: meta.ref_count,
            created_at,
        })
    }

    async fn has_segment(&self, hash: &SegmentHash) -> StorageResult<bool> {
        Ok(self.segment_path(hash).exists())
    }

    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()> {
        let _guard = self.mu.lock().await;
        let mut meta = self.read_meta(hash).await?;
        meta.ref_count += 1;
        self.write_meta(hash, &meta).await
    }

    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool> {
        let _guard = self.mu.lock().await;
        let mut meta = self.read_meta(hash).await?;
        meta.ref_count = meta.ref_count.saturating_sub(1);
        self.write_meta(hash, &meta).await?;
        Ok(meta.ref_count == 0)
    }

    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()> {
        let _ = fs::remove_file(self.segment_path(hash)).await;
        let _ = fs::remove_file(self.segment_meta_path(hash)).await;
        Ok(())
    }

    // ── Manifest operations ───────────────────────────────────────────────

    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()> {
        let data = serde_json::to_vec(manifest)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        fs::write(self.manifest_path(&manifest.id), data).await?;
        Ok(())
    }

    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest> {
        let data = fs::read(self.manifest_path(id)).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::ConversationNotFound(id.to_owned())
            } else {
                StorageError::IoError(e)
            }
        })?;
        serde_json::from_slice(&data)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    async fn delete_manifest(&self, id: &str) -> StorageResult<()> {
        let _ = fs::remove_file(self.manifest_path(id)).await;
        Ok(())
    }

    async fn list_conversations(
        &self,
        query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> StorageResult<Vec<String>> {
        let manifest_dir = self.root.join("manifests");
        let mut read_dir = fs::read_dir(&manifest_dir).await?;
        let mut ids: Vec<String> = Vec::new();

        while let Some(entry) = read_dir.next_entry().await? {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".json") {
                let id = name_str.trim_end_matches(".json").to_owned();
                // Apply filters — read manifests only when needed
                if query.model.is_some() || query.application.is_some()
                    || query.date_from.is_some() || query.date_to.is_some()
                {
                    if let Ok(manifest) = self.get_manifest(&id).await {
                        if let Some(m) = &query.model {
                            if &manifest.model != m {
                                continue;
                            }
                        }
                        if let Some(a) = &query.application {
                            if manifest.application.as_deref() != Some(a.as_str()) {
                                continue;
                            }
                        }
                        if let Some(from) = query.date_from {
                            if manifest.created_at < from {
                                continue;
                            }
                        }
                        if let Some(to) = query.date_to {
                            if manifest.created_at > to {
                                continue;
                            }
                        }
                    } else {
                        continue;
                    }
                }
                ids.push(id);
            }
        }

        Ok(ids.into_iter().skip(offset as usize).take(limit as usize).collect())
    }

    async fn list_garbage(&self) -> StorageResult<Vec<SegmentHash>> {
        let segments_dir = self.root.join("segments");
        let mut out = Vec::new();
        if !segments_dir.exists() {
            return Ok(out);
        }
        let mut top = fs::read_dir(&segments_dir).await?;
        while let Some(prefix1) = top.next_entry().await? {
            let mut mid = fs::read_dir(prefix1.path()).await?;
            while let Some(prefix2) = mid.next_entry().await? {
                let mut bottom = fs::read_dir(prefix2.path()).await?;
                while let Some(file) = bottom.next_entry().await? {
                    let name = file.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.ends_with(".meta") {
                        continue;
                    }
                    let hash = SegmentHash(name_str.to_string());
                    if let Ok(meta) = self.read_meta(&hash).await {
                        if meta.ref_count == 0 {
                            out.push(hash);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    async fn garbage_collect(&self) -> StorageResult<u64> {
        let candidates = self.list_garbage().await?;
        let mut deleted = 0u64;
        for hash in candidates {
            self.delete_segment(&hash).await?;
            deleted += 1;
        }
        Ok(deleted)
    }

    async fn storage_size_bytes(&self) -> StorageResult<u64> {
        let segments_dir = self.root.join("segments");
        let mut total: u64 = 0;
        if !segments_dir.exists() {
            return Ok(0);
        }
        let mut top = fs::read_dir(&segments_dir).await?;
        while let Some(prefix1) = top.next_entry().await? {
            let mut mid = fs::read_dir(prefix1.path()).await?;
            while let Some(prefix2) = mid.next_entry().await? {
                let mut bottom = fs::read_dir(prefix2.path()).await?;
                while let Some(file) = bottom.next_entry().await? {
                    if let Ok(metadata) = file.metadata().await {
                        total += metadata.len();
                    }
                }
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SegmentRef, SegmentType};
    use chrono::Utc;
    use tempfile::TempDir;

    // ── Helpers ───────────────────────────────────────────────────────────────

    async fn backend() -> (FilesystemBackend, TempDir) {
        let dir = TempDir::new().unwrap();
        let b = FilesystemBackend::new(dir.path()).await.unwrap();
        (b, dir)
    }

    fn make_segment(hash: &str) -> StoredSegment {
        StoredSegment {
            hash: SegmentHash(hash.to_owned()),
            segment_type: SegmentType::UserTurn,
            tokenizer: "test".to_owned(),
            token_count: 5,
            compressed_data: vec![1, 2, 3, 4, 5],
            raw_size: 20,
            compressed_size: 5,
            ref_count: 1,
            created_at: Utc::now(),
        }
    }

    fn make_segment_with_data(hash: &str, data: Vec<u8>) -> StoredSegment {
        let compressed_size = data.len() as u32;
        StoredSegment {
            hash: SegmentHash(hash.to_owned()),
            segment_type: SegmentType::SystemPrompt,
            tokenizer: "cl100k_base".to_owned(),
            token_count: data.len() as u32 / 4,
            raw_size: data.len() as u32,
            compressed_size,
            compressed_data: data,
            ref_count: 1,
            created_at: Utc::now(),
        }
    }

    fn make_manifest(id: &str) -> ConversationManifest {
        ConversationManifest {
            schema_version: crate::types::MANIFEST_SCHEMA_VERSION,
            id: id.to_owned(),
            application: None,
            model: "gpt-4".to_owned(),
            tokenizer: "test".to_owned(),
            total_tokens: 5,
            segments: vec![SegmentRef {
                segment_type: SegmentType::UserTurn,
                hash: SegmentHash("abc123".to_owned()),
                token_count: 5,
                position: 0,
            }],
            created_at: Utc::now(),
            metadata: None,
        }
    }

    // Use a 64-char hash so segment_dir() indexing into [0..2] and [2..4] is safe.
    const HASH_A: &str = "aabbccddeeff001122334455667788990011223344556677889900112233445566";
    const HASH_B: &str = "bbccddeeff00112233445566778899001122334455667788990011223344556677";
    const HASH_C: &str = "ccddeeff0011223344556677889900112233445566778899001122334455667788";

    // ── Basic correctness ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn segment_roundtrip() {
        let (b, _dir) = backend().await;
        let seg = make_segment(HASH_A);
        b.put_segment(&seg).await.unwrap();
        assert!(b.has_segment(&seg.hash).await.unwrap());
        let fetched = b.get_segment(&seg.hash).await.unwrap();
        assert_eq!(fetched.compressed_data, seg.compressed_data);
        assert_eq!(fetched.token_count, seg.token_count);
        assert_eq!(fetched.ref_count, 1);
    }

    #[tokio::test]
    async fn segment_data_integrity() {
        // Verify every byte of a 256-byte payload survives the roundtrip.
        let (b, _dir) = backend().await;
        let payload: Vec<u8> = (0u8..=255).collect();
        let seg = make_segment_with_data(HASH_A, payload.clone());
        b.put_segment(&seg).await.unwrap();
        let fetched = b.get_segment(&seg.hash).await.unwrap();
        assert_eq!(fetched.compressed_data, payload, "payload bytes corrupted");
    }

    #[tokio::test]
    async fn manifest_roundtrip() {
        let (b, _dir) = backend().await;
        let m = make_manifest("conv-fs-1");
        b.put_manifest(&m).await.unwrap();
        let fetched = b.get_manifest("conv-fs-1").await.unwrap();
        assert_eq!(fetched.id, "conv-fs-1");
        assert_eq!(fetched.model, "gpt-4");
    }

    #[tokio::test]
    async fn missing_segment_returns_error() {
        let (b, _dir) = backend().await;
        let result = b.get_segment(&SegmentHash("nonexistent0011223344556677889900112233445566778899".to_owned())).await;
        assert!(matches!(result, Err(StorageError::SegmentNotFound(_))));
    }

    #[tokio::test]
    async fn missing_manifest_returns_error() {
        let (b, _dir) = backend().await;
        let result = b.get_manifest("no-such-conv").await;
        assert!(matches!(result, Err(StorageError::ConversationNotFound(_))));
    }

    // ── Ref-count lifecycle ───────────────────────────────────────────────────

    #[tokio::test]
    async fn ref_count_increment_decrement() {
        let (b, _dir) = backend().await;
        let seg = make_segment(HASH_A);
        b.put_segment(&seg).await.unwrap();
        b.increment_ref(&seg.hash).await.unwrap();
        let fetched = b.get_segment(&seg.hash).await.unwrap();
        assert_eq!(fetched.ref_count, 2);
        let zero = b.decrement_ref(&seg.hash).await.unwrap();
        assert!(!zero, "ref_count=1, should not be zero yet");
        let zero2 = b.decrement_ref(&seg.hash).await.unwrap();
        assert!(zero2, "ref_count=0, should signal GC eligible");
    }

    #[tokio::test]
    async fn gc_removes_zero_ref_segments_only() {
        let (b, _dir) = backend().await;
        let seg_a = make_segment(HASH_A);
        let seg_b = make_segment(HASH_B);
        b.put_segment(&seg_a).await.unwrap();
        b.put_segment(&seg_b).await.unwrap();

        // Zero out A, keep B alive.
        b.decrement_ref(&seg_a.hash).await.unwrap();

        let deleted = b.garbage_collect().await.unwrap();
        assert_eq!(deleted, 1, "exactly one segment should be GC'd");
        assert!(!b.has_segment(&seg_a.hash).await.unwrap(), "A should be gone");
        assert!(b.has_segment(&seg_b.hash).await.unwrap(), "B should survive");
    }

    #[tokio::test]
    async fn delete_manifest_then_not_found() {
        let (b, _dir) = backend().await;
        let m = make_manifest("to-delete");
        b.put_manifest(&m).await.unwrap();
        b.delete_manifest("to-delete").await.unwrap();
        let result = b.get_manifest("to-delete").await;
        assert!(matches!(result, Err(StorageError::ConversationNotFound(_))));
    }

    // ── Durability ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn data_survives_reopen() {
        // Write data with backend A, drop it, reopen from same directory — data must still be there.
        let dir = TempDir::new().unwrap();
        let hash = SegmentHash(HASH_A.to_owned());

        {
            let b = FilesystemBackend::new(dir.path()).await.unwrap();
            b.put_segment(&make_segment(HASH_A)).await.unwrap();
            b.put_manifest(&make_manifest("durable-conv")).await.unwrap();
        } // backend dropped here

        let b2 = FilesystemBackend::new(dir.path()).await.unwrap();
        assert!(b2.has_segment(&hash).await.unwrap(), "segment missing after reopen");
        let fetched = b2.get_segment(&hash).await.unwrap();
        assert_eq!(fetched.compressed_data, vec![1, 2, 3, 4, 5]);
        let manifest = b2.get_manifest("durable-conv").await.unwrap();
        assert_eq!(manifest.id, "durable-conv");
    }

    #[tokio::test]
    async fn ref_count_persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let hash = SegmentHash(HASH_A.to_owned());

        {
            let b = FilesystemBackend::new(dir.path()).await.unwrap();
            b.put_segment(&make_segment(HASH_A)).await.unwrap();
            b.increment_ref(&hash).await.unwrap(); // ref_count = 2
            b.increment_ref(&hash).await.unwrap(); // ref_count = 3
        }

        let b2 = FilesystemBackend::new(dir.path()).await.unwrap();
        let fetched = b2.get_segment(&hash).await.unwrap();
        assert_eq!(fetched.ref_count, 3, "ref_count should be 3 after reopen");
    }

    // ── Concurrency ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn concurrent_ref_increments_are_serialised() {
        // 20 tasks all increment the same segment's ref_count concurrently.
        // Final ref_count must equal 1 (initial) + 20.
        let (b, _dir) = backend().await;
        let seg = make_segment(HASH_A);
        b.put_segment(&seg).await.unwrap();

        let b = Arc::new(b);
        let hash = seg.hash.clone();
        let mut handles = Vec::new();
        for _ in 0..20 {
            let b = Arc::clone(&b);
            let h = hash.clone();
            handles.push(tokio::spawn(async move {
                b.increment_ref(&h).await.unwrap();
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        let fetched = b.get_segment(&hash).await.unwrap();
        assert_eq!(fetched.ref_count, 21, "expected 1 + 20 increments = 21");
    }

    #[tokio::test]
    async fn concurrent_distinct_writes_all_succeed() {
        // 20 tasks each write a distinct segment — no overlaps, no dropped writes.
        let (b, _dir) = backend().await;
        let b = Arc::new(b);

        // Pre-build 20 distinct 64-char hex hashes.
        let hashes: Vec<String> = (0..20u8)
            .map(|i| format!("{:0>64}", format!("{i:x}")))
            .collect();

        let mut handles = Vec::new();
        for hash in &hashes {
            let b = Arc::clone(&b);
            let seg = make_segment_with_data(hash, vec![42u8; 32]);
            handles.push(tokio::spawn(async move {
                b.put_segment(&seg).await.unwrap();
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        for hash in &hashes {
            let h = SegmentHash(hash.clone());
            assert!(b.has_segment(&h).await.unwrap(), "segment {hash} missing");
            let fetched = b.get_segment(&h).await.unwrap();
            assert_eq!(fetched.compressed_data, vec![42u8; 32]);
        }
    }

    // ── Pagination and listing ────────────────────────────────────────────────

    #[tokio::test]
    async fn list_conversations_pagination() {
        let (b, _dir) = backend().await;
        for i in 0..20 {
            let m = ConversationManifest {
                schema_version: crate::types::MANIFEST_SCHEMA_VERSION,
                id: format!("conv-{i:02}"),
                application: None,
                model: "gpt-4".to_owned(),
                tokenizer: "test".to_owned(),
                total_tokens: 10,
                segments: vec![],
                created_at: Utc::now(),
                metadata: None,
            };
            b.put_manifest(&m).await.unwrap();
        }

        let query = AnalyticsQuery::default();
        let page1 = b.list_conversations(&query, 5, 0).await.unwrap();
        let page2 = b.list_conversations(&query, 5, 5).await.unwrap();
        let page3 = b.list_conversations(&query, 5, 10).await.unwrap();
        let page4 = b.list_conversations(&query, 5, 15).await.unwrap();

        assert_eq!(page1.len(), 5);
        assert_eq!(page2.len(), 5);
        assert_eq!(page3.len(), 5);
        assert_eq!(page4.len(), 5);

        let all: Vec<_> = [page1, page2, page3, page4].concat();
        let mut sorted = all.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 20, "all 20 conversations should be distinct");
    }

    // ── Storage size ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn storage_size_bytes_accumulates() {
        let (b, _dir) = backend().await;
        let before = b.storage_size_bytes().await.unwrap();

        let payload = vec![0u8; 512];
        b.put_segment(&make_segment_with_data(HASH_A, payload.clone())).await.unwrap();
        b.put_segment(&make_segment_with_data(HASH_B, payload.clone())).await.unwrap();
        b.put_segment(&make_segment_with_data(HASH_C, payload.clone())).await.unwrap();

        let after = b.storage_size_bytes().await.unwrap();
        // Each segment file is 512 bytes; meta files add a little extra.
        // At minimum we should see the 3 × 512 = 1536 bytes of segment data.
        assert!(
            after >= before + 3 * 512,
            "expected at least 1536 more bytes, got {} → {}",
            before, after
        );
    }

    // ── Large corpus ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn large_corpus_roundtrip() {
        let (b, _dir) = backend().await;
        let n = 200;

        // Write 200 distinct segments with randomised 128-byte payloads.
        let mut hashes = Vec::new();
        for i in 0u32..n {
            let hash = format!("{:0>64x}", (i as u64).wrapping_mul(0x9e3779b97f4a7c15u64));
            let data: Vec<u8> = (0..128).map(|j| ((i * 17 + j) % 256) as u8).collect();
            b.put_segment(&make_segment_with_data(&hash, data.clone())).await.unwrap();
            hashes.push((hash, data));
        }

        // Verify every segment is intact.
        for (hash, expected_data) in &hashes {
            let h = SegmentHash(hash.clone());
            let fetched = b.get_segment(&h).await.unwrap();
            assert_eq!(&fetched.compressed_data, expected_data, "data mismatch for {hash}");
        }

        // GC with all ref_counts=1 should delete nothing.
        let deleted = b.garbage_collect().await.unwrap();
        assert_eq!(deleted, 0);
    }

    // ── Full vault integration ────────────────────────────────────────────────

    #[tokio::test]
    async fn vault_store_retrieve_delete_via_filesystem_backend() {
        use crate::{
            storage::FilesystemBackend,
            types::{Conversation, Message, MessageContent, StowkenConfig},
            vault::Stowken,
        };

        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path()).await.unwrap();
        let db_path = dir.path().join("meta.db").to_str().unwrap().to_owned();
        let vault = Stowken::open(backend, StowkenConfig::default(), &db_path).await.unwrap();

        let tokens: Vec<u32> = (0u32..100).collect();
        let conv = Conversation {
            id: Some("fs-test".to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![Message {
                role: "system".to_owned(),
                content: MessageContent::Tokens(tokens.clone()),
                name: None,
                tool_call_id: None,
            }],
            metadata: None,
        };

        let result = vault.store(conv).await.unwrap();
        assert_eq!(result.id, "fs-test");
        assert_eq!(result.new_segments, 1);

        let retrieved = vault.retrieve("fs-test").await.unwrap();
        assert_eq!(retrieved.segments[0].tokens, tokens);

        vault.delete("fs-test").await.unwrap();
        let deleted_gc = vault.gc().await.unwrap();
        assert_eq!(deleted_gc, 1, "GC should clean up the dereferenced segment");
    }

    #[tokio::test]
    async fn vault_data_survives_reopen_via_filesystem() {
        use crate::{
            storage::FilesystemBackend,
            types::{Conversation, Message, MessageContent, StowkenConfig},
            vault::Stowken,
        };

        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("meta.db").to_str().unwrap().to_owned();
        let tokens: Vec<u32> = (0u32..50).collect();

        // Session 1: store a conversation.
        {
            let backend = FilesystemBackend::new(dir.path()).await.unwrap();
            let vault = Stowken::open(backend, StowkenConfig::default(), &db_path).await.unwrap();
            let conv = Conversation {
                id: Some("persist-test".to_owned()),
                model: "gpt-4".to_owned(),
                tokenizer: "cl100k_base".to_owned(),
                application: None,
                messages: vec![Message {
                    role: "user".to_owned(),
                    content: MessageContent::Tokens(tokens.clone()),
                    name: None,
                    tool_call_id: None,
                }],
                metadata: None,
            };
            vault.store(conv).await.unwrap();
        }

        // Session 2: open fresh vault on same directory, data must still be there.
        {
            let backend = FilesystemBackend::new(dir.path()).await.unwrap();
            let vault = Stowken::open(backend, StowkenConfig::default(), &db_path).await.unwrap();
            let retrieved = vault.retrieve("persist-test").await.unwrap();
            assert_eq!(retrieved.segments[0].tokens, tokens, "data corrupted across reopen");
        }
    }
}
