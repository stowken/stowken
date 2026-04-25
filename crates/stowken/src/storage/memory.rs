//! In-memory `StorageBackend` implementation for testing and ephemeral use cases.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::types::{
    AnalyticsQuery, ConversationManifest, SegmentHash, SegmentType, SegmentTypeStats,
    StoredSegment, TokenUsageStats,
};

use super::backend::{StorageBackend, StorageError, StorageResult};

/// All state lives inside a single `Arc<RwLock<...>>` so the backend can be
/// cloned cheaply and shared across async tasks.
#[derive(Debug, Default)]
struct MemoryState {
    segments: HashMap<String, StoredSegment>,
    manifests: HashMap<String, ConversationManifest>,
}

/// Thread-safe in-memory storage backend.
#[derive(Debug, Clone, Default)]
pub struct MemoryBackend {
    state: Arc<RwLock<MemoryState>>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StorageBackend for MemoryBackend {
    // ── Segment operations ────────────────────────────────────────────────

    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()> {
        let mut state = self.state.write().await;
        let key = segment.hash.0.clone();
        if let Some(existing) = state.segments.get_mut(&key) {
            existing.ref_count += 1;
        } else {
            state.segments.insert(key, segment.clone());
        }
        Ok(())
    }

    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment> {
        let state = self.state.read().await;
        state
            .segments
            .get(&hash.0)
            .cloned()
            .ok_or_else(|| StorageError::SegmentNotFound(hash.0.clone()))
    }

    async fn has_segment(&self, hash: &SegmentHash) -> StorageResult<bool> {
        let state = self.state.read().await;
        Ok(state.segments.contains_key(&hash.0))
    }

    async fn has_segments(&self, hashes: &[SegmentHash]) -> StorageResult<Vec<SegmentHash>> {
        let state = self.state.read().await;
        Ok(hashes
            .iter()
            .filter(|h| state.segments.contains_key(&h.0))
            .cloned()
            .collect())
    }

    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()> {
        let mut state = self.state.write().await;
        state
            .segments
            .get_mut(&hash.0)
            .ok_or_else(|| StorageError::SegmentNotFound(hash.0.clone()))?
            .ref_count += 1;
        Ok(())
    }

    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool> {
        let mut state = self.state.write().await;
        let seg = state
            .segments
            .get_mut(&hash.0)
            .ok_or_else(|| StorageError::SegmentNotFound(hash.0.clone()))?;
        seg.ref_count = seg.ref_count.saturating_sub(1);
        Ok(seg.ref_count == 0)
    }

    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()> {
        let mut state = self.state.write().await;
        state.segments.remove(&hash.0);
        Ok(())
    }

    // ── Manifest operations ───────────────────────────────────────────────

    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()> {
        let mut state = self.state.write().await;
        state.manifests.insert(manifest.id.clone(), manifest.clone());
        Ok(())
    }

    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest> {
        let state = self.state.read().await;
        state
            .manifests
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::ConversationNotFound(id.to_owned()))
    }

    async fn delete_manifest(&self, id: &str) -> StorageResult<()> {
        let mut state = self.state.write().await;
        state.manifests.remove(id);
        Ok(())
    }

    async fn list_conversations(
        &self,
        query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> StorageResult<Vec<String>> {
        let state = self.state.read().await;
        let ids: Vec<String> = state
            .manifests
            .values()
            .filter(|m| {
                if let Some(model) = &query.model {
                    if &m.model != model {
                        return false;
                    }
                }
                if let Some(app) = &query.application {
                    if m.application.as_deref() != Some(app.as_str()) {
                        return false;
                    }
                }
                if let Some(from) = query.date_from {
                    if m.created_at < from {
                        return false;
                    }
                }
                if let Some(to) = query.date_to {
                    if m.created_at > to {
                        return false;
                    }
                }
                true
            })
            .map(|m| m.id.clone())
            .skip(offset as usize)
            .take(limit as usize)
            .collect();
        Ok(ids)
    }

    // ── Analytics ─────────────────────────────────────────────────────────

    async fn get_stats(&self) -> StorageResult<TokenUsageStats> {
        let state = self.state.read().await;
        let total_conversations = state.manifests.len() as u64;
        let unique_segments = state.segments.len() as u64;
        let total_segments: u64 = state.segments.values().map(|s| s.ref_count).sum();
        let total_tokens: u64 = state
            .manifests
            .values()
            .map(|m| m.total_tokens)
            .sum();
        let storage_bytes: u64 = state.segments.values().map(|s| s.compressed_size as u64).sum();
        let naive_bytes: u64 = state.segments.values().map(|s| s.raw_size as u64 * s.ref_count).sum();
        let dedup_ratio = if total_segments == 0 {
            0.0
        } else {
            1.0 - (unique_segments as f64 / total_segments as f64)
        };
        let raw_bytes: u64 = state.segments.values().map(|s| s.raw_size as u64).sum();
        let compression_ratio = if raw_bytes == 0 {
            1.0
        } else {
            storage_bytes as f64 / raw_bytes as f64
        };
        let savings_percentage = if naive_bytes == 0 {
            0.0
        } else {
            (1.0 - (storage_bytes as f64 / naive_bytes as f64)) * 100.0
        };
        Ok(TokenUsageStats {
            total_tokens,
            total_conversations,
            unique_segments,
            total_segments,
            dedup_ratio,
            compression_ratio,
            storage_bytes,
            naive_bytes,
            savings_percentage,
        })
    }

    async fn get_segment_type_stats(&self) -> StorageResult<Vec<SegmentTypeStats>> {
        let state = self.state.read().await;
        let mut map: HashMap<String, (u64, u64, u64, u64)> = HashMap::new();
        // key → (unique, total_refs, total_tokens, compressed_bytes)
        for seg in state.segments.values() {
            let key = seg.segment_type.to_string();
            let e = map.entry(key).or_default();
            e.0 += 1;
            e.1 += seg.ref_count;
            e.2 += seg.token_count as u64 * seg.ref_count;
            e.3 += seg.compressed_size as u64;
        }
        let stats = map
            .into_iter()
            .map(|(k, (unique, refs, tokens, compressed))| {
                let seg_type: SegmentType = k.parse().unwrap_or(SegmentType::UserTurn);
                crate::types::SegmentTypeStats {
                    segment_type: seg_type,
                    unique_count: unique,
                    total_references: refs,
                    dedup_ratio: if refs == 0 { 0.0 } else { 1.0 - unique as f64 / refs as f64 },
                    avg_token_count: if unique == 0 { 0.0 } else { tokens as f64 / refs as f64 },
                    total_tokens: tokens,
                    compressed_bytes: compressed,
                }
            })
            .collect();
        Ok(stats)
    }

    async fn query_analytics(
        &self,
        _query: &AnalyticsQuery,
    ) -> StorageResult<Vec<HashMap<String, serde_json::Value>>> {
        // For the memory backend a full analytics engine is not needed.
        // Return an empty result; analytics are driven by MetadataIndex in vault.rs.
        Ok(vec![])
    }

    async fn list_unique_system_prompts(&self) -> StorageResult<Vec<(SegmentHash, u32, u64)>> {
        let state = self.state.read().await;
        let mut prompts: Vec<(SegmentHash, u32, u64)> = state
            .segments
            .values()
            .filter(|s| s.segment_type == SegmentType::SystemPrompt)
            .map(|s| (s.hash.clone(), s.token_count, s.ref_count))
            .collect();
        prompts.sort_by_key(|p| Reverse(p.2));
        Ok(prompts)
    }

    async fn find_conversations_by_segment(
        &self,
        hash: &SegmentHash,
    ) -> StorageResult<Vec<String>> {
        let state = self.state.read().await;
        let ids: Vec<String> = state
            .manifests
            .values()
            .filter(|m| m.segments.iter().any(|s| &s.hash == hash))
            .map(|m| m.id.clone())
            .collect();
        Ok(ids)
    }

    // ── Maintenance ───────────────────────────────────────────────────────

    async fn garbage_collect(&self) -> StorageResult<u64> {
        let mut state = self.state.write().await;
        let before = state.segments.len();
        state.segments.retain(|_, s| s.ref_count > 0);
        Ok((before - state.segments.len()) as u64)
    }

    async fn storage_size_bytes(&self) -> StorageResult<u64> {
        let state = self.state.read().await;
        let total: u64 = state.segments.values().map(|s| s.compressed_size as u64).sum();
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SegmentRef, SegmentType};
    use chrono::Utc;

    fn make_segment(hash: &str, ref_count: u64) -> StoredSegment {
        StoredSegment {
            hash: SegmentHash(hash.to_owned()),
            segment_type: SegmentType::UserTurn,
            tokenizer: "test".to_owned(),
            token_count: 10,
            compressed_data: vec![0u8; 20],
            raw_size: 40,
            compressed_size: 20,
            ref_count,
            created_at: Utc::now(),
        }
    }

    fn make_manifest(id: &str, hash: &str) -> ConversationManifest {
        ConversationManifest {
            id: id.to_owned(),
            application: None,
            model: "gpt-4".to_owned(),
            tokenizer: "test".to_owned(),
            total_tokens: 10,
            segments: vec![SegmentRef {
                segment_type: SegmentType::UserTurn,
                hash: SegmentHash(hash.to_owned()),
                token_count: 10,
                position: 0,
            }],
            created_at: Utc::now(),
            metadata: None,
        }
    }

    #[tokio::test]
    async fn put_and_get_segment() {
        let backend = MemoryBackend::new();
        let seg = make_segment("abc123", 1);
        backend.put_segment(&seg).await.unwrap();
        let fetched = backend.get_segment(&SegmentHash("abc123".to_owned())).await.unwrap();
        assert_eq!(fetched.token_count, 10);
    }

    #[tokio::test]
    async fn put_segment_is_idempotent() {
        let backend = MemoryBackend::new();
        let seg = make_segment("abc123", 1);
        backend.put_segment(&seg).await.unwrap();
        backend.put_segment(&seg).await.unwrap(); // second put → ref_count = 2
        let fetched = backend.get_segment(&SegmentHash("abc123".to_owned())).await.unwrap();
        assert_eq!(fetched.ref_count, 2);
    }

    #[tokio::test]
    async fn decrement_ref_signals_gc_eligible() {
        let backend = MemoryBackend::new();
        let seg = make_segment("abc123", 1);
        backend.put_segment(&seg).await.unwrap();
        let zero = backend.decrement_ref(&SegmentHash("abc123".to_owned())).await.unwrap();
        assert!(zero);
    }

    #[tokio::test]
    async fn garbage_collect_removes_zero_refs() {
        let backend = MemoryBackend::new();
        let seg = make_segment("abc123", 1);
        backend.put_segment(&seg).await.unwrap();
        backend.decrement_ref(&SegmentHash("abc123".to_owned())).await.unwrap();
        let deleted = backend.garbage_collect().await.unwrap();
        assert_eq!(deleted, 1);
        assert!(!backend.has_segment(&SegmentHash("abc123".to_owned())).await.unwrap());
    }

    #[tokio::test]
    async fn manifest_roundtrip() {
        let backend = MemoryBackend::new();
        let manifest = make_manifest("conv-1", "abc123");
        backend.put_manifest(&manifest).await.unwrap();
        let fetched = backend.get_manifest("conv-1").await.unwrap();
        assert_eq!(fetched.id, "conv-1");
    }

    #[tokio::test]
    async fn missing_segment_returns_error() {
        let backend = MemoryBackend::new();
        let result = backend.get_segment(&SegmentHash("nonexistent".to_owned())).await;
        assert!(matches!(result, Err(StorageError::SegmentNotFound(_))));
    }
}
