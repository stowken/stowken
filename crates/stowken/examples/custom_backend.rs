//! Custom backend: implement StorageBackend for your own persistence layer.
//!
//! This example shows a minimal in-memory backend backed by a HashMap,
//! demonstrating the interface you need to implement for any custom store.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use stowken::{
    storage::{StorageBackend, StorageError, StorageResult},
    types::{
        AnalyticsQuery, Conversation, ConversationManifest, Message, MessageContent, SegmentHash,
        StoredSegment, StowkenConfig,
    },
    Stowken,
};

#[derive(Default, Clone)]
struct MyBackend {
    segments: Arc<RwLock<HashMap<String, StoredSegment>>>,
    manifests: Arc<RwLock<HashMap<String, ConversationManifest>>>,
}

#[async_trait]
impl StorageBackend for MyBackend {
    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()> {
        let mut map = self.segments.write().await;
        map.entry(segment.hash.0.clone())
            .and_modify(|s| s.ref_count += 1)
            .or_insert_with(|| segment.clone());
        Ok(())
    }

    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment> {
        self.segments
            .read()
            .await
            .get(&hash.0)
            .cloned()
            .ok_or_else(|| StorageError::SegmentNotFound(hash.0.clone()))
    }

    async fn has_segment(&self, hash: &SegmentHash) -> StorageResult<bool> {
        Ok(self.segments.read().await.contains_key(&hash.0))
    }

    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()> {
        let mut map = self.segments.write().await;
        if let Some(seg) = map.get_mut(&hash.0) {
            seg.ref_count += 1;
            Ok(())
        } else {
            Err(StorageError::SegmentNotFound(hash.0.clone()))
        }
    }

    async fn replace_segment_data(
        &self,
        hash: &SegmentHash,
        new_data: Vec<u8>,
    ) -> StorageResult<()> {
        let mut map = self.segments.write().await;
        if let Some(seg) = map.get_mut(&hash.0) {
            seg.compressed_size = new_data.len() as u32;
            seg.compressed_data = new_data;
            Ok(())
        } else {
            Err(StorageError::SegmentNotFound(hash.0.clone()))
        }
    }

    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool> {
        let mut map = self.segments.write().await;
        if let Some(seg) = map.get_mut(&hash.0) {
            seg.ref_count = seg.ref_count.saturating_sub(1);
            Ok(seg.ref_count == 0)
        } else {
            Err(StorageError::SegmentNotFound(hash.0.clone()))
        }
    }

    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()> {
        self.segments.write().await.remove(&hash.0);
        Ok(())
    }

    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()> {
        self.manifests
            .write()
            .await
            .insert(manifest.id.clone(), manifest.clone());
        Ok(())
    }

    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest> {
        self.manifests
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::ConversationNotFound(id.to_owned()))
    }

    async fn delete_manifest(&self, id: &str) -> StorageResult<()> {
        self.manifests.write().await.remove(id);
        Ok(())
    }

    async fn list_conversations(
        &self,
        _query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> StorageResult<Vec<String>> {
        let ids: Vec<String> = self.manifests.read().await.keys().cloned().collect();
        Ok(ids
            .into_iter()
            .skip(offset as usize)
            .take(limit as usize)
            .collect())
    }

    async fn garbage_collect(&self) -> StorageResult<u64> {
        let mut segs = self.segments.write().await;
        let before = segs.len() as u64;
        segs.retain(|_, seg| seg.ref_count > 0);
        Ok(before - segs.len() as u64)
    }

    async fn storage_size_bytes(&self) -> StorageResult<u64> {
        Ok(self
            .segments
            .read()
            .await
            .values()
            .map(|s| s.compressed_size as u64)
            .sum())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vault = Stowken::new(MyBackend::default(), StowkenConfig::default()).await?;

    let conv = Conversation {
        id: None,
        model: "custom-model".into(),
        tokenizer: "cl100k_base".into(),
        application: Some("custom-backend-demo".into()),
        metadata: None,
        messages: vec![Message {
            role: "user".into(),
            content: MessageContent::Text("Stored in my custom backend!".into()),
            name: None,
            tool_call_id: None,
        }],
    };

    let result = vault.store(conv).await?;
    println!(
        "Stored via custom backend: {} segments ({} new)",
        result.total_segments, result.new_segments
    );

    let stats = vault.stats().await?;
    println!("Storage size: {} bytes", stats.storage_bytes);

    Ok(())
}
