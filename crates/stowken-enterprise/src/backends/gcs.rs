//! Google Cloud Storage backend.
//!
//! Layout mirrors the S3 backend:
//!   `{prefix}segments/{hash}`       — raw compressed segment bytes
//!   `{prefix}segments/{hash}.meta`  — JSON-encoded segment metadata
//!   `{prefix}manifests/{id}.json`   — JSON-encoded conversation manifest

use std::collections::HashMap;

use async_trait::async_trait;
use google_cloud_storage::{
    client::{Client, ClientConfig},
    http::objects::{
        delete::DeleteObjectRequest,
        download::Range,
        get::GetObjectRequest,
        list::ListObjectsRequest,
        upload::{Media, UploadObjectRequest, UploadType},
    },
};
use serde::{Deserialize, Serialize};

use stowken::{
    storage::{StorageBackend, StorageError, StorageResult},
    types::{
        AnalyticsQuery, ConversationManifest, SegmentHash, SegmentTypeStats, StoredSegment,
        TokenUsageStats,
    },
};

#[derive(Debug, Serialize, Deserialize)]
struct SegmentMeta {
    segment_type: String,
    tokenizer: String,
    token_count: u32,
    raw_size: u32,
    compressed_size: u32,
    ref_count: u64,
    created_at: String,
}

fn gcs_err(e: impl std::fmt::Display) -> StorageError {
    StorageError::BackendError(e.to_string())
}

/// Google Cloud Storage-backed storage backend.
///
/// # Construction
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use stowken_enterprise::backends::GcsBackend;
/// let backend = GcsBackend::from_env("my-bucket", "stowken/").await?;
/// # Ok(()) }
/// ```
pub struct GcsBackend {
    client: Client,
    bucket: String,
    prefix: String,
}

impl GcsBackend {
    pub fn new(client: Client, bucket: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            client,
            bucket: bucket.into(),
            prefix: prefix.into(),
        }
    }

    /// Construct by loading Application Default Credentials from the environment.
    pub async fn from_env(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let config = ClientConfig::default().with_auth().await.map_err(gcs_err)?;
        let client = Client::new(config);
        Ok(Self::new(client, bucket, prefix))
    }

    fn seg_key(&self, hash: &SegmentHash) -> String {
        format!("{}segments/{}", self.prefix, hash.0)
    }

    fn seg_meta_key(&self, hash: &SegmentHash) -> String {
        format!("{}segments/{}.meta", self.prefix, hash.0)
    }

    fn manifest_key(&self, id: &str) -> String {
        format!("{}manifests/{id}.json", self.prefix)
    }

    async fn get_bytes(&self, key: &str) -> StorageResult<Vec<u8>> {
        self.client
            .download_object(
                &GetObjectRequest {
                    bucket: self.bucket.clone(),
                    object: key.to_owned(),
                    ..Default::default()
                },
                &Range::default(),
            )
            .await
            .map_err(|e| {
                let msg = e.to_string();
                if msg.contains("404") || msg.contains("notFound") {
                    StorageError::SegmentNotFound(key.to_owned())
                } else {
                    gcs_err(e)
                }
            })
    }

    async fn put_bytes(&self, key: &str, data: Vec<u8>, content_type: &str) -> StorageResult<()> {
        let upload_type = UploadType::Simple(Media::new(key.to_owned()));
        self.client
            .upload_object(
                &UploadObjectRequest {
                    bucket: self.bucket.clone(),
                    ..Default::default()
                },
                data,
                &upload_type,
            )
            .await
            .map_err(gcs_err)?;
        let _ = content_type;
        Ok(())
    }

    async fn delete_key(&self, key: &str) -> StorageResult<()> {
        self.client
            .delete_object(&DeleteObjectRequest {
                bucket: self.bucket.clone(),
                object: key.to_owned(),
                ..Default::default()
            })
            .await
            .map_err(gcs_err)?;
        Ok(())
    }

    async fn key_exists(&self, key: &str) -> StorageResult<bool> {
        match self.get_bytes(key).await {
            Ok(_) => Ok(true),
            Err(StorageError::SegmentNotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn read_meta(&self, hash: &SegmentHash) -> StorageResult<SegmentMeta> {
        let data = self.get_bytes(&self.seg_meta_key(hash)).await?;
        serde_json::from_slice(&data).map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    async fn write_meta(&self, hash: &SegmentHash, meta: &SegmentMeta) -> StorageResult<()> {
        let data = serde_json::to_vec(meta)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        self.put_bytes(&self.seg_meta_key(hash), data, "application/json").await
    }
}

#[async_trait]
impl StorageBackend for GcsBackend {
    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()> {
        let seg_key = self.seg_key(&segment.hash);
        if self.key_exists(&seg_key).await? {
            let mut meta = self.read_meta(&segment.hash).await?;
            meta.ref_count += 1;
            return self.write_meta(&segment.hash, &meta).await;
        }

        self.put_bytes(&seg_key, segment.compressed_data.clone(), "application/octet-stream")
            .await?;

        let meta = SegmentMeta {
            segment_type: segment.segment_type.to_string(),
            tokenizer: segment.tokenizer.clone(),
            token_count: segment.token_count,
            raw_size: segment.raw_size,
            compressed_size: segment.compressed_size,
            ref_count: segment.ref_count,
            created_at: segment.created_at.to_rfc3339(),
        };
        self.write_meta(&segment.hash, &meta).await
    }

    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment> {
        let compressed_data = self.get_bytes(&self.seg_key(hash)).await.map_err(|_| {
            StorageError::SegmentNotFound(hash.0.clone())
        })?;
        let meta = self.read_meta(hash).await?;
        let created_at = chrono::DateTime::parse_from_rfc3339(&meta.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let segment_type: stowken::types::SegmentType =
            meta.segment_type.parse().unwrap_or(stowken::types::SegmentType::UserTurn);

        Ok(StoredSegment {
            hash: hash.clone(),
            segment_type,
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
        self.key_exists(&self.seg_key(hash)).await
    }

    async fn has_segments(&self, hashes: &[SegmentHash]) -> StorageResult<Vec<SegmentHash>> {
        let mut present = Vec::new();
        for hash in hashes {
            if self.key_exists(&self.seg_key(hash)).await? {
                present.push(hash.clone());
            }
        }
        Ok(present)
    }

    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()> {
        let mut meta = self.read_meta(hash).await?;
        meta.ref_count += 1;
        self.write_meta(hash, &meta).await
    }

    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool> {
        let mut meta = self.read_meta(hash).await?;
        meta.ref_count = meta.ref_count.saturating_sub(1);
        self.write_meta(hash, &meta).await?;
        Ok(meta.ref_count == 0)
    }

    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()> {
        let _ = self.delete_key(&self.seg_key(hash)).await;
        let _ = self.delete_key(&self.seg_meta_key(hash)).await;
        Ok(())
    }

    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()> {
        let data = serde_json::to_vec(manifest)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        self.put_bytes(&self.manifest_key(&manifest.id), data, "application/json").await
    }

    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest> {
        let key = self.manifest_key(id);
        let data = self.get_bytes(&key).await.map_err(|_| {
            StorageError::ConversationNotFound(id.to_owned())
        })?;
        serde_json::from_slice(&data).map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    async fn delete_manifest(&self, id: &str) -> StorageResult<()> {
        let _ = self.delete_key(&self.manifest_key(id)).await;
        Ok(())
    }

    async fn list_conversations(
        &self,
        _query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> StorageResult<Vec<String>> {
        let prefix = format!("{}manifests/", self.prefix);
        let resp = self
            .client
            .list_objects(&ListObjectsRequest {
                bucket: self.bucket.clone(),
                prefix: Some(prefix.clone()),
                ..Default::default()
            })
            .await
            .map_err(gcs_err)?;

        let ids: Vec<String> = resp
            .items
            .unwrap_or_default()
            .into_iter()
            .filter_map(|obj| {
                let name = obj.name.strip_prefix(&prefix)?.to_owned();
                Some(name.trim_end_matches(".json").to_owned())
            })
            .skip(offset as usize)
            .take(limit as usize)
            .collect();

        Ok(ids)
    }

    async fn get_stats(&self) -> StorageResult<TokenUsageStats> {
        Ok(TokenUsageStats {
            total_tokens: 0,
            total_conversations: 0,
            unique_segments: 0,
            total_segments: 0,
            dedup_ratio: 0.0,
            compression_ratio: 1.0,
            storage_bytes: 0,
            naive_bytes: 0,
            savings_percentage: 0.0,
        })
    }

    async fn get_segment_type_stats(&self) -> StorageResult<Vec<SegmentTypeStats>> {
        Ok(vec![])
    }

    async fn query_analytics(
        &self,
        _query: &AnalyticsQuery,
    ) -> StorageResult<Vec<HashMap<String, serde_json::Value>>> {
        Ok(vec![])
    }

    async fn list_unique_system_prompts(&self) -> StorageResult<Vec<(SegmentHash, u32, u64)>> {
        Ok(vec![])
    }

    async fn find_conversations_by_segment(
        &self,
        _hash: &SegmentHash,
    ) -> StorageResult<Vec<String>> {
        Ok(vec![])
    }

    async fn garbage_collect(&self) -> StorageResult<u64> {
        Ok(0)
    }

    async fn storage_size_bytes(&self) -> StorageResult<u64> {
        let prefix = format!("{}segments/", self.prefix);
        let resp = self
            .client
            .list_objects(&ListObjectsRequest {
                bucket: self.bucket.clone(),
                prefix: Some(prefix),
                ..Default::default()
            })
            .await
            .map_err(gcs_err)?;

        let total: u64 = resp
            .items
            .unwrap_or_default()
            .into_iter()
            .map(|obj| obj.size as u64)
            .sum();
        Ok(total)
    }
}
