//! AWS S3 storage backend.
//!
//! Layout:
//!   `{prefix}segments/{hash}`       — raw compressed segment bytes
//!   `{prefix}segments/{hash}.meta`  — JSON-encoded segment metadata
//!   `{prefix}manifests/{id}.json`   — JSON-encoded conversation manifest

use std::collections::HashMap;

use async_trait::async_trait;
use aws_sdk_s3::{
    error::SdkError,
    operation::{
        delete_object::DeleteObjectError, get_object::GetObjectError,
        head_object::HeadObjectError, put_object::PutObjectError,
    },
    primitives::ByteStream,
    Client,
};
use serde::{Deserialize, Serialize};

use stowken::{
    storage::{StorageBackend, StorageError, StorageResult},
    types::{
        AnalyticsQuery, ConversationManifest, SegmentHash, SegmentTypeStats, StoredSegment,
        TokenUsageStats,
    },
};

/// Metadata sidecar stored alongside each segment blob.
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

fn s3_err(e: impl std::fmt::Display) -> StorageError {
    StorageError::BackendError(e.to_string())
}

fn map_get_err(e: SdkError<GetObjectError>, key: &str) -> StorageError {
    match &e {
        SdkError::ServiceError(svc) if svc.err().is_no_such_key() => {
            if key.ends_with(".meta") || key.contains("manifests/") {
                StorageError::ConversationNotFound(key.to_owned())
            } else {
                StorageError::SegmentNotFound(key.to_owned())
            }
        }
        _ => s3_err(e),
    }
}

fn map_head_err(e: SdkError<HeadObjectError>, key: &str) -> StorageError {
    match &e {
        SdkError::ServiceError(svc) => match svc.err() {
            HeadObjectError::NotFound(_) => StorageError::SegmentNotFound(key.to_owned()),
            _ => s3_err(e),
        },
        _ => s3_err(e),
    }
}

/// AWS S3-backed storage backend.
///
/// # Construction
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use stowken_enterprise::backends::S3Backend;
/// let backend = S3Backend::from_env("my-bucket", "stowken/").await?;
/// # Ok(()) }
/// ```
pub struct S3Backend {
    client: Client,
    bucket: String,
    /// Optional key prefix (e.g. `"stowken/"`)
    prefix: String,
}

impl S3Backend {
    /// Construct from an existing SDK client.
    pub fn new(client: Client, bucket: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            client,
            bucket: bucket.into(),
            prefix: prefix.into(),
        }
    }

    /// Construct by loading credentials from the environment (standard AWS env vars / ~/.aws).
    pub async fn from_env(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = Client::new(&config);
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
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| map_get_err(e, key))?;
        resp.body
            .collect()
            .await
            .map(|d| d.into_bytes().to_vec())
            .map_err(s3_err)
    }

    async fn put_bytes(&self, key: &str, data: Vec<u8>) -> StorageResult<()> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(data))
            .send()
            .await
            .map_err(|e: SdkError<PutObjectError>| s3_err(e))?;
        Ok(())
    }

    async fn delete_key(&self, key: &str) -> StorageResult<()> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e: SdkError<DeleteObjectError>| s3_err(e))?;
        Ok(())
    }

    async fn key_exists(&self, key: &str) -> StorageResult<bool> {
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => match map_head_err(e, key) {
                StorageError::SegmentNotFound(_) => Ok(false),
                other => Err(other),
            },
        }
    }

    async fn read_meta(&self, hash: &SegmentHash) -> StorageResult<SegmentMeta> {
        let data = self.get_bytes(&self.seg_meta_key(hash)).await?;
        serde_json::from_slice(&data).map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    async fn write_meta(&self, hash: &SegmentHash, meta: &SegmentMeta) -> StorageResult<()> {
        let data = serde_json::to_vec(meta)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        self.put_bytes(&self.seg_meta_key(hash), data).await
    }
}

#[async_trait]
impl StorageBackend for S3Backend {
    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()> {
        let seg_key = self.seg_key(&segment.hash);
        if self.key_exists(&seg_key).await? {
            // Already stored — increment ref count
            let mut meta = self.read_meta(&segment.hash).await?;
            meta.ref_count += 1;
            return self.write_meta(&segment.hash, &meta).await;
        }

        self.put_bytes(&seg_key, segment.compressed_data.clone()).await?;

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
        let compressed_data = self
            .get_bytes(&self.seg_key(hash))
            .await
            .map_err(|e| match e {
                StorageError::ConversationNotFound(_) => {
                    StorageError::SegmentNotFound(hash.0.clone())
                }
                other => other,
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
        self.put_bytes(&self.manifest_key(&manifest.id), data).await
    }

    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest> {
        let key = self.manifest_key(id);
        let data = self.get_bytes(&key).await.map_err(|e| match e {
            StorageError::ConversationNotFound(_) | StorageError::SegmentNotFound(_) => {
                StorageError::ConversationNotFound(id.to_owned())
            }
            other => other,
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
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&prefix)
            .send()
            .await
            .map_err(s3_err)?;

        let ids: Vec<String> = resp
            .contents()
            .iter()
            .filter_map(|obj| {
                let key = obj.key()?;
                let name = key.strip_prefix(&prefix)?;
                Some(name.trim_end_matches(".json").to_owned())
            })
            .skip(offset as usize)
            .take(limit as usize)
            .collect();

        Ok(ids)
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
        // List all `.meta` sidecar files, read their ref-counts, and delete
        // the data + meta objects for any segment with ref_count == 0.
        let meta_prefix = format!("{}segments/", self.prefix);
        let mut deleted = 0u64;
        let mut continuation: Option<String> = None;

        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&meta_prefix);
            if let Some(token) = continuation.take() {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.map_err(s3_err)?;

            for obj in resp.contents() {
                let key = match obj.key() {
                    Some(k) if k.ends_with(".meta") => k,
                    _ => continue,
                };
                // Derive the SegmentHash from the meta key.
                let hash_str = key
                    .strip_prefix(&meta_prefix)
                    .and_then(|s| s.strip_suffix(".meta"))
                    .unwrap_or_default();
                let hash = SegmentHash(hash_str.to_owned());

                let meta = match self.read_meta(&hash).await {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if meta.ref_count == 0 {
                    let _ = self.delete_key(&self.seg_key(&hash)).await;
                    let _ = self.delete_key(&self.seg_meta_key(&hash)).await;
                    deleted += 1;
                }
            }

            match resp.next_continuation_token() {
                Some(token) => continuation = Some(token.to_owned()),
                None => break,
            }
        }
        Ok(deleted)
    }

    async fn get_stats(&self) -> StorageResult<TokenUsageStats> {
        // Walk all segment meta files to compute aggregate statistics.
        let meta_prefix = format!("{}segments/", self.prefix);
        let manifest_prefix = format!("{}manifests/", self.prefix);

        let mut unique_segments: u64 = 0;
        let mut total_segments: u64 = 0;
        let mut total_tokens: u64 = 0;
        let mut storage_bytes: u64 = 0;
        let mut naive_bytes: u64 = 0;
        let mut continuation: Option<String> = None;

        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&meta_prefix);
            if let Some(token) = continuation.take() {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.map_err(s3_err)?;

            for obj in resp.contents() {
                let key = match obj.key() {
                    Some(k) if k.ends_with(".meta") => k,
                    _ => continue,
                };
                let hash_str = key
                    .strip_prefix(&meta_prefix)
                    .and_then(|s| s.strip_suffix(".meta"))
                    .unwrap_or_default();
                let hash = SegmentHash(hash_str.to_owned());

                if let Ok(meta) = self.read_meta(&hash).await {
                    if meta.ref_count > 0 {
                        unique_segments += 1;
                        total_segments += meta.ref_count;
                        total_tokens += meta.token_count as u64 * meta.ref_count;
                        storage_bytes += meta.compressed_size as u64;
                        naive_bytes += meta.raw_size as u64 * meta.ref_count;
                    }
                }
            }

            match resp.next_continuation_token() {
                Some(token) => continuation = Some(token.to_owned()),
                None => break,
            }
        }

        // Count manifests for total_conversations.
        let mut total_conversations: u64 = 0;
        continuation = None;
        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&manifest_prefix);
            if let Some(token) = continuation.take() {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.map_err(s3_err)?;
            total_conversations += resp.contents().len() as u64;
            match resp.next_continuation_token() {
                Some(token) => continuation = Some(token.to_owned()),
                None => break,
            }
        }

        let dedup_ratio = if total_segments > 0 {
            1.0 - (unique_segments as f64 / total_segments as f64)
        } else {
            0.0
        };
        let compression_ratio = if naive_bytes > 0 {
            storage_bytes as f64 / naive_bytes as f64
        } else {
            1.0
        };
        let savings_percentage = (1.0 - compression_ratio) * 100.0;

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

    async fn storage_size_bytes(&self) -> StorageResult<u64> {
        let prefix = format!("{}segments/", self.prefix);
        let mut total = 0u64;
        let mut continuation: Option<String> = None;

        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix);
            if let Some(token) = continuation.take() {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.map_err(s3_err)?;
            total += resp
                .contents()
                .iter()
                .filter(|obj| obj.key().map_or(false, |k| !k.ends_with(".meta")))
                .filter_map(|obj| obj.size())
                .map(|s| s as u64)
                .sum::<u64>();
            match resp.next_continuation_token() {
                Some(token) => continuation = Some(token.to_owned()),
                None => break,
            }
        }
        Ok(total)
    }
}
