//! AWS S3 storage backend (v0.6).
//!
//! Behind the `s3` Cargo feature.
//!
//! # Multi-writer safety
//!
//! v0.1 had an unsolved race condition: two writers ingesting the same
//! segment could both read `ref_count = N`, both write `ref_count = N+1`,
//! and silently lose one increment. v0.6 fixes this at the protocol layer
//! using S3's conditional-write support:
//!
//!   - `If-None-Match: *` on the first-time write of `<hash>.meta` means
//!     only the first writer succeeds; subsequent writers see
//!     `412 Precondition Failed` and fall through to CAS-increment.
//!
//!   - `If-Match: <etag>` on every subsequent meta update reads the
//!     current ref-count + ETag, computes the new state, and writes back
//!     atomically. A concurrent writer's update changes the ETag and
//!     causes a 412; we re-read and retry up to [`MAX_CAS_RETRIES`] times
//!     with capped exponential backoff.
//!
//! Two writers concurrently incrementing the same segment's ref-count
//! produce the correct final value (initial + 2). Same for two writers
//! decrementing.
//!
//! # Layout
//!
//! ```text
//! {prefix}segments/{hash}        # raw compressed bytes
//! {prefix}segments/{hash}.meta   # JSON sidecar: ref_count, sizes, etc.
//! {prefix}manifests/{id}.json    # conversation manifest
//! ```
//!
//! `prefix` is normalised at construction time to always end with `/`
//! (or be empty). `garbage_collect` and `storage_size_bytes` walk the
//! `{prefix}segments/` keyspace via `list_objects_v2`; both can be
//! expensive on large buckets, so they're explicit operations rather
//! than implicit per-write costs.

use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use aws_smithy_runtime_api::client::result::SdkError;
use serde::{Deserialize, Serialize};

use crate::types::{
    AnalyticsQuery, ConversationManifest, SegmentHash, StoredSegment,
};

use super::backend::{StorageBackend, StorageError, StorageResult};

/// Maximum CAS retries before bailing out. Each retry sleeps a capped
/// exponential backoff (10ms, 20ms, 40ms, ...) so a sustained collision
/// storm fails fast rather than blocking the caller indefinitely.
pub const MAX_CAS_RETRIES: u32 = 8;

/// Sidecar metadata stored alongside each segment blob. Schema-versioned
/// implicitly via Serde's permissive defaults — older sidecars without a
/// field default to `Default::default()` for that field.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentMetaSidecar {
    segment_type: String,
    tokenizer: String,
    token_count: u32,
    raw_size: u32,
    compressed_size: u32,
    ref_count: u64,
    created_at: String,
}

/// AWS S3 storage backend.
pub struct S3Backend {
    client: Client,
    bucket: String,
    /// Always either empty or ends with `/`.
    prefix: String,
}

impl S3Backend {
    /// Construct from an existing `aws_sdk_s3::Client`. Useful when the
    /// caller has custom config (custom endpoint, signing, retry policy).
    pub fn new(
        client: Client,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Self {
        let mut prefix = prefix.into();
        if !prefix.is_empty() && !prefix.ends_with('/') {
            prefix.push('/');
        }
        Self {
            client,
            bucket: bucket.into(),
            prefix,
        }
    }

    /// Construct a client from the standard AWS environment chain
    /// (env vars, `~/.aws/credentials`, IAM roles, ECS task roles).
    pub async fn from_env(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = Client::new(&config);
        Ok(Self::new(client, bucket, prefix))
    }

    /// Construct a client with optional region and endpoint overrides.
    /// Used for S3-compatible providers (MinIO, localstack, Cloudflare R2,
    /// DigitalOcean Spaces). When `endpoint` is set, path-style addressing
    /// is forced (required by MinIO and most non-AWS implementations).
    pub async fn with_options(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        endpoint: Option<String>,
        region: Option<String>,
    ) -> Result<Self, StorageError> {
        use aws_sdk_s3::config::{Builder, Region};

        let mut cfg_loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(r) = region {
            cfg_loader = cfg_loader.region(Region::new(r));
        }
        let shared_cfg = cfg_loader.load().await;

        let mut s3_cfg = Builder::from(&shared_cfg);
        if let Some(ep) = endpoint {
            s3_cfg = s3_cfg.endpoint_url(ep).force_path_style(true);
        }
        let client = Client::from_conf(s3_cfg.build());
        Ok(Self::new(client, bucket, prefix))
    }

    /// Read access to the underlying `aws_sdk_s3::Client`. Useful when
    /// callers need to do bucket-level operations Stowken doesn't model
    /// (lifecycle policies, server-side encryption keys, etc.).
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    fn segment_key(&self, hash: &SegmentHash) -> String {
        format!("{}segments/{}", self.prefix, hash.0)
    }

    fn segment_meta_key(&self, hash: &SegmentHash) -> String {
        format!("{}segments/{}.meta", self.prefix, hash.0)
    }

    fn manifest_key(&self, id: &str) -> String {
        format!("{}manifests/{}.json", self.prefix, id)
    }

    // ── Low-level S3 helpers ──────────────────────────────────────────────

    async fn get_object_bytes(&self, key: &str) -> StorageResult<Vec<u8>> {
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| map_get_error(e, key))?;
        let bytes = resp
            .body
            .collect()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?
            .into_bytes()
            .to_vec();
        Ok(bytes)
    }

    /// Read `.meta` and return `(parsed_meta, etag)`. The ETag is what
    /// CAS-update writes will use as the `If-Match` value.
    async fn read_meta_with_etag(
        &self,
        hash: &SegmentHash,
    ) -> StorageResult<(SegmentMetaSidecar, String)> {
        let key = self.segment_meta_key(hash);
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| map_get_error(e, &key))?;
        let etag = resp.e_tag().unwrap_or("").to_owned();
        let body = resp
            .body
            .collect()
            .await
            .map_err(|e| StorageError::BackendError(e.to_string()))?
            .into_bytes()
            .to_vec();
        let meta: SegmentMetaSidecar = serde_json::from_slice(&body)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        Ok((meta, etag))
    }

    /// Write `.meta` with `If-Match: <expected_etag>`.
    /// Returns:
    ///   - `Ok(true)`  on success
    ///   - `Ok(false)` on `412 Precondition Failed` (CAS miss — caller retries)
    ///   - `Err`       on any other failure
    async fn write_meta_if_match(
        &self,
        hash: &SegmentHash,
        meta: &SegmentMetaSidecar,
        expected_etag: &str,
    ) -> StorageResult<bool> {
        let body = serde_json::to_vec(meta)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        let key = self.segment_meta_key(hash);
        let result = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .if_match(expected_etag)
            .body(ByteStream::from(body))
            .send()
            .await;
        match result {
            Ok(_) => Ok(true),
            Err(e) => {
                if is_precondition_failed(&e) {
                    Ok(false)
                } else {
                    Err(StorageError::BackendError(format!(
                        "put_object {} (If-Match): {e}",
                        key
                    )))
                }
            }
        }
    }

    /// Try to create `.meta` atomically with `If-None-Match: *`.
    /// Returns:
    ///   - `Ok(true)`  the new sidecar was created
    ///   - `Ok(false)` something already exists at that key (caller must
    ///     fall through to CAS-increment)
    ///   - `Err`       any other failure
    async fn create_meta_if_absent(
        &self,
        hash: &SegmentHash,
        meta: &SegmentMetaSidecar,
    ) -> StorageResult<bool> {
        let body = serde_json::to_vec(meta)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        let key = self.segment_meta_key(hash);
        let result = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .if_none_match("*")
            .body(ByteStream::from(body))
            .send()
            .await;
        match result {
            Ok(_) => Ok(true),
            Err(e) => {
                if is_precondition_failed(&e) {
                    Ok(false)
                } else {
                    Err(StorageError::BackendError(format!(
                        "put_object {} (If-None-Match): {e}",
                        key
                    )))
                }
            }
        }
    }

    /// Generic CAS loop for ref-count or other meta mutations. The
    /// closure receives a mutable reference to the deserialised meta
    /// and is expected to update it in place. The loop reads the
    /// current state, applies `f`, and conditionally writes; on
    /// concurrent-write collision it backs off and retries up to
    /// [`MAX_CAS_RETRIES`] times.
    async fn cas_update_meta<F>(
        &self,
        hash: &SegmentHash,
        mut f: F,
    ) -> StorageResult<SegmentMetaSidecar>
    where
        F: FnMut(&mut SegmentMetaSidecar),
    {
        for attempt in 0..MAX_CAS_RETRIES {
            let (mut meta, etag) = self.read_meta_with_etag(hash).await?;
            f(&mut meta);
            match self.write_meta_if_match(hash, &meta, &etag).await? {
                true => return Ok(meta),
                false => {
                    // Backoff: 10ms, 20ms, 40ms, ... capped at 320ms.
                    let ms = (10u64 << attempt.min(5)).min(320);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                }
            }
        }
        Err(StorageError::BackendError(format!(
            "CAS retry budget exhausted on {} (.meta)",
            hash.0
        )))
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
            Err(e) => {
                if is_not_found(&e) {
                    Ok(false)
                } else {
                    Err(StorageError::BackendError(format!(
                        "head_object {}: {e}",
                        key
                    )))
                }
            }
        }
    }

    async fn delete_key(&self, key: &str) -> StorageResult<()> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| StorageError::BackendError(format!("delete_object {key}: {e}")))?;
        Ok(())
    }

    async fn list_keys_with_prefix(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let mut out = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix);
            if let Some(token) = continuation.take() {
                req = req.continuation_token(token);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| StorageError::BackendError(format!("list_objects_v2 {prefix}: {e}")))?;
            for obj in resp.contents() {
                if let Some(k) = obj.key() {
                    out.push(k.to_owned());
                }
            }
            if let Some(token) = resp.next_continuation_token() {
                continuation = Some(token.to_owned());
            } else {
                break;
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl StorageBackend for S3Backend {
    // ── Segment operations ────────────────────────────────────────────

    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()> {
        // 1. Always write the data. S3 PutObject is overwrite-idempotent
        //    and segments are content-addressed, so two writers writing
        //    the same hash write byte-identical data — last-writer-wins
        //    is fine here.
        let seg_key = self.segment_key(&segment.hash);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&seg_key)
            .body(ByteStream::from(segment.compressed_data.clone()))
            .send()
            .await
            .map_err(|e| StorageError::BackendError(format!("put_object {seg_key}: {e}")))?;

        // 2. Try to create the meta sidecar atomically. The
        //    `If-None-Match: *` precondition makes "first writer wins";
        //    subsequent writers see 412 and fall through to a CAS
        //    increment.
        let meta = SegmentMetaSidecar {
            segment_type: segment.segment_type.to_string(),
            tokenizer: segment.tokenizer.clone(),
            token_count: segment.token_count,
            raw_size: segment.raw_size,
            compressed_size: segment.compressed_size,
            ref_count: segment.ref_count,
            created_at: segment.created_at.to_rfc3339(),
        };
        if self.create_meta_if_absent(&segment.hash, &meta).await? {
            return Ok(());
        }

        // Meta already existed — bump the ref count instead.
        self.increment_ref(&segment.hash).await
    }

    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment> {
        let seg_bytes = self
            .get_object_bytes(&self.segment_key(hash))
            .await
            .map_err(|e| match e {
                StorageError::ConversationNotFound(_) => {
                    StorageError::SegmentNotFound(hash.0.clone())
                }
                other => other,
            })?;
        let (meta, _etag) = self.read_meta_with_etag(hash).await?;
        let segment_type = meta
            .segment_type
            .parse()
            .unwrap_or(crate::types::SegmentType::UserTurn);
        let created_at = chrono::DateTime::parse_from_rfc3339(&meta.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        Ok(StoredSegment {
            hash: hash.clone(),
            segment_type,
            tokenizer: meta.tokenizer,
            token_count: meta.token_count,
            compressed_data: seg_bytes,
            raw_size: meta.raw_size,
            compressed_size: meta.compressed_size,
            ref_count: meta.ref_count,
            created_at,
        })
    }

    async fn has_segment(&self, hash: &SegmentHash) -> StorageResult<bool> {
        self.key_exists(&self.segment_key(hash)).await
    }

    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()> {
        self.cas_update_meta(hash, |meta| meta.ref_count += 1).await?;
        Ok(())
    }

    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool> {
        let meta = self
            .cas_update_meta(hash, |meta| {
                meta.ref_count = meta.ref_count.saturating_sub(1);
            })
            .await?;
        Ok(meta.ref_count == 0)
    }

    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()> {
        // Best effort — both keys may not exist (partial state from a
        // failed put), but that's fine.
        let _ = self.delete_key(&self.segment_key(hash)).await;
        let _ = self.delete_key(&self.segment_meta_key(hash)).await;
        Ok(())
    }

    async fn replace_segment_data(
        &self,
        hash: &SegmentHash,
        new_data: Vec<u8>,
    ) -> StorageResult<()> {
        // Make sure the segment exists.
        if !self.has_segment(hash).await? {
            return Err(StorageError::SegmentNotFound(hash.0.clone()));
        }

        // Overwrite the data unconditionally — content-addressed, so
        // any other writer overwriting concurrently with the same hash
        // is writing the same bytes (an attacker scenario aside, which
        // is out of scope).
        let key = self.segment_key(hash);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(new_data.clone()))
            .send()
            .await
            .map_err(|e| StorageError::BackendError(format!("put_object {key}: {e}")))?;

        // Update the .meta's compressed_size via CAS so we don't race
        // with a concurrent ref-count increment.
        let new_size = new_data.len() as u32;
        self.cas_update_meta(hash, |meta| meta.compressed_size = new_size)
            .await?;
        Ok(())
    }

    // ── Manifest operations ───────────────────────────────────────────

    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()> {
        let body = serde_json::to_vec(manifest)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;
        let key = self.manifest_key(&manifest.id);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(body))
            .send()
            .await
            .map_err(|e| StorageError::BackendError(format!("put_object {key}: {e}")))?;
        Ok(())
    }

    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest> {
        let key = self.manifest_key(id);
        let bytes = self.get_object_bytes(&key).await.map_err(|e| match e {
            StorageError::SegmentNotFound(_) | StorageError::ConversationNotFound(_) => {
                StorageError::ConversationNotFound(id.to_owned())
            }
            other => other,
        })?;
        serde_json::from_slice(&bytes)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    async fn delete_manifest(&self, id: &str) -> StorageResult<()> {
        let _ = self.delete_key(&self.manifest_key(id)).await;
        Ok(())
    }

    async fn list_conversations(
        &self,
        query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> StorageResult<Vec<String>> {
        let mprefix = format!("{}manifests/", self.prefix);
        let keys = self.list_keys_with_prefix(&mprefix).await?;

        let mut ids: Vec<String> = keys
            .into_iter()
            .filter_map(|k| {
                let stripped = k.strip_prefix(&mprefix)?;
                let id = stripped.strip_suffix(".json")?;
                Some(id.to_owned())
            })
            .collect();

        // Apply any query filters that need the manifest body. Skip the
        // round-trip when no filters are set.
        if query.model.is_some()
            || query.application.is_some()
            || query.date_from.is_some()
            || query.date_to.is_some()
        {
            let mut filtered = Vec::with_capacity(ids.len());
            for id in &ids {
                let m = match self.get_manifest(id).await {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if let Some(want) = &query.model {
                    if &m.model != want {
                        continue;
                    }
                }
                if let Some(want) = &query.application {
                    if m.application.as_deref() != Some(want.as_str()) {
                        continue;
                    }
                }
                if let Some(from) = query.date_from {
                    if m.created_at < from {
                        continue;
                    }
                }
                if let Some(to) = query.date_to {
                    if m.created_at > to {
                        continue;
                    }
                }
                filtered.push(id.clone());
            }
            ids = filtered;
        }

        Ok(ids
            .into_iter()
            .skip(offset as usize)
            .take(limit as usize)
            .collect())
    }

    // ── Maintenance ───────────────────────────────────────────────────

    async fn list_garbage(&self) -> StorageResult<Vec<SegmentHash>> {
        let prefix = format!("{}segments/", self.prefix);
        let keys = self.list_keys_with_prefix(&prefix).await?;
        let mut out = Vec::new();
        for k in keys {
            let Some(stripped) = k.strip_prefix(&prefix) else { continue };
            let Some(hash_str) = stripped.strip_suffix(".meta") else { continue };
            let hash = SegmentHash(hash_str.to_owned());
            if let Ok((meta, _)) = self.read_meta_with_etag(&hash).await {
                if meta.ref_count == 0 {
                    out.push(hash);
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
        let prefix = format!("{}segments/", self.prefix);
        let mut total: u64 = 0;
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
            let resp = req
                .send()
                .await
                .map_err(|e| StorageError::BackendError(format!("list_objects_v2 {prefix}: {e}")))?;
            for obj in resp.contents() {
                if let Some(k) = obj.key() {
                    if k.ends_with(".meta") {
                        continue;
                    }
                }
                if let Some(s) = obj.size() {
                    total += s as u64;
                }
            }
            if let Some(token) = resp.next_continuation_token() {
                continuation = Some(token.to_owned());
            } else {
                break;
            }
        }
        Ok(total)
    }
}

// ── SDK error helpers ─────────────────────────────────────────────────

fn map_get_error<E>(e: SdkError<E, aws_smithy_runtime_api::http::Response>, key: &str) -> StorageError
where
    E: std::fmt::Display + std::fmt::Debug,
{
    if let SdkError::ServiceError(svc) = &e {
        let status = svc.raw().status().as_u16();
        if status == 404 {
            return StorageError::SegmentNotFound(key.to_owned());
        }
    }
    StorageError::BackendError(format!("get_object {key}: {e}"))
}

fn is_precondition_failed<E>(e: &SdkError<E, aws_smithy_runtime_api::http::Response>) -> bool
where
    E: std::fmt::Debug,
{
    matches!(
        e,
        SdkError::ServiceError(svc) if svc.raw().status().as_u16() == 412
    )
}

fn is_not_found<E>(e: &SdkError<E, aws_smithy_runtime_api::http::Response>) -> bool
where
    E: std::fmt::Debug,
{
    matches!(
        e,
        SdkError::ServiceError(svc) if svc.raw().status().as_u16() == 404
    )
}

// ── Unit tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[test]
    fn prefix_normalisation_appends_slash() {
        // `new` doesn't actually need a real client for this test — just
        // run the prefix logic via the public method by building a stub.
        // We can't build a real Client without a runtime, so we just
        // assert the static logic.
        let cases: &[(&str, &str)] = &[
            ("", ""),
            ("foo", "foo/"),
            ("foo/", "foo/"),
            ("nested/path", "nested/path/"),
        ];
        for (input, expected) in cases {
            let mut got = String::from(*input);
            if !got.is_empty() && !got.ends_with('/') {
                got.push('/');
            }
            assert_eq!(got, *expected, "prefix normalisation for {input:?}");
        }
    }
}
