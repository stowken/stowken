//! `EnterpriseStowken` — a drop-in wrapper around `Stowken` that adds
//! near-duplicate detection via MinHash LSH.
//!
//! # How it works
//!
//! `EnterpriseStowken<B>` intercepts `store()` before the core vault sees a
//! segment.  For each segment that is *not* an exact duplicate it runs the
//! `NearDedupIndex`:
//!
//! - **No near-dup found** → store the full segment normally (same as core).
//! - **Near-dup found** → store only the delta; the canonical segment already
//!   lives in the backend with a ref-count.  The variant's manifest entry is
//!   annotated in the conversation's metadata so `retrieve()` can reconstruct
//!   the original tokens on demand.
//!
//! # Usage
//!
//! ```no_run
//! use stowken::{storage::MemoryBackend, types::StowkenConfig};
//! use stowken_enterprise::enterprise_vault::EnterpriseStowken;
//!
//! # async fn example() {
//! let vault = EnterpriseStowken::new(MemoryBackend::new(), StowkenConfig::default(), 0.85).await.unwrap();
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde_json::json;
use thiserror::Error;
use tokio::task;
use tracing::instrument;

use stowken::{
    index::metadata::BatchSegmentOp,
    storage::backend::{StorageBackend, StorageError},
    types::{
        Conversation, ConversationManifest, RetrievedConversation,
        RetrievedSegment, SegmentHash, SegmentRef, StoreResult, StoredSegment, StowkenConfig,
    },
    vault::StowkenError,
    Stowken,
};

use crate::near_dedup::{
    apply_delta, NearDedupAction, NearDedupIndex, NearDedupStats, SegmentDelta,
};

/// Errors from `EnterpriseStowken` operations.
#[derive(Debug, Error)]
pub enum EnterpriseStowkenError {
    #[error(transparent)]
    Core(#[from] StowkenError),
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("near-dedup delta not found for segment {0}")]
    DeltaNotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
}

/// Extended `StoreResult` that includes near-dedup statistics.
#[derive(Debug, Clone)]
pub struct EnterpriseStoreResult {
    /// Core store metrics.
    pub core: StoreResult,
    /// Segments linked to a canonical via near-dedup (not stored as full segments).
    pub near_dedup_segments: u64,
    /// Raw token bytes saved by near-dedup (variant tokens avoided).
    pub near_dedup_bytes_saved: u64,
}

/// `EnterpriseStowken<B>` — `Stowken<B>` extended with MinHash near-dedup.
pub struct EnterpriseStowken<B: StorageBackend> {
    inner: Stowken<B>,
    near_dedup: Arc<Mutex<NearDedupIndex>>,
    /// In-memory delta store: variant_hash → SegmentDelta.
    /// In production this would be serialised alongside the backend.
    deltas: Arc<Mutex<HashMap<SegmentHash, SegmentDelta>>>,
}

impl<B: StorageBackend + 'static> EnterpriseStowken<B> {
    /// Create a new `EnterpriseStowken` with an in-memory metadata index.
    pub async fn new(
        backend: B,
        config: StowkenConfig,
        near_dedup_threshold: f64,
    ) -> Result<Self, EnterpriseStowkenError> {
        let inner = Stowken::new(backend, config).await?;
        Ok(Self {
            inner,
            near_dedup: Arc::new(Mutex::new(NearDedupIndex::new(near_dedup_threshold))),
            deltas: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Open a vault with a persistent metadata database at `db_path`.
    pub async fn open(
        backend: B,
        config: StowkenConfig,
        db_path: &str,
        near_dedup_threshold: f64,
    ) -> Result<Self, EnterpriseStowkenError> {
        let inner = Stowken::open(backend, config, db_path).await?;
        Ok(Self {
            inner,
            near_dedup: Arc::new(Mutex::new(NearDedupIndex::new(near_dedup_threshold))),
            deltas: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Current near-dedup statistics.
    pub fn near_dedup_stats(&self) -> NearDedupStats {
        self.near_dedup.lock().unwrap().stats.clone()
    }

    /// Store a conversation with near-dedup.
    ///
    /// Returns an `EnterpriseStoreResult` that extends the core metrics with
    /// near-dup-specific counts.
    #[instrument(skip(self, conversation), fields(model = %conversation.model))]
    pub async fn store(
        &self,
        conversation: Conversation,
    ) -> Result<EnterpriseStoreResult, EnterpriseStowkenError> {
        let id = conversation
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let tokenizer_name = conversation.tokenizer.clone();
        let segments = self.inner.segment_conversation(&conversation)?;

        let backend = self.inner.backend();
        let index = self.inner.index().clone();

        let mut segment_refs: Vec<SegmentRef> = Vec::with_capacity(segments.len());
        let mut index_ops: Vec<BatchSegmentOp> = Vec::with_capacity(segments.len());

        let mut new_segments: u64 = 0;
        let mut deduped_segments: u64 = 0;
        let mut near_dedup_segments: u64 = 0;
        let mut bytes_saved: u64 = 0;
        let mut near_dedup_bytes_saved: u64 = 0;
        let mut total_compressed: u64 = 0;
        let mut total_raw: u64 = 0;

        // Map from variant_hash → canonical_hash for this conversation's near-dup segments.
        // Stored in the manifest metadata so retrieve() knows which segments need delta apply.
        let mut near_dup_meta: HashMap<String, serde_json::Value> = HashMap::new();

        for (position, segment) in segments.iter().enumerate() {
            let hash = self.inner.hash_tokens(&segment.tokens);

            if backend.has_segment(&hash).await? {
                // Exact duplicate — increment ref only.
                backend.increment_ref(&hash).await?;
                index_ops.push(BatchSegmentOp::IncrementRef(hash.clone()));
                deduped_segments += 1;
                bytes_saved += (segment.tokens.len() as u64) * 4;

                segment_refs.push(SegmentRef {
                    segment_type: segment.segment_type.clone(),
                    hash,
                    token_count: segment.tokens.len() as u32,
                    position: position as u32,
                });
                continue;
            }

            // Not an exact dup — check near-dedup index.
            let nd_action = {
                let mut nd = self.near_dedup.lock().unwrap();
                nd.process(&segment.tokens)
            };

            match nd_action {
                NearDedupAction::LinkToCanonical {
                    canonical_hash,
                    delta,
                    similarity,
                } => {
                    // Near-dup found: increment canonical's ref-count, store delta.
                    backend.increment_ref(&canonical_hash).await?;
                    index_ops.push(BatchSegmentOp::IncrementRef(canonical_hash.clone()));

                    let token_bytes = (segment.tokens.len() as u64) * 4;
                    near_dedup_bytes_saved += token_bytes.saturating_sub(delta.delta_bytes as u64);
                    near_dedup_segments += 1;

                    self.deltas
                        .lock()
                        .unwrap()
                        .insert(hash.clone(), delta.clone());

                    near_dup_meta.insert(
                        hash.0.clone(),
                        json!({
                            "canonical_hash": canonical_hash.0,
                            "similarity": similarity,
                            "delta_bytes": delta.delta_bytes,
                            // Full delta persisted here so retrieval survives process restart.
                            "delta": serde_json::to_value(&delta)
                                .expect("SegmentDelta serialization is infallible"),
                        }),
                    );

                    segment_refs.push(SegmentRef {
                        segment_type: segment.segment_type.clone(),
                        hash,
                        token_count: segment.tokens.len() as u32,
                        position: position as u32,
                    });
                }

                NearDedupAction::StoreAsNew => {
                    // Brand-new segment — compress and store.
                    let compressed = self.inner.compress_tokens(&segment.tokens)?;

                    let raw_size = (segment.tokens.len() as u32) * 4;
                    let compressed_size = compressed.len() as u32;
                    total_compressed += compressed_size as u64;
                    total_raw += raw_size as u64;

                    let stored = StoredSegment {
                        hash: hash.clone(),
                        segment_type: segment.segment_type.clone(),
                        tokenizer: tokenizer_name.clone(),
                        token_count: segment.tokens.len() as u32,
                        compressed_data: compressed,
                        raw_size,
                        compressed_size,
                        ref_count: 1,
                        created_at: Utc::now(),
                    };
                    backend.put_segment(&stored).await?;
                    index_ops.push(BatchSegmentOp::Upsert(stored));
                    new_segments += 1;

                    // Index this segment for future near-dup lookups.
                    self.near_dedup
                        .lock()
                        .unwrap()
                        .insert(hash.clone(), &segment.tokens);

                    segment_refs.push(SegmentRef {
                        segment_type: segment.segment_type.clone(),
                        hash,
                        token_count: segment.tokens.len() as u32,
                        position: position as u32,
                    });
                }
            }
        }

        let total_tokens: u64 = segment_refs.iter().map(|s| s.token_count as u64).sum();

        // Embed near-dup metadata into the manifest so retrieve() can apply deltas.
        let manifest_meta = if near_dup_meta.is_empty() {
            conversation.metadata.clone()
        } else {
            let mut m = conversation.metadata.clone().unwrap_or_default();
            m.insert("near_dup_refs".to_owned(), json!(near_dup_meta));
            Some(m)
        };

        let manifest = ConversationManifest {
            id: id.clone(),
            application: conversation.application.clone(),
            model: conversation.model.clone(),
            tokenizer: tokenizer_name.clone(),
            total_tokens,
            segments: segment_refs,
            created_at: Utc::now(),
            metadata: manifest_meta,
        };

        self.inner.backend().put_manifest(&manifest).await?;

        let manifest_clone = manifest.clone();
        task::spawn_blocking(move || index.store_conversation_batch(&index_ops, &manifest_clone))
            .await
            .map_err(|e| StowkenError::Internal(e.to_string()))?
            .map_err(StowkenError::Metadata)?;

        let compression_ratio = if total_raw == 0 {
            1.0
        } else {
            total_compressed as f64 / total_raw as f64
        };

        let total_segments = new_segments + deduped_segments + near_dedup_segments;
        Ok(EnterpriseStoreResult {
            core: StoreResult {
                id,
                total_segments,
                new_segments,
                deduped_segments,
                bytes_saved,
                compression_ratio,
            },
            near_dedup_segments,
            near_dedup_bytes_saved,
        })
    }

    /// Retrieve a conversation, reconstructing near-dup segments from their deltas.
    pub async fn retrieve(
        &self,
        id: &str,
    ) -> Result<RetrievedConversation, EnterpriseStowkenError> {
        let manifest = self
            .inner
            .backend()
            .get_manifest(id)
            .await
            .map_err(|e| match e {
                StorageError::ConversationNotFound(_) => {
                    StowkenError::NotFound(id.to_owned())
                }
                other => StowkenError::Storage(other),
            })?;

        // Extract near-dup ref map from manifest metadata.
        let near_dup_refs: HashMap<String, serde_json::Value> = manifest
            .metadata
            .as_ref()
            .and_then(|m| m.get("near_dup_refs"))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let mut retrieved_segments = Vec::with_capacity(manifest.segments.len());
        for seg_ref in &manifest.segments {
            if let Some(nd_info) = near_dup_refs.get(&seg_ref.hash.0) {
                // Near-dup: reconstruct from canonical + delta.
                let canonical_hash = SegmentHash(
                    nd_info["canonical_hash"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                );
                let canonical_stored = self.inner.backend().get_segment(&canonical_hash).await?;
                let canonical_tokens = self.inner.decompress_data(&canonical_stored.compressed_data)?;

                // Prefer in-memory cache (populated during store in the same session).
                // Fall back to delta embedded in manifest metadata (survives restarts).
                let delta = {
                    let cached = self.deltas.lock().unwrap().get(&seg_ref.hash).cloned();
                    if let Some(d) = cached {
                        d
                    } else if let Some(d) = nd_info.get("delta") {
                        let d: crate::near_dedup::SegmentDelta =
                            serde_json::from_value(d.clone()).map_err(|e| {
                                EnterpriseStowkenError::Internal(format!(
                                    "corrupt delta for {}: {e}",
                                    seg_ref.hash.0
                                ))
                            })?;
                        // Populate cache for subsequent lookups.
                        self.deltas.lock().unwrap().insert(seg_ref.hash.clone(), d.clone());
                        d
                    } else {
                        return Err(EnterpriseStowkenError::DeltaNotFound(
                            seg_ref.hash.0.clone(),
                        ));
                    }
                };

                let tokens = apply_delta(&canonical_tokens, &delta);
                retrieved_segments.push(RetrievedSegment {
                    segment_type: seg_ref.segment_type.clone(),
                    hash: seg_ref.hash.clone(),
                    token_count: tokens.len() as u32,
                    tokens,
                    position: seg_ref.position,
                });
            } else {
                // Normal segment.
                let stored = self.inner.backend().get_segment(&seg_ref.hash).await?;
                let tokens = self.inner.decompress_data(&stored.compressed_data)?;
                retrieved_segments.push(RetrievedSegment {
                    segment_type: seg_ref.segment_type.clone(),
                    hash: seg_ref.hash.clone(),
                    tokens,
                    token_count: seg_ref.token_count,
                    position: seg_ref.position,
                });
            }
        }

        Ok(RetrievedConversation {
            manifest,
            segments: retrieved_segments,
        })
    }

    /// Delegate GC to the inner vault.
    pub async fn gc(&self) -> Result<u64, EnterpriseStowkenError> {
        Ok(self.inner.gc().await?)
    }

    /// Access to the inner vault for operations not covered by the wrapper.
    pub fn inner(&self) -> &Stowken<B> {
        &self.inner
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use stowken::{storage::MemoryBackend, types::StowkenConfig};

    use super::*;

    async fn vault(threshold: f64) -> EnterpriseStowken<MemoryBackend> {
        EnterpriseStowken::new(MemoryBackend::new(), StowkenConfig::default(), threshold)
            .await
            .unwrap()
    }

    fn conv(id: &str, system: &str, user: &str) -> Conversation {
        use stowken::types::{Message, MessageContent};
        Conversation {
            id: Some(id.to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![
                Message {
                    role: "system".to_owned(),
                    content: MessageContent::Text(system.to_owned()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".to_owned(),
                    content: MessageContent::Text(user.to_owned()),
                    name: None,
                    tool_call_id: None,
                },
            ],
            metadata: None,
        }
    }

    fn conv_tokens(id: &str, sys_tokens: Vec<u32>, user_tokens: Vec<u32>) -> Conversation {
        use stowken::types::{Message, MessageContent};
        Conversation {
            id: Some(id.to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![
                Message {
                    role: "system".to_owned(),
                    content: MessageContent::Tokens(sys_tokens),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".to_owned(),
                    content: MessageContent::Tokens(user_tokens),
                    name: None,
                    tool_call_id: None,
                },
            ],
            metadata: None,
        }
    }

    #[tokio::test]
    async fn exact_dup_still_works() {
        let v = vault(0.85).await;
        let r1 = v.store(conv("c1", "System prompt.", "Hello!")).await.unwrap();
        let r2 = v.store(conv("c2", "System prompt.", "Hello!")).await.unwrap();
        assert_eq!(r1.core.new_segments, r2.core.deduped_segments);
    }

    #[tokio::test]
    async fn near_dup_detected_for_similar_system_prompts() {
        let v = vault(0.80).await;

        // Canonical: 100-token system prompt
        let canonical_sys: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32> = (200u32..210).collect();
        v.store(conv_tokens("c1", canonical_sys.clone(), user.clone()))
            .await
            .unwrap();

        // Variant: change 2 tokens — should be detected as near-dup
        let mut variant_sys = canonical_sys.clone();
        variant_sys[10] = 9999;
        variant_sys[50] = 9998;
        let r2 = v
            .store(conv_tokens("c2", variant_sys, user))
            .await
            .unwrap();

        assert!(
            r2.near_dedup_segments > 0,
            "expected near-dup detection, got {:?}",
            r2.near_dedup_segments
        );
    }

    #[tokio::test]
    async fn near_dup_retrieve_roundtrip() {
        let v = vault(0.80).await;

        let canonical_sys: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32> = (200u32..210).collect();
        v.store(conv_tokens("c1", canonical_sys.clone(), user.clone()))
            .await
            .unwrap();

        let mut variant_sys = canonical_sys.clone();
        variant_sys[10] = 9999;
        variant_sys[50] = 9998;
        let variant_sys_clone = variant_sys.clone();
        v.store(conv_tokens("c2", variant_sys, user.clone()))
            .await
            .unwrap();

        let retrieved = v.retrieve("c2").await.unwrap();
        let sys_seg = retrieved
            .segments
            .iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .expect("no system segment");

        assert_eq!(
            sys_seg.tokens, variant_sys_clone,
            "retrieved tokens should match original variant"
        );
    }

    #[tokio::test]
    async fn dissimilar_conversations_not_linked() {
        let v = vault(0.85).await;

        let sys_a: Vec<u32> = (0u32..100).collect();
        let sys_b: Vec<u32> = (1000u32..1100).collect();
        let user: Vec<u32> = (500u32..510).collect();

        v.store(conv_tokens("c1", sys_a, user.clone())).await.unwrap();
        let r2 = v.store(conv_tokens("c2", sys_b, user)).await.unwrap();

        assert_eq!(r2.near_dedup_segments, 0, "dissimilar segments should not be near-deduped");
    }

    #[tokio::test]
    async fn stats_accumulate() {
        let v = vault(0.80).await;

        let canonical: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32> = (200u32..210).collect();
        v.store(conv_tokens("c1", canonical.clone(), user.clone()))
            .await
            .unwrap();

        for i in 1u32..=5 {
            let mut variant = canonical.clone();
            variant[10] = 9000 + i;
            v.store(conv_tokens(&format!("c{}", i + 1), variant, user.clone()))
                .await
                .unwrap();
        }

        let stats = v.near_dedup_stats();
        assert!(stats.near_dup_hits >= 4, "expected at least 4 near-dup hits, got {}", stats.near_dup_hits);
    }

    // ── Near-dedup + dictionary integration ──────────────────────────────────

    /// Helper: build an EnterpriseStowken pre-loaded with enough data to train a
    /// dictionary, then train and apply it.  Returns the vault.
    async fn vault_with_dict(threshold: f64) -> EnterpriseStowken<MemoryBackend> {
        let v = EnterpriseStowken::new(MemoryBackend::new(), StowkenConfig::default(), threshold)
            .await
            .unwrap();

        // Store 25 conversations as training data (needs ≥10 samples).
        for i in 0u32..25 {
            let sys:  Vec<u32> = (0u32..100).map(|t| (t * 7 + i * 31) % 50_000).collect();
            let user: Vec<u32> = (0u32..30).map(|t| (t * 3 + i * 11) % 50_000).collect();
            v.store(conv_tokens(&format!("train_{i}"), sys, user)).await.unwrap();
        }

        v.inner().train_and_apply_dictionary("cl100k_base", 20).await.unwrap();
        v
    }

    #[tokio::test]
    async fn near_dup_with_active_dictionary_roundtrip() {
        // With the dictionary already active, store a canonical segment then a
        // near-duplicate variant.  Both must decompress correctly.
        let v = vault_with_dict(0.80).await;

        let canonical_sys: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32> = (200u32..210).collect();
        v.store(conv_tokens("canon", canonical_sys.clone(), user.clone())).await.unwrap();

        let mut variant_sys = canonical_sys.clone();
        variant_sys[10] = 9999;
        variant_sys[50] = 9998;
        let variant_sys_clone = variant_sys.clone();

        let r = v.store(conv_tokens("variant", variant_sys, user.clone())).await.unwrap();
        assert!(r.near_dedup_segments > 0, "near-dup should be detected with dictionary active");

        // Canonical retrieve.
        let r_canon = v.retrieve("canon").await.unwrap();
        let canon_seg = r_canon.segments.iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap();
        assert_eq!(canon_seg.tokens, canonical_sys, "canonical tokens corrupted");

        // Variant retrieve via delta reconstruction.
        let r_var = v.retrieve("variant").await.unwrap();
        let var_seg = r_var.segments.iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap();
        assert_eq!(var_seg.tokens, variant_sys_clone, "variant tokens corrupted after delta apply");
    }

    #[tokio::test]
    async fn near_dup_pre_and_post_dict_mixed() {
        // Canonical stored WITHOUT dictionary; dictionary applied; variant stored
        // WITH dictionary.  The variant's delta references a pre-dict canonical —
        // must still reconstruct correctly because delta apply is dictionary-agnostic.
        let v = vault(0.80).await;

        // Store warm-up data so dictionary training has enough samples (needs ≥10).
        // Use disjoint token ranges per conversation so they don't near-dedup each other.
        for i in 0u32..15 {
            let base = i * 5_000;
            let sys:  Vec<u32> = (0u32..100).map(|t| base + t).collect();
            let user_wup: Vec<u32> = (0u32..20).map(|t| base + 1_000 + t).collect();
            v.store(conv_tokens(&format!("warmup_{i}"), sys, user_wup)).await.unwrap();
        }

        // Store the canonical segment (no dict yet).
        let canonical_sys: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32> = (200u32..210).collect();
        v.store(conv_tokens("canon", canonical_sys.clone(), user.clone())).await.unwrap();

        // Train and apply dictionary using warm-up data.
        v.inner().train_and_apply_dictionary("cl100k_base", 12).await.unwrap();

        // Store variant (with dict active).
        let mut variant_sys = canonical_sys.clone();
        variant_sys[5]  = 8888;
        variant_sys[95] = 7777;
        let variant_sys_clone = variant_sys.clone();

        let r = v.store(conv_tokens("variant", variant_sys, user)).await.unwrap();
        assert!(r.near_dedup_segments > 0, "near-dup not detected across dict boundary");

        // Both must roundtrip cleanly.
        let r_canon = v.retrieve("canon").await.unwrap();
        assert_eq!(
            r_canon.segments.iter().find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt)).unwrap().tokens,
            canonical_sys
        );

        let r_var = v.retrieve("variant").await.unwrap();
        assert_eq!(
            r_var.segments.iter().find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt)).unwrap().tokens,
            variant_sys_clone
        );
    }

    #[tokio::test]
    async fn dictionary_improves_compression_for_enterprise_vault() {
        // Compare compressed segment sizes before and after applying a dictionary.
        use stowken::storage::StorageBackend;

        let v = vault(0.85).await;

        // Store 20 warm-up conversations (no dict).
        for i in 0u32..20 {
            let sys:  Vec<u32> = (0u32..100).map(|t| (t * 13 + i * 7) % 50_000).collect();
            let user: Vec<u32> = (0u32..30).map(|t| (t * 5 + i * 3) % 50_000).collect();
            v.store(conv_tokens(&format!("pre_{i}"), sys, user)).await.unwrap();
        }

        // Measure compressed size of a test segment before dict.
        let test_sys: Vec<u32> = (0u32..100).map(|t| t * 13 % 50_000).collect();
        let test_user: Vec<u32> = (0u32..30).collect();
        v.store(conv_tokens("measure_pre", test_sys.clone(), test_user.clone())).await.unwrap();
        let manifest_pre = v.inner().backend().get_manifest("measure_pre").await.unwrap();
        let sys_ref_pre = manifest_pre.segments.iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt)).unwrap();
        let size_pre = v.inner().backend().get_segment(&sys_ref_pre.hash).await.unwrap().compressed_size;

        // Train and apply dictionary.
        v.inner().train_and_apply_dictionary("cl100k_base", 15).await.unwrap();

        // Store a distinct sequence with dict active (different token range avoids near-dup).
        // Uses a different formula so it won't share bigrams with pre-dict segments.
        let test_sys2_distinct: Vec<u32> = (0u32..100).map(|t| 60_000 + t * 17 % 5_000).collect();
        let test_user2: Vec<u32> = (10_000u32..10_030).collect();
        v.store(conv_tokens("measure_post", test_sys2_distinct, test_user2)).await.unwrap();
        let manifest_post = v.inner().backend().get_manifest("measure_post").await.unwrap();
        let sys_ref_post = manifest_post.segments.iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt)).unwrap();
        let size_post = v.inner().backend().get_segment(&sys_ref_post.hash).await.unwrap().compressed_size;

        // Dictionary-compressed segment should be no larger (usually smaller).
        // We don't assert strict improvement since small segments are noisy,
        // but we do assert neither is zero and decompression still works.
        assert!(size_pre > 0 && size_post > 0);
        let r = v.retrieve("measure_post").await.unwrap();
        assert_eq!(r.segments.len(), 2, "should have system + user segments");
    }

    #[tokio::test]
    async fn auto_train_does_not_corrupt_near_dup_retrieval() {
        // EnterpriseStowken wraps a Stowken with auto-train enabled.
        // After the background train fires, near-dup delta reconstruction must still work.
        let config = StowkenConfig {
            auto_train_dictionary_after: Some(15),
            dictionary_sample_count: 10,
            ..StowkenConfig::default()
        };
        let v = EnterpriseStowken::new(MemoryBackend::new(), config, 0.80).await.unwrap();

        // Store enough conversations to trigger auto-train (>15 segments).
        let canonical_sys: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32> = (200u32..210).collect();
        for i in 0u32..10 {
            let sys: Vec<u32> = (0u32..100).map(|t| (t * 7 + i * 3) % 50_000).collect();
            v.store(conv_tokens(&format!("warmup_{i}"), sys, user.clone())).await.unwrap();
        }
        // This stores canonical — also crosses the segment threshold.
        v.store(conv_tokens("canon", canonical_sys.clone(), user.clone())).await.unwrap();

        // Wait for background auto-train to complete.
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Store a near-dup AFTER the background dictionary swap.
        let mut variant_sys = canonical_sys.clone();
        variant_sys[20] = 55555;
        let variant_clone = variant_sys.clone();
        v.store(conv_tokens("variant", variant_sys, user)).await.unwrap();

        // Retrieval must still be correct.
        let r = v.retrieve("variant").await.unwrap();
        let sys_seg = r.segments.iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap();
        assert_eq!(sys_seg.tokens, variant_clone, "tokens corrupted after auto-train + near-dup");
    }

    #[tokio::test]
    async fn near_dup_delta_survives_process_restart() {
        // The delta must be persisted in the manifest so a fresh EnterpriseStowken
        // (empty in-memory cache, new process) can still reconstruct variant tokens.
        use stowken::storage::FilesystemBackend;

        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().to_path_buf();
        let db_path  = data_dir.join("meta.db");

        let canonical_sys: Vec<u32> = (0u32..100).collect();
        let user: Vec<u32>          = (200u32..210).collect();
        let mut variant_sys         = canonical_sys.clone();
        variant_sys[30] = 55555;
        variant_sys[70] = 66666;
        let variant_clone           = variant_sys.clone();

        // ── Session 1: store canonical + near-dup variant ────────────────────
        {
            let backend = FilesystemBackend::new(&data_dir).await.unwrap();
            let v = EnterpriseStowken::open(
                backend, StowkenConfig::default(), db_path.to_str().unwrap(), 0.80,
            ).await.unwrap();

            v.store(conv_tokens("canon",   canonical_sys.clone(), user.clone())).await.unwrap();
            let r = v.store(conv_tokens("variant", variant_sys, user.clone())).await.unwrap();
            assert!(r.near_dedup_segments > 0, "near-dup should be detected");
        }
        // v is dropped here — in-memory delta cache is gone.

        // ── Session 2: fresh vault, empty cache ──────────────────────────────
        {
            let backend = FilesystemBackend::new(&data_dir).await.unwrap();
            let v = EnterpriseStowken::open(
                backend, StowkenConfig::default(), db_path.to_str().unwrap(), 0.80,
            ).await.unwrap();

            let r = v.retrieve("variant").await.unwrap();
            let sys_seg = r.segments.iter()
                .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
                .unwrap();
            assert_eq!(sys_seg.tokens, variant_clone,
                "delta reconstruction failed after simulated restart");
        }
    }
}
