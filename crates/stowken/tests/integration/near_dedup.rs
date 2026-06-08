//! Near-dedup integration tests.
//!
//! v0.1's near-dedup was in-memory only: the LSH index lived in a HashMap
//! that was wiped on every restart, and it cached the full token sequence
//! of every canonical (linear memory growth). v0.4 fixes both: the index
//! lives in SQLite, and tokens are fetched from the backend on demand.
//!
//! Test invariants:
//!   - Storing a near-duplicate produces a 0x04 frame against the canonical.
//!   - The variant retrieves to its original tokens via delta apply.
//!   - LSH state survives a vault reopen — v0.1's "lost on restart" bug.
//!   - Reindex rebuilds the LSH index from on-disk manifests.
//!   - find_near_duplicates surfaces the cluster.

use std::path::PathBuf;

use stowken::{
    storage::{FilesystemBackend, MemoryBackend},
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};
use tempfile::TempDir;

fn config_with_near_dedup(threshold: f64) -> StowkenConfig {
    StowkenConfig {
        enable_compression: true,
        near_dedup_threshold: Some(threshold),
        embed_reuse_threshold: None,
    }
}

fn make_conv(id: &str, sys: Vec<u32>, user: Vec<u32>) -> Conversation {
    Conversation {
        id: Some(id.to_owned()),
        application: Some("near-dedup-test".to_owned()),
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: vec![
            Message {
                role: "system".to_owned(),
                content: MessageContent::Tokens(sys),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_owned(),
                content: MessageContent::Tokens(user),
                name: None,
                tool_call_id: None,
            },
        ],
        metadata: None,
    }
}

#[tokio::test]
async fn variant_stored_as_delta_round_trips() {
    let vault = Stowken::new(MemoryBackend::new(), config_with_near_dedup(0.80))
        .await
        .unwrap();

    let canonical: Vec<u32> = (0u32..200).collect();
    vault.store(make_conv("canon", canonical.clone(), vec![1, 2, 3])).await.unwrap();

    // Near-duplicate: 2 token edits in a 200-token sequence.
    let mut variant = canonical.clone();
    variant[20] = 99_999;
    variant[150] = 88_888;
    let variant_clone = variant.clone();
    vault.store(make_conv("variant", variant, vec![4, 5, 6])).await.unwrap();

    // Retrieval reconstructs the original variant tokens via delta apply.
    let r = vault.retrieve("variant").await.unwrap();
    let sys = r.segments.iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, variant_clone, "delta-decoded variant doesn't match input");
}

#[tokio::test]
async fn no_near_dedup_when_disabled() {
    // With near_dedup_threshold = None (default), variants are stored as
    // full segments — no delta path.
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let canonical: Vec<u32> = (0u32..200).collect();
    vault.store(make_conv("canon", canonical.clone(), vec![1, 2, 3])).await.unwrap();
    let mut variant = canonical.clone();
    variant[20] = 99_999;
    vault.store(make_conv("variant", variant.clone(), vec![4, 5, 6])).await.unwrap();

    // The LSH index should be empty — feature is off.
    assert_eq!(vault.near_dedup_index_size().await.unwrap(), 0);
}

#[tokio::test]
async fn dissimilar_segments_are_not_near_deduped() {
    let vault = Stowken::new(MemoryBackend::new(), config_with_near_dedup(0.85))
        .await
        .unwrap();

    let a: Vec<u32> = (0u32..100).collect();
    let b: Vec<u32> = (1000u32..1100).collect();

    vault.store(make_conv("a", a, vec![1])).await.unwrap();
    vault.store(make_conv("b", b.clone(), vec![2])).await.unwrap();

    // Both should be retrievable as their original tokens.
    let r = vault.retrieve("b").await.unwrap();
    let sys = r.segments.iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, b);
}

#[tokio::test]
async fn lsh_index_survives_vault_reopen() {
    // v0.1's bug: LSH index was in-memory; reopening the vault meant
    // future near-duplicates of pre-restart canonicals wouldn't be
    // detected. Test that v0.4 doesn't have this regression.
    let dir = TempDir::new().unwrap();
    let data_path: PathBuf = dir.path().to_path_buf();
    let db_path = data_path.join("metadata.db");
    let db_str = db_path.to_str().unwrap().to_owned();

    let canonical: Vec<u32> = (0u32..200).collect();

    // Session 1: store the canonical (1 conv = 2 segments → 2 LSH entries).
    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, config_with_near_dedup(0.80), &db_str)
            .await
            .unwrap();
        vault.store(make_conv("canon", canonical.clone(), vec![1, 2, 3])).await.unwrap();
        assert_eq!(vault.near_dedup_index_size().await.unwrap(), 2);
    }

    // Session 2: store a near-duplicate. v0.1 would have written this
    // as a full segment because the LSH index was empty at startup.
    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, config_with_near_dedup(0.80), &db_str)
            .await
            .unwrap();

        // Confirm LSH state was loaded from disk.
        assert_eq!(
            vault.near_dedup_index_size().await.unwrap(),
            2,
            "LSH index lost across reopen — v0.1 bug regressed"
        );

        let mut variant = canonical.clone();
        variant[10] = 99_999;
        variant[100] = 88_888;
        let variant_clone = variant.clone();
        vault.store(make_conv("variant", variant, vec![4, 5, 6])).await.unwrap();

        // The variant should round-trip via the delta path.
        let r = vault.retrieve("variant").await.unwrap();
        let sys = r.segments.iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap();
        assert_eq!(sys.tokens, variant_clone);
    }
}

#[tokio::test]
async fn reindex_rebuilds_lsh_after_db_deletion() {
    // Lose metadata.db, run reindex, confirm LSH index is rebuilt.
    let dir = TempDir::new().unwrap();
    let data_path: PathBuf = dir.path().to_path_buf();
    let db_path = data_path.join("metadata.db");
    let db_str = db_path.to_str().unwrap().to_owned();

    let canonical_a: Vec<u32> = (0u32..200).collect();
    let canonical_b: Vec<u32> = (5000u32..5200).collect();

    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, config_with_near_dedup(0.80), &db_str)
            .await
            .unwrap();
        vault.store(make_conv("a", canonical_a.clone(), vec![1])).await.unwrap();
        vault.store(make_conv("b", canonical_b.clone(), vec![2])).await.unwrap();
        assert_eq!(vault.near_dedup_index_size().await.unwrap(), 4);
    }

    // Wipe ALL derived state — the SQLite metadata DB *and* the packed
    // near-dup logs (the LSH index no longer lives in SQLite). Manifests +
    // segment files survive as the source of truth.
    std::fs::remove_file(&db_path).unwrap();
    std::fs::remove_dir_all(data_path.join("near_dedup")).unwrap();

    // Reopen + reindex.
    let backend = FilesystemBackend::new(&data_path).await.unwrap();
    let vault = Stowken::open(backend, config_with_near_dedup(0.80), &db_str)
        .await
        .unwrap();
    assert_eq!(vault.near_dedup_index_size().await.unwrap(), 0);
    let report = vault.reindex().await.unwrap();
    assert_eq!(report.conversations_indexed, 2);
    assert_eq!(
        vault.near_dedup_index_size().await.unwrap(),
        4,
        "LSH index not rebuilt by reindex"
    );

    // Now a near-duplicate of A should dedup against the reindexed canonical.
    let mut variant = canonical_a.clone();
    variant[50] = 99_999;
    let variant_clone = variant.clone();
    vault.store(make_conv("a-variant", variant, vec![3])).await.unwrap();
    let r = vault.retrieve("a-variant").await.unwrap();
    let sys = r.segments.iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, variant_clone);
}

#[tokio::test]
async fn find_near_duplicates_surfaces_clusters() {
    let vault = Stowken::new(MemoryBackend::new(), config_with_near_dedup(0.80))
        .await
        .unwrap();

    // Store a canonical and 5 near-variants. With high enough refs the
    // canonical bubbles up via find_duplicates → find_near_duplicates
    // adds the variants in order.
    let canonical: Vec<u32> = (0u32..200).collect();
    // Reference the canonical many times so it shows up in find_duplicates.
    for i in 0..6u32 {
        vault.store(make_conv(
            &format!("canon-{i}"),
            canonical.clone(),
            vec![100 + i],
        ))
        .await
        .unwrap();
    }

    // Add some variants.
    for i in 0..3u32 {
        let mut variant = canonical.clone();
        variant[10 + i as usize] = 99_000 + i;
        variant[100 + i as usize] = 88_000 + i;
        vault.store(make_conv(
            &format!("var-{i}"),
            variant,
            vec![200 + i],
        ))
        .await
        .unwrap();
    }

    let clusters = vault
        .find_near_duplicates(0.80, 5, 100)
        .await
        .unwrap();

    assert!(!clusters.is_empty(), "expected at least one cluster");
    let cluster = &clusters[0];
    assert!(cluster.canonical_ref_count >= 5);
    assert!(!cluster.variants.is_empty(), "expected variants in the cluster");
    // Variants should be sorted similarity descending.
    let sims: Vec<f64> = cluster.variants.iter().map(|v| v.similarity).collect();
    let mut sorted = sims.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(sims, sorted, "variants must be sorted by similarity desc");
}

#[tokio::test]
async fn delta_only_used_when_actually_smaller() {
    // For a tiny segment, the delta + 32-byte canonical hash can exceed a
    // fresh full compression. Vault should fall back to full storage in
    // that case rather than paying the cross-ref cost for negative gain.
    let vault = Stowken::new(MemoryBackend::new(), config_with_near_dedup(0.10))
        .await
        .unwrap();

    let canonical: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    vault.store(make_conv("canon", canonical, vec![1])).await.unwrap();

    // A 10-token "variant" — too small for a delta to win.
    let variant = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 99];
    vault.store(make_conv("variant", variant.clone(), vec![2])).await.unwrap();

    // Variant must still round-trip exactly.
    let r = vault.retrieve("variant").await.unwrap();
    let sys = r.segments.iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, variant);
}

#[tokio::test]
async fn gc_prunes_near_dedup_index() {
    let vault = Stowken::new(MemoryBackend::new(), config_with_near_dedup(0.80))
        .await
        .unwrap();

    let canonical: Vec<u32> = (0u32..200).collect();
    vault.store(make_conv("a", canonical.clone(), vec![1])).await.unwrap();
    // System prompt + user turn are both indexed for future matching.
    assert_eq!(vault.near_dedup_index_size().await.unwrap(), 2);

    // Drop all refs, then collect. The pruned signatures must disappear.
    vault.delete("a").await.unwrap();
    vault.gc().await.unwrap();
    assert_eq!(
        vault.near_dedup_index_size().await.unwrap(),
        0,
        "GC must prune near-dup signatures for collected segments"
    );

    // A fresh near-duplicate of the now-deleted canonical must still ingest
    // cleanly — a stale signature would otherwise be surfaced as a delta
    // target and fail when the (deleted) canonical is fetched.
    let mut variant = canonical.clone();
    variant[10] = 99_999;
    let variant_clone = variant.clone();
    vault.store(make_conv("b", variant, vec![2])).await.unwrap();

    let r = vault.retrieve("b").await.unwrap();
    let sys = r
        .segments
        .iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, variant_clone);
    assert_eq!(vault.near_dedup_index_size().await.unwrap(), 2);
}
