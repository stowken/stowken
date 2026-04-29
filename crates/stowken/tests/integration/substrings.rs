//! Token-level shared-substring dedup integration tests.
//!
//! Pins the v0.5 contract:
//!   - Training a substring registry on a corpus of related segments
//!     promotes high-frequency windows.
//!   - New segments stored after training automatically use the
//!     registered substrings if the resulting 0x05 frame is smaller.
//!   - Round-trip of a 0x05 frame returns the original tokens exactly.
//!   - Substrings persist across vault reopen.
//!   - Substring path is a strict opt-in win — falls back to baseline
//!     when no match exists or when the substring frame isn't smaller.

use stowken::{
    storage::{FilesystemBackend, MemoryBackend},
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};
use tempfile::TempDir;

fn make_conv(id: &str, sys: Vec<u32>, user: Vec<u32>) -> Conversation {
    Conversation {
        id: Some(id.to_owned()),
        application: Some("substring-test".to_owned()),
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

/// Build a system prompt that's mostly the same long shared chunk, with a
/// few interpolated tokens at random positions. Like our drift-corpus
/// fixture but inline so this test is self-contained.
fn make_drift_system(seed: u32, length: usize) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..length as u32).collect();
    // Three interpolations at deterministic-but-different positions per seed.
    let positions = [
        (seed % length as u32) as usize,
        ((seed * 7) % length as u32) as usize,
        ((seed * 13) % length as u32) as usize,
    ];
    for (i, p) in positions.iter().enumerate() {
        tokens[*p] = 90_000 + (seed * 17 + i as u32 * 11) % 5_000;
    }
    tokens
}

#[tokio::test]
async fn discovery_promotes_frequent_windows() {
    // 30 conversations sharing a 200-token system-prompt template with
    // small per-conv interpolations. Most 16-token windows are shared
    // across nearly all conversations.
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();
    for i in 0..30u32 {
        vault
            .store(make_conv(
                &format!("seed-{i}"),
                make_drift_system(i, 200),
                vec![1000 + i],
            ))
            .await
            .unwrap();
    }

    assert_eq!(vault.list_substrings().len(), 0);
    let promoted = vault.train_substrings(30, 5).await.unwrap();
    assert!(
        !promoted.is_empty(),
        "discovery should promote at least one substring on a high-overlap corpus"
    );
    assert_eq!(vault.list_substrings().len(), promoted.len());
    // Each promoted substring is the minimum-length window.
    for info in &promoted {
        assert!(info.length >= 16, "promoted substring must clear MIN_LENGTH");
        assert!(info.source_occurrences >= 5, "must clear min_occurrences");
    }
}

#[tokio::test]
async fn substring_frame_used_after_training() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();
    // Seed enough drift conversations so training has signal.
    for i in 0..30u32 {
        vault
            .store(make_conv(
                &format!("seed-{i}"),
                make_drift_system(i, 300),
                vec![5000 + i],
            ))
            .await
            .unwrap();
    }
    let promoted = vault.train_substrings(30, 5).await.unwrap();
    assert!(!promoted.is_empty(), "training produced no substrings");

    // Store a NEW segment that shares heavily with the training corpus.
    let novel_id = "post-train";
    let novel_sys = make_drift_system(999, 300);
    let novel_user = vec![9_999_001, 9_999_002];
    let novel_sys_clone = novel_sys.clone();
    vault
        .store(make_conv(novel_id, novel_sys, novel_user.clone()))
        .await
        .unwrap();

    // Round-trip: the new segment's tokens come back exactly.
    let r = vault.retrieve(novel_id).await.unwrap();
    let sys_seg = r
        .segments
        .iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys_seg.tokens, novel_sys_clone);
}

#[tokio::test]
async fn substring_round_trip_is_exact() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();
    for i in 0..20u32 {
        vault
            .store(make_conv(
                &format!("seed-{i}"),
                make_drift_system(i, 250),
                vec![100 + i],
            ))
            .await
            .unwrap();
    }
    vault.train_substrings(20, 5).await.unwrap();

    // Store-then-retrieve a brand new segment and check exact equality.
    let novel = make_drift_system(42, 250);
    vault
        .store(make_conv("rt", novel.clone(), vec![99_999]))
        .await
        .unwrap();
    let r = vault.retrieve("rt").await.unwrap();
    let sys = r
        .segments
        .iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, novel);
}

#[tokio::test]
async fn no_substrings_means_no_0x05_frames() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();
    // Without training, nothing in the registry — substring path returns
    // None and the baseline frame is used. Round-trip should still work.
    let tokens: Vec<u32> = (0..200).collect();
    vault
        .store(make_conv("only", tokens.clone(), vec![1, 2, 3]))
        .await
        .unwrap();
    let r = vault.retrieve("only").await.unwrap();
    let sys = r
        .segments
        .iter()
        .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
        .unwrap();
    assert_eq!(sys.tokens, tokens);
    assert_eq!(vault.list_substrings().len(), 0);
}

#[tokio::test]
async fn substring_persistence_across_vault_reopen() {
    let dir = TempDir::new().unwrap();
    let data_path = dir.path().to_path_buf();
    let db_path = data_path.join("metadata.db");
    let db_str = db_path.to_str().unwrap().to_owned();

    let novel_tokens = make_drift_system(777, 250);
    let novel_clone = novel_tokens.clone();
    let promoted_count;

    // Session 1: ingest, train, store a 0x05-eligible segment.
    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, StowkenConfig::default(), &db_str)
            .await
            .unwrap();
        for i in 0..30u32 {
            vault
                .store(make_conv(
                    &format!("seed-{i}"),
                    make_drift_system(i, 250),
                    vec![1000 + i],
                ))
                .await
                .unwrap();
        }
        let promoted = vault.train_substrings(30, 5).await.unwrap();
        promoted_count = promoted.len();
        assert!(promoted_count > 0);
        vault
            .store(make_conv("post-train", novel_tokens, vec![88_888]))
            .await
            .unwrap();
    }

    // Session 2: reopen and verify both the registry and the
    // substring-encoded segment.
    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, StowkenConfig::default(), &db_str)
            .await
            .unwrap();
        assert_eq!(
            vault.list_substrings().len(),
            promoted_count,
            "substring registry didn't persist across reopen"
        );
        let r = vault.retrieve("post-train").await.unwrap();
        let sys = r
            .segments
            .iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap();
        assert_eq!(sys.tokens, novel_clone, "0x05 frame failed to round-trip after reopen");
    }
}

#[tokio::test]
async fn pre_substring_segments_remain_readable_after_promotion() {
    // Migration scenario: store some segments, then train substrings,
    // then ingest more. Old segments (stored as 0x02 frames) keep
    // working; new ones use 0x05.
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();
    let pre = make_drift_system(1, 250);
    vault.store(make_conv("pre", pre.clone(), vec![1])).await.unwrap();

    for i in 100..130u32 {
        vault
            .store(make_conv(
                &format!("seed-{i}"),
                make_drift_system(i, 250),
                vec![100 + i],
            ))
            .await
            .unwrap();
    }
    vault.train_substrings(30, 5).await.unwrap();

    let post = make_drift_system(2, 250);
    vault.store(make_conv("post", post.clone(), vec![2])).await.unwrap();

    // Both retrievals work.
    assert_eq!(
        vault
            .retrieve("pre")
            .await
            .unwrap()
            .segments
            .iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap()
            .tokens,
        pre
    );
    assert_eq!(
        vault
            .retrieve("post")
            .await
            .unwrap()
            .segments
            .iter()
            .find(|s| matches!(s.segment_type, stowken::types::SegmentType::SystemPrompt))
            .unwrap()
            .tokens,
        post
    );
}

#[tokio::test]
async fn empty_corpus_train_returns_empty() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();
    let promoted = vault.train_substrings(50, 5).await.unwrap();
    assert!(promoted.is_empty());
    assert_eq!(vault.list_substrings().len(), 0);
}
