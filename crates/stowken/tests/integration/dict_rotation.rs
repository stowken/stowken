//! Dictionary rotation safety tests.
//!
//! In v0.1, applying a new dictionary replaced the old one in memory. Frames
//! compressed with the previous dictionary became unreadable on the next
//! call to `retrieve()` — silent data loss. v0.3 fixes this at the format
//! layer: each frame carries its own dict_id, multiple dicts coexist in the
//! registry, and rotation is non-destructive.
//!
//! These tests pin that contract at the vault API level. If any of them
//! ever fail, the v0.1 bug has come back.

use std::path::PathBuf;

use stowken::{
    storage::{FilesystemBackend, MemoryBackend},
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};
use tempfile::TempDir;

fn make_conv(id: &str, sys: Vec<u32>, user: Vec<u32>) -> Conversation {
    Conversation {
        id: Some(id.to_owned()),
        application: Some("dict-rotation-test".to_owned()),
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

/// Seed a vault with enough segments to train a dictionary.
async fn seed<B: stowken::storage::backend::StorageBackend + 'static>(
    vault: &Stowken<B>,
    n: u32,
    sys_seed: u32,
) {
    for i in 0..n {
        let sys: Vec<u32> = (0u32..120)
            .map(|t| (t * 13 + i * 7 + sys_seed) % 50_000)
            .collect();
        let user: Vec<u32> = (0u32..40).map(|t| (t * 5 + i * 11) % 50_000).collect();
        vault
            .store(make_conv(&format!("seed-{sys_seed}-{i}"), sys, user))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn segments_compressed_with_old_dict_remain_readable_after_rotation() {
    // Reproduces the v0.1 data-loss scenario at the vault level:
    // 1. Train dict A, activate it
    // 2. Store conversation X (will be compressed with A)
    // 3. Train dict B, activate it
    // 4. Store conversation Y (will be compressed with B)
    // 5. Retrieve X — must still come back intact
    // 6. Retrieve Y — must also come back intact
    //
    // v0.1 lost X's data here.

    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    seed(&vault, 30, 1).await;

    // Train and activate dict A.
    let info_a = vault.train_and_activate_dictionary(20).await.unwrap();
    assert!(info_a.is_active);

    // Store conversation X under dict A.
    let x_sys: Vec<u32> = (0u32..120).collect();
    let x_user: Vec<u32> = (1000u32..1040).collect();
    vault
        .store(make_conv("X-pre-rotation", x_sys.clone(), x_user.clone()))
        .await
        .unwrap();

    // Seed more data so dict B trains on something different.
    seed(&vault, 30, 2).await;

    // Train and activate dict B.
    let info_b = vault.train_and_activate_dictionary(20).await.unwrap();
    assert_ne!(info_a.id, info_b.id, "rotation: dicts must differ");

    // Store conversation Y under dict B.
    let y_sys: Vec<u32> = (5000u32..5120).collect();
    let y_user: Vec<u32> = (6000u32..6040).collect();
    vault
        .store(make_conv("Y-post-rotation", y_sys.clone(), y_user.clone()))
        .await
        .unwrap();

    // Retrieve both and confirm exact roundtrip.
    let x = vault.retrieve("X-pre-rotation").await.unwrap();
    let y = vault.retrieve("Y-post-rotation").await.unwrap();

    assert_eq!(
        x.segments[0].tokens, x_sys,
        "X's system tokens corrupted — old dict frame lost!"
    );
    assert_eq!(x.segments[1].tokens, x_user, "X's user tokens corrupted");
    assert_eq!(y.segments[0].tokens, y_sys, "Y's system tokens corrupted");
    assert_eq!(y.segments[1].tokens, y_user, "Y's user tokens corrupted");
}

#[tokio::test]
async fn three_dict_rotation_all_frames_readable() {
    // More aggressive: rotate three times, store under each, retrieve all.
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    seed(&vault, 30, 1).await;
    let _a = vault.train_and_activate_dictionary(20).await.unwrap();

    let conv_a_sys: Vec<u32> = (0u32..100).collect();
    vault
        .store(make_conv("conv-a", conv_a_sys.clone(), vec![1, 2, 3]))
        .await
        .unwrap();

    seed(&vault, 30, 7).await;
    let _b = vault.train_and_activate_dictionary(20).await.unwrap();

    let conv_b_sys: Vec<u32> = (200u32..300).collect();
    vault
        .store(make_conv("conv-b", conv_b_sys.clone(), vec![4, 5, 6]))
        .await
        .unwrap();

    seed(&vault, 30, 19).await;
    let _c = vault.train_and_activate_dictionary(20).await.unwrap();

    let conv_c_sys: Vec<u32> = (400u32..500).collect();
    vault
        .store(make_conv("conv-c", conv_c_sys.clone(), vec![7, 8, 9]))
        .await
        .unwrap();

    // Three dicts now in the registry.
    assert_eq!(vault.list_dictionaries().len(), 3);

    // All three pre-rotation conversations must round-trip.
    assert_eq!(
        vault.retrieve("conv-a").await.unwrap().segments[0].tokens,
        conv_a_sys
    );
    assert_eq!(
        vault.retrieve("conv-b").await.unwrap().segments[0].tokens,
        conv_b_sys
    );
    assert_eq!(
        vault.retrieve("conv-c").await.unwrap().segments[0].tokens,
        conv_c_sys
    );
}

#[tokio::test]
async fn dict_persistence_across_vault_reopen() {
    // Train + activate a dict, drop the vault, reopen — the same dict must
    // be active and old frames must still decompress.
    let dir = TempDir::new().unwrap();
    let data_path: PathBuf = dir.path().to_path_buf();
    let db_path = data_path.join("metadata.db");
    let db_str = db_path.to_str().unwrap().to_owned();

    let dict_id;
    let pre_rotate_tokens: Vec<u32> = (0u32..150).collect();

    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, StowkenConfig::default(), &db_str)
            .await
            .unwrap();

        seed(&vault, 30, 1).await;

        let info = vault.train_and_activate_dictionary(20).await.unwrap();
        dict_id = info.id;

        vault
            .store(make_conv(
                "persist-test",
                pre_rotate_tokens.clone(),
                vec![100, 101, 102],
            ))
            .await
            .unwrap();
    }

    // Reopen.
    {
        let backend = FilesystemBackend::new(&data_path).await.unwrap();
        let vault = Stowken::open(backend, StowkenConfig::default(), &db_str)
            .await
            .unwrap();

        // Registry should have the same dict, still active.
        let dicts = vault.list_dictionaries();
        assert_eq!(dicts.len(), 1, "dict not persisted across reopen");
        assert_eq!(dicts[0].id, dict_id);
        assert!(
            dicts[0].is_active,
            "active marker not persisted across reopen"
        );

        // The dict-compressed frame must decompress correctly.
        let retrieved = vault.retrieve("persist-test").await.unwrap();
        assert_eq!(
            retrieved.segments[0].tokens, pre_rotate_tokens,
            "dict-compressed frame corrupted across reopen"
        );
    }
}

#[tokio::test]
async fn pre_dict_frames_remain_readable_after_dict_introduced() {
    // Migration scenario: vault accumulates 0x02 (no-dict) frames, then a
    // dictionary is introduced. Old frames must keep working forever.
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let pre_dict_tokens: Vec<u32> = (0u32..200).collect();
    vault
        .store(make_conv(
            "pre-dict",
            pre_dict_tokens.clone(),
            vec![1, 2, 3],
        ))
        .await
        .unwrap();

    seed(&vault, 30, 1).await;
    vault.train_and_activate_dictionary(20).await.unwrap();

    let post_dict_tokens: Vec<u32> = (1000u32..1200).collect();
    vault
        .store(make_conv(
            "post-dict",
            post_dict_tokens.clone(),
            vec![4, 5, 6],
        ))
        .await
        .unwrap();

    let pre = vault.retrieve("pre-dict").await.unwrap();
    let post = vault.retrieve("post-dict").await.unwrap();

    assert_eq!(
        pre.segments[0].tokens, pre_dict_tokens,
        "pre-dict (0x02) frame corrupted"
    );
    assert_eq!(
        post.segments[0].tokens, post_dict_tokens,
        "post-dict (0x03) frame corrupted"
    );
}

#[tokio::test]
async fn training_with_too_few_samples_errors_clearly() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    // Only 3 conversations stored, each 1 segment — far below MIN_TRAINING_SAMPLES.
    for i in 0..3u32 {
        vault
            .store(make_conv(&format!("c{i}"), vec![i, i + 1], vec![i + 2]))
            .await
            .unwrap();
    }

    let err = vault.train_dictionary(50).await.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("not enough segments") && msg.contains("required"),
        "expected clear too-few-samples message, got: {msg}"
    );
}
