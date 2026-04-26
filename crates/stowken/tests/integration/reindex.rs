use stowken::{
    storage::FilesystemBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};
use tempfile::TempDir;

fn make_conv(id: &str, sys: Vec<u32>, user: Vec<u32>, asst: Vec<u32>) -> Conversation {
    Conversation {
        id: Some(id.to_owned()),
        application: Some("reindex-test".to_owned()),
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
            Message {
                role: "assistant".to_owned(),
                content: MessageContent::Tokens(asst),
                name: None,
                tool_call_id: None,
            },
        ],
        metadata: None,
    }
}

#[tokio::test]
async fn reindex_after_db_deletion_restores_all_stats() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("metadata.db");
    let shared_sys = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

    // ── Session 1: ingest 5 conversations sharing a system prompt. ───────
    let stats_before;
    let segment_stats_before;
    {
        let backend = FilesystemBackend::new(dir.path()).await.unwrap();
        let vault = Stowken::open(backend, StowkenConfig::default(), db_path.to_str().unwrap())
            .await
            .unwrap();

        for i in 0..5u32 {
            vault
                .store(make_conv(
                    &format!("conv-{i}"),
                    shared_sys.clone(),
                    vec![100 + i, 101 + i],
                    vec![200 + i, 201 + i],
                ))
                .await
                .unwrap();
        }

        stats_before = vault.stats().await.unwrap();
        segment_stats_before = vault.segment_stats().await.unwrap();
    }

    // Sanity: 5 conversations, 1 shared system prompt, 5 unique users + 5 unique assistants = 11 unique.
    assert_eq!(stats_before.total_conversations, 5);
    assert_eq!(stats_before.unique_segments, 11);
    assert_eq!(stats_before.total_segments, 15);

    // ── Disaster: nuke metadata.db. Manifests + segment files survive. ──
    std::fs::remove_file(&db_path).unwrap();
    assert!(!db_path.exists());

    // ── Session 2: reopen (creates an empty DB), confirm stats are wrecked. ──
    {
        let backend = FilesystemBackend::new(dir.path()).await.unwrap();
        let vault = Stowken::open(backend, StowkenConfig::default(), db_path.to_str().unwrap())
            .await
            .unwrap();

        let empty_stats = vault.stats().await.unwrap();
        assert_eq!(
            empty_stats.total_conversations, 0,
            "fresh empty DB should report 0 conversations before reindex"
        );

        // ── Reindex from manifests. ─────────────────────────────────────
        let report = vault.reindex().await.unwrap();
        assert_eq!(report.conversations_indexed, 5);
        assert_eq!(report.unique_segments_indexed, 11);
        assert_eq!(report.segments_missing, 0);

        // ── Stats should now match the pre-deletion state exactly. ──────
        let stats_after = vault.stats().await.unwrap();
        assert_eq!(stats_after.total_conversations, stats_before.total_conversations);
        assert_eq!(stats_after.unique_segments, stats_before.unique_segments);
        assert_eq!(stats_after.total_segments, stats_before.total_segments);
        assert_eq!(stats_after.total_tokens, stats_before.total_tokens);
        assert_eq!(stats_after.naive_bytes, stats_before.naive_bytes);
        assert_eq!(stats_after.storage_bytes, stats_before.storage_bytes);

        // Per-segment-type breakdowns should match too. Sort first since
        // SQL row order isn't guaranteed.
        let mut after = vault.segment_stats().await.unwrap();
        let mut before = segment_stats_before.clone();
        after.sort_by(|a, b| a.segment_type.to_string().cmp(&b.segment_type.to_string()));
        before.sort_by(|a, b| a.segment_type.to_string().cmp(&b.segment_type.to_string()));
        for (a, b) in after.iter().zip(before.iter()) {
            assert_eq!(a.segment_type, b.segment_type);
            assert_eq!(a.unique_count, b.unique_count);
            assert_eq!(a.total_references, b.total_references);
            assert_eq!(a.total_tokens, b.total_tokens);
        }

        // Differentiator commands should also work post-reindex.
        let prompts = vault.list_system_prompts().await.unwrap();
        assert_eq!(prompts.len(), 1, "shared system prompt should be findable after reindex");
        assert_eq!(prompts[0].ref_count, 5, "system prompt should still report 5 refs");

        let dups = vault.find_duplicates(2, 100).await.unwrap();
        assert_eq!(dups.len(), 1, "find-duplicates should still surface the shared prompt");
        assert_eq!(dups[0].ref_count, 5);
    }
}

#[tokio::test]
async fn reindex_is_idempotent() {
    // Running reindex twice in a row should produce the same state, not
    // double-count anything.
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("metadata.db");

    let backend = FilesystemBackend::new(dir.path()).await.unwrap();
    let vault = Stowken::open(backend, StowkenConfig::default(), db_path.to_str().unwrap())
        .await
        .unwrap();

    let shared_sys = vec![1u32, 2, 3];
    for i in 0..3u32 {
        vault
            .store(make_conv(
                &format!("idem-{i}"),
                shared_sys.clone(),
                vec![10 + i],
                vec![20 + i],
            ))
            .await
            .unwrap();
    }

    let baseline = vault.stats().await.unwrap();
    vault.reindex().await.unwrap();
    let after_one = vault.stats().await.unwrap();
    vault.reindex().await.unwrap();
    let after_two = vault.stats().await.unwrap();

    assert_eq!(baseline.unique_segments, after_one.unique_segments);
    assert_eq!(baseline.total_segments, after_one.total_segments);
    assert_eq!(after_one.unique_segments, after_two.unique_segments);
    assert_eq!(after_one.total_segments, after_two.total_segments);
}
