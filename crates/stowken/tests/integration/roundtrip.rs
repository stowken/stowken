use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

fn make_conversation(id: &str, tokens: Vec<Vec<u32>>) -> Conversation {
    let roles = ["system", "user", "assistant"];
    Conversation {
        id: Some(id.to_owned()),
        application: Some("test-app".to_owned()),
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: tokens
            .into_iter()
            .enumerate()
            .map(|(i, t)| Message {
                role: roles[i % 3].to_owned(),
                content: MessageContent::Tokens(t),
                name: None,
                tool_call_id: None,
            })
            .collect(),
        metadata: None,
    }
}

#[tokio::test]
async fn roundtrip_tokens_match_exactly() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let expected_tokens = vec![
        vec![1u32, 2, 3, 4, 5],
        vec![100u32, 200, 300, 16_384, u32::MAX],
        vec![42u32; 200],
    ];

    let conv = make_conversation("rt-1", expected_tokens.clone());
    let result = vault.store(conv).await.unwrap();

    let retrieved = vault.retrieve(&result.id).await.unwrap();
    assert_eq!(retrieved.segments.len(), expected_tokens.len());

    for (seg, expected) in retrieved.segments.iter().zip(expected_tokens.iter()) {
        assert_eq!(&seg.tokens, expected, "tokens must be perfectly lossless");
    }
}

#[tokio::test]
async fn retrieve_segment_by_hash() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let tokens = vec![10u32, 20, 30, 40, 50];
    let conv = make_conversation("rt-2", vec![tokens.clone()]);
    let result = vault.store(conv).await.unwrap();

    let retrieved = vault.retrieve(&result.id).await.unwrap();
    let hash = &retrieved.segments[0].hash;

    let seg_by_hash = vault.retrieve_segment_by_hash(hash).await.unwrap();
    assert_eq!(seg_by_hash.tokens, tokens);
}

#[tokio::test]
async fn retrieve_segments_filtered_by_type() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conv = make_conversation(
        "rt-3",
        vec![vec![1u32, 2], vec![3u32, 4], vec![5u32, 6]],
    );
    let result = vault.store(conv).await.unwrap();

    let user_segs = vault
        .retrieve_segments(&result.id, Some(stowken::types::SegmentType::UserTurn))
        .await
        .unwrap();
    assert_eq!(user_segs.len(), 1, "expected exactly one user turn");
    assert_eq!(user_segs[0].tokens, vec![3u32, 4]);
}

#[tokio::test]
async fn delete_removes_conversation() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conv = make_conversation("rt-4", vec![vec![1u32, 2, 3]]);
    let result = vault.store(conv).await.unwrap();

    vault.delete(&result.id).await.unwrap();

    let retrieve_result = vault.retrieve(&result.id).await;
    assert!(retrieve_result.is_err(), "deleted conversation should not be retrievable");
}

#[tokio::test]
async fn large_token_sequence_roundtrip() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    // 10K tokens spanning all varint encoding ranges
    let tokens: Vec<u32> = (0..10_000).map(|i| {
        match i % 4 {
            0 => i % 128,
            1 => 128 + (i % (16_383 - 128)),
            2 => 16_384 + (i % (2_097_151 - 16_384)),
            _ => 2_097_152 + (i % 100_000),
        }
    }).collect();

    let conv = make_conversation("rt-5", vec![tokens.clone()]);
    let result = vault.store(conv).await.unwrap();

    let retrieved = vault.retrieve(&result.id).await.unwrap();
    assert_eq!(retrieved.segments[0].tokens, tokens);
}

#[tokio::test]
async fn roundtrip_with_filesystem_backend() {
    use tempfile::TempDir;
    use stowken::storage::FilesystemBackend;

    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("meta.db");
    let backend = FilesystemBackend::new(dir.path()).await.unwrap();
    let vault = Stowken::open(backend, StowkenConfig::default(), db_path.to_str().unwrap())
        .await
        .unwrap();

    let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let conv = make_conversation("fs-rt-1", vec![tokens.clone()]);
    let result = vault.store(conv).await.unwrap();

    let retrieved = vault.retrieve(&result.id).await.unwrap();
    assert_eq!(retrieved.segments[0].tokens, tokens);
}

#[tokio::test]
async fn retrieve_batch_returns_all_stored_conversations() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let mut ids: Vec<String> = Vec::new();
    for i in 0..4u32 {
        let conv = make_conversation(&format!("batch-{i}"), vec![vec![i + 1, 10, 20]]);
        ids.push(vault.store(conv).await.unwrap().id);
    }

    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let results = vault.retrieve_batch(&id_refs).await.unwrap();

    assert_eq!(results.len(), 4, "should return one entry per stored id");
    for r in &results {
        assert!(ids.contains(&r.conversation_id), "returned id should be one we stored");
        assert!(!r.text.is_empty(), "text should be non-empty");
        assert!(!r.turns.is_empty(), "turns should be non-empty");
    }
}

#[tokio::test]
async fn retrieve_batch_skips_missing_ids_silently() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conv = make_conversation("present", vec![vec![1u32, 2, 3]]);
    vault.store(conv).await.unwrap();

    let results = vault
        .retrieve_batch(&["present", "does-not-exist"])
        .await
        .unwrap();

    assert_eq!(results.len(), 1, "missing id should be silently skipped");
    assert_eq!(results[0].conversation_id, "present");
}

#[tokio::test]
async fn retrieve_batch_text_contains_role_markers() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    // Store a minimal conversation with known structure
    let conv = make_conversation("markers", vec![
        vec![1u32, 2],   // system
        vec![3u32, 4],   // user
        vec![5u32, 6],   // assistant
    ]);
    vault.store(conv).await.unwrap();

    let results = vault.retrieve_batch(&["markers"]).await.unwrap();
    assert_eq!(results.len(), 1);

    let text = &results[0].text;
    assert!(text.contains("[system]"),    "text should contain [system] marker");
    assert!(text.contains("[user]"),      "text should contain [user] marker");
    assert!(text.contains("[assistant]"), "text should contain [assistant] marker");

    let turns = &results[0].turns;
    assert_eq!(turns.len(), 3, "should have 3 turns");
    assert_eq!(turns[0].role, "system");
    assert_eq!(turns[1].role, "user");
    assert_eq!(turns[2].role, "assistant");
}
