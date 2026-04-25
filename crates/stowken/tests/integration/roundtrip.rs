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
