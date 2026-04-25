use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

fn shared_system_prompt_conversations(count: usize) -> Vec<Conversation> {
    let system_tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    (0..count)
        .map(|i| Conversation {
            id: Some(format!("dedup-{i}")),
            application: None,
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            messages: vec![
                Message {
                    role: "system".to_owned(),
                    content: MessageContent::Tokens(system_tokens.clone()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".to_owned(),
                    content: MessageContent::Tokens(vec![100u32 + i as u32, 200]),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "assistant".to_owned(),
                    content: MessageContent::Tokens(vec![300u32 + i as u32, 400]),
                    name: None,
                    tool_call_id: None,
                },
            ],
            metadata: None,
        })
        .collect()
}

#[tokio::test]
async fn exact_dedup_system_prompt() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conversations = shared_system_prompt_conversations(10);
    let mut first_result = None;
    for conv in conversations {
        let r = vault.store(conv).await.unwrap();
        if first_result.is_none() {
            first_result = Some(r);
        }
    }

    let stats = vault.stats().await.unwrap();
    // 10 conversations, each with 3 segments = 30 total refs
    // System prompt is shared → only 1 unique, but user/assistant are different → 20 unique + 1 shared = 21
    assert!(stats.total_segments >= 30, "expected at least 30 total segment refs");
    assert!(stats.unique_segments < stats.total_segments, "dedup should reduce unique count");
    assert!(stats.dedup_ratio > 0.0, "dedup ratio should be positive");
}

#[tokio::test]
async fn same_conversation_twice_increments_refs() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conv = Conversation {
        id: None,
        application: None,
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: MessageContent::Tokens(vec![1u32, 2, 3]),
            name: None,
            tool_call_id: None,
        }],
        metadata: None,
    };

    let _r1 = vault.store(conv.clone()).await.unwrap();
    let r2 = vault.store(conv).await.unwrap();

    // Second store should report 1 deduped segment, not a new one
    assert_eq!(r2.new_segments, 0, "segment already exists; should not create a new one");
    assert_eq!(r2.deduped_segments, 1);

    let stats = vault.stats().await.unwrap();
    assert_eq!(stats.unique_segments, 1);
    assert_eq!(stats.total_segments, 2);
}

#[tokio::test]
async fn gc_removes_orphaned_segments_after_delete() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conv = Conversation {
        id: Some("gc-test".to_owned()),
        application: None,
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: MessageContent::Tokens(vec![10u32, 20, 30]),
            name: None,
            tool_call_id: None,
        }],
        metadata: None,
    };

    vault.store(conv).await.unwrap();
    vault.delete("gc-test").await.unwrap();

    let deleted = vault.gc().await.unwrap();
    assert_eq!(deleted, 1, "GC should remove 1 orphaned segment");
}

#[tokio::test]
async fn system_prompt_listing() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conversations = shared_system_prompt_conversations(5);
    for conv in conversations {
        vault.store(conv).await.unwrap();
    }

    let prompts = vault.list_system_prompts().await.unwrap();
    assert_eq!(prompts.len(), 1, "there should be exactly one unique system prompt");
    assert_eq!(prompts[0].ref_count, 5, "system prompt should be referenced 5 times");
}

#[tokio::test]
async fn find_conversations_by_segment_hash() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    let conversations = shared_system_prompt_conversations(3);
    for conv in conversations {
        vault.store(conv).await.unwrap();
    }

    let prompts = vault.list_system_prompts().await.unwrap();
    let hash = &prompts[0].hash;

    let conv_ids = vault.find_by_segment(hash).await.unwrap();
    assert_eq!(conv_ids.len(), 3, "all 3 conversations should reference the shared system prompt");
}
