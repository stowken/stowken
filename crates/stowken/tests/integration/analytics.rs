use stowken::{
    storage::MemoryBackend,
    types::{
        AnalyticsQuery, Conversation, ExportConfig, ExportFormat, Message, MessageContent,
        StowkenConfig,
    },
    Stowken,
};

fn make_conv(id: &str, model: &str, app: &str, user: Vec<u32>, asst: Vec<u32>) -> Conversation {
    Conversation {
        id: Some(id.to_owned()),
        application: Some(app.to_owned()),
        model: model.to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: vec![
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

async fn populated_vault() -> Stowken<MemoryBackend> {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    vault
        .store(make_conv("a1", "gpt-4", "app-a", vec![1, 2], vec![3, 4]))
        .await
        .unwrap();
    vault
        .store(make_conv("a2", "gpt-4", "app-a", vec![5, 6], vec![7, 8]))
        .await
        .unwrap();
    vault
        .store(make_conv(
            "b1",
            "gpt-3.5",
            "app-b",
            vec![9, 10],
            vec![11, 12],
        ))
        .await
        .unwrap();
    vault
}

#[tokio::test]
async fn stats_count_conversations() {
    let vault = populated_vault().await;
    let stats = vault.stats().await.unwrap();
    assert_eq!(stats.total_conversations, 3);
}

#[tokio::test]
async fn stats_count_unique_segments() {
    let vault = populated_vault().await;
    let stats = vault.stats().await.unwrap();
    // 3 conversations × 2 segments each, all unique = 6 unique segments
    assert_eq!(stats.unique_segments, 6);
    assert_eq!(stats.total_segments, 6);
}

#[tokio::test]
async fn segment_stats_by_type() {
    let vault = populated_vault().await;
    let type_stats = vault.segment_stats().await.unwrap();
    // Should have UserTurn and AssistantTurn entries
    let has_user = type_stats
        .iter()
        .any(|s| s.segment_type == stowken::types::SegmentType::UserTurn);
    let has_asst = type_stats
        .iter()
        .any(|s| s.segment_type == stowken::types::SegmentType::AssistantTurn);
    assert!(has_user);
    assert!(has_asst);
}

#[tokio::test]
async fn query_analytics_by_model() {
    let vault = populated_vault().await;
    let rows = vault
        .query(AnalyticsQuery {
            group_by: Some(vec!["model".to_owned()]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!rows.is_empty(), "expected at least one row");
    for row in &rows {
        assert!(row.contains_key("model") || row.contains_key("total_tokens"));
    }
}

#[tokio::test]
async fn export_training_data_jsonl() {
    let vault = populated_vault().await;
    let mut output = Vec::new();
    let stats = vault
        .export_training_data(
            ExportConfig {
                format: ExportFormat::Jsonl,
                include_system_prompts: false,
                include_context: false,
                deduplicate_pairs: true,
                tokenizer: None,
                model: None,
                application: None,
                max_conversations: None,
            },
            &mut output,
        )
        .await
        .unwrap();

    assert!(stats.total_pairs > 0, "should export at least one pair");
    let text = String::from_utf8(output).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), stats.unique_pairs as usize);

    // Each line must be valid JSON with a "messages" key
    for line in &lines {
        let parsed: serde_json::Value = serde_json::from_str(line).expect("invalid JSON line");
        assert!(
            parsed["messages"].is_array(),
            "each line must have a 'messages' array"
        );
    }
}

#[tokio::test]
async fn export_deduplicates_pairs() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    // Two conversations with identical user/assistant pairs
    let user_tokens = vec![1u32, 2, 3];
    let asst_tokens = vec![4u32, 5, 6];
    for id in &["dup-1", "dup-2"] {
        let conv = Conversation {
            id: Some(id.to_string()),
            application: None,
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k".to_owned(),
            messages: vec![
                Message {
                    role: "user".to_owned(),
                    content: MessageContent::Tokens(user_tokens.clone()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "assistant".to_owned(),
                    content: MessageContent::Tokens(asst_tokens.clone()),
                    name: None,
                    tool_call_id: None,
                },
            ],
            metadata: None,
        };
        vault.store(conv).await.unwrap();
    }

    let mut output = Vec::new();
    let stats = vault
        .export_training_data(
            ExportConfig {
                format: ExportFormat::Jsonl,
                include_system_prompts: false,
                include_context: false,
                deduplicate_pairs: true,
                tokenizer: None,
                model: None,
                application: None,
                max_conversations: None,
            },
            &mut output,
        )
        .await
        .unwrap();

    assert_eq!(stats.total_pairs, 2);
    assert_eq!(
        stats.unique_pairs, 1,
        "dedup should yield 1 unique pair from 2 identical conversations"
    );
}
