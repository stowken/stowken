//! Analytics: query token usage by model, application, and date.

use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;

    // Store conversations from two different apps and models
    let configs = [
        ("gpt-4", "chat-app"),
        ("gpt-4", "chat-app"),
        ("gpt-3.5-turbo", "batch-processor"),
        ("gpt-3.5-turbo", "batch-processor"),
        ("claude-3", "internal-tool"),
    ];

    for (model, app) in &configs {
        let conv = Conversation {
            id: None,
            model: model.to_string(),
            tokenizer: "cl100k_base".into(),
            application: Some(app.to_string()),
            metadata: None,
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Text(format!("A question for {model} via {app}")),
                name: None,
                tool_call_id: None,
            }],
        };
        vault.store(conv).await?;
    }

    // Query all conversations
    let all_stats = vault.stats().await?;
    println!("=== Global stats ===");
    println!("Conversations: {}", all_stats.total_conversations);
    println!("Total tokens:  {}", all_stats.total_tokens);
    println!();

    // Segment type breakdown
    let type_stats = vault.segment_stats().await?;
    println!("=== Segment type breakdown ===");
    for stat in &type_stats {
        println!(
            "  {:?}: {} unique segs, {} total refs, {} tokens",
            stat.segment_type, stat.unique_count, stat.total_references, stat.total_tokens
        );
    }

    Ok(())
}
