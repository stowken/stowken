//! Basic usage: store and retrieve a conversation.

use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = StowkenConfig::default();
    let backend = MemoryBackend::new();
    let vault = Stowken::new(backend, config).await?;

    let system_prompt = "You are a helpful assistant.";

    // Store several conversations that share the same system prompt.
    for i in 0..5 {
        let conv = Conversation {
            id: None,
            model: "gpt-4".into(),
            tokenizer: "cl100k_base".into(),
            application: Some("example-app".into()),
            metadata: None,
            messages: vec![
                Message {
                    role: "system".into(),
                    content: MessageContent::Text(system_prompt.into()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".into(),
                    content: MessageContent::Text(format!("Hello, this is conversation {}!", i)),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "assistant".into(),
                    content: MessageContent::Text(format!(
                        "Hello! I'm conversation {} response.",
                        i
                    )),
                    name: None,
                    tool_call_id: None,
                },
            ],
        };

        let result = vault.store(conv).await?;
        println!(
            "conv-{i}: {} total segs ({} new, {} deduped), {:.1}x compression",
            result.total_segments,
            result.new_segments,
            result.deduped_segments,
            result.compression_ratio,
        );
    }

    // Storage statistics
    let stats = vault.stats().await?;
    println!("\nStorage stats:");
    println!("  Total tokens:   {}", stats.total_tokens);
    println!("  Unique segments: {}", stats.unique_segments);
    println!("  Dedup ratio:    {:.1}%", stats.dedup_ratio * 100.0);
    println!("  Compression:    {:.1}x", stats.compression_ratio);

    Ok(())
}
