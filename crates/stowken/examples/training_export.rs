//! Training export: export stored conversations as JSONL for fine-tuning.

use stowken::{
    export::training::export_jsonl,
    storage::FilesystemBackend,
    types::{Conversation, ExportConfig, ExportFormat, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = std::env::temp_dir().join("stowken-training-example");
    let backend = FilesystemBackend::new(&tmp).await?;
    let vault = Stowken::new(backend, StowkenConfig::default()).await?;

    let system = "You are a helpful, accurate assistant.";
    let training_pairs = [
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("Explain gravity.", "Gravity is the force that attracts objects with mass toward each other."),
        ("Write a haiku.", "Old silent pond\nA frog jumps into the pond\nSplash! Silence again."),
    ];

    for (question, answer) in &training_pairs {
        let conv = Conversation {
            id: None,
            model: "gpt-4".into(),
            tokenizer: "cl100k_base".into(),
            application: Some("training".into()),
            metadata: None,
            messages: vec![
                Message {
                    role: "system".into(),
                    content: MessageContent::Text(system.into()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".into(),
                    content: MessageContent::Text(question.to_string()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "assistant".into(),
                    content: MessageContent::Text(answer.to_string()),
                    name: None,
                    tool_call_id: None,
                },
            ],
        };
        vault.store(conv).await?;
    }

    let config = ExportConfig {
        format: ExportFormat::Jsonl,
        include_system_prompts: true,
        include_context: false,
        deduplicate_pairs: true,
        tokenizer: None,
        model: None,
        application: Some("training".into()),
        max_conversations: None,
    };

    let output_path = tmp.join("training_data.jsonl");
    let mut file = std::fs::File::create(&output_path)?;
    let stats = export_jsonl(&vault, &config, &mut file).await?;

    println!("Export complete:");
    println!("  Total pairs:     {}", stats.total_pairs);
    println!("  Unique pairs:    {}", stats.unique_pairs);
    println!("  Tokens exported: {}", stats.tokens_exported);
    println!("  Output: {}", output_path.display());

    Ok(())
}
