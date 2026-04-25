use std::time::Instant;
use stowken::{
    get_tokenizer,
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;

    let sys   = "You are a helpful, harmless, and honest AI assistant. Answer questions clearly.";
    let user  = "What is the difference between async/await and threads in Rust?";
    let asst  = "Async/await is cooperative multitasking suited for I/O-bound work. Threads are preemptive and run in parallel on multiple cores, better for CPU-bound tasks.";

    // ── 1. tokenizer init cost ────────────────────────────────────────────────
    let t = Instant::now();
    for _ in 0..500 {
        let _ = get_tokenizer("cl100k_base").unwrap();
    }
    println!("500× get_tokenizer()        : {:>6}ms  ({:.2}ms each)", t.elapsed().as_millis(), t.elapsed().as_millis() as f64 / 500.0);

    // ── 2. tokenization cost ─────────────────────────────────────────────────
    let adapter = get_tokenizer("cl100k_base").unwrap();
    let t = Instant::now();
    for _ in 0..500 {
        let _ = adapter.tokenize(sys);
        let _ = adapter.tokenize(user);
        let _ = adapter.tokenize(asst);
    }
    println!("500× tokenize(3 messages)   : {:>6}ms  ({:.2}ms each)", t.elapsed().as_millis(), t.elapsed().as_millis() as f64 / 500.0);

    // ── 3. full store() — first conv (all new segments) ──────────────────────
    let make_conv = || Conversation {
        id: None,
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        application: None,
        messages: vec![
            Message { role: "system".to_owned(),    content: MessageContent::Text(sys.to_owned()),  name: None, tool_call_id: None },
            Message { role: "user".to_owned(),      content: MessageContent::Text(user.to_owned()), name: None, tool_call_id: None },
            Message { role: "assistant".to_owned(), content: MessageContent::Text(asst.to_owned()), name: None, tool_call_id: None },
        ],
        metadata: None,
    };

    vault.store(make_conv()).await?; // warm up
    let t = Instant::now();
    for _ in 0..100 {
        vault.store(make_conv()).await?; // all deduped after warm-up
    }
    println!("100× store() fully deduped  : {:>6}ms  ({:.2}ms each)", t.elapsed().as_millis(), t.elapsed().as_millis() as f64 / 100.0);

    // ── 4. store() — unique conversations (new segments each time) ────────────
    let vault2 = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;
    let t = Instant::now();
    for i in 0u32..100 {
        let conv = Conversation {
            id: None,
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![
                Message { role: "system".to_owned(),    content: MessageContent::Text(sys.to_owned()),  name: None, tool_call_id: None },
                Message { role: "user".to_owned(),      content: MessageContent::Text(format!("Question number {i}: {user}")), name: None, tool_call_id: None },
                Message { role: "assistant".to_owned(), content: MessageContent::Text(format!("Answer {i}: {asst}")), name: None, tool_call_id: None },
            ],
            metadata: None,
        };
        vault2.store(conv).await?;
    }
    println!("100× store() unique segments: {:>6}ms  ({:.2}ms each)", t.elapsed().as_millis(), t.elapsed().as_millis() as f64 / 100.0);

    Ok(())
}
