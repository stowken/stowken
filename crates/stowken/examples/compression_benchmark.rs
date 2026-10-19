//! Comprehensive compression benchmark.
//!
//! Generates a realistic multi-pattern corpus and reports deduplication,
//! compression, and combined savings vs naive u32 token storage.
//!
//! Run with:
//!   cargo run --example compression_benchmark --release

use std::time::Instant;

use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

// ── Corpus parameters ─────────────────────────────────────────────────────────

const TOTAL_CONVERSATIONS: usize = 500;

// ── Realistic text fixtures ───────────────────────────────────────────────────

const SYSTEM_PROMPTS: &[&str] = &[
    // ~40 tokens each — shared across many conversations to exercise dedup
    "You are a helpful, harmless, and honest AI assistant. \
     Answer questions clearly and concisely. If you are unsure about something, \
     say so rather than making things up. Always be respectful and professional.",
    "You are an expert software engineer with deep knowledge of Rust, Python, \
     TypeScript, and systems programming. Help users debug code, review pull \
     requests, and design robust software architectures. Prefer idiomatic \
     solutions and explain trade-offs when relevant.",
    "You are a customer support agent for Acme Corp. Be empathetic, patient, \
     and solution-oriented. Follow the company escalation policy: resolve tier-1 \
     issues yourself; escalate billing disputes to the billing team; escalate \
     technical outages to engineering. Always end with a satisfaction check.",
    "You are a medical information assistant. Provide general health information \
     only — never diagnose or prescribe. Always recommend consulting a licensed \
     physician for personal medical concerns. Cite reputable sources when possible.",
    "You are a creative writing assistant specializing in science fiction and \
     fantasy. Help users develop worlds, characters, and plots. Ask clarifying \
     questions to understand the tone and audience. Offer multiple options when \
     stuck. Avoid clichés unless subverted intentionally.",
];

// RAG context injected into user turns (simulates retrieval-augmented generation)
const RAG_CONTEXTS: &[&str] = &[
    "Retrieved document: The Rust programming language achieves memory safety \
     without a garbage collector by using an ownership system with rules that \
     the compiler checks at compile time. Every value in Rust has a variable \
     called its owner, and there can only be one owner at a time.",
    "Retrieved document: Zstandard (zstd) is a fast lossless compression \
     algorithm targeting real-time compression scenarios at zlib-level and \
     better compression ratios. It offers a wide range of speed/compression \
     trade-offs backed by a very fast decoder.",
    "Retrieved document: The Transformer architecture was introduced in the \
     paper 'Attention Is All You Need' (Vaswani et al., 2017). It relies \
     entirely on attention mechanisms, dispensing with recurrence and \
     convolutions entirely, and has become the dominant architecture for \
     large language models.",
    "Retrieved document: Python's Global Interpreter Lock (GIL) is a mutex \
     that protects access to Python objects, preventing multiple native threads \
     from executing Python bytecode simultaneously. This simplifies CPython \
     implementation but limits multi-core parallelism for CPU-bound workloads.",
];

const USER_QUESTIONS: &[&str] = &[
    "Can you explain how ownership works in Rust?",
    "What is the difference between async/await and threads?",
    "How should I structure a large Python project?",
    "What are the best practices for REST API design?",
    "Can you review this code and suggest improvements?",
    "Why is my Docker container running out of memory?",
    "How do I set up CI/CD for a Rust project?",
    "What is the CAP theorem and why does it matter?",
    "Explain the difference between SQL and NoSQL databases.",
    "How does garbage collection work in Go?",
    "What is tail recursion and why is it useful?",
    "How do I handle errors idiomatically in Rust?",
    "What's the best way to cache database queries?",
    "Can you write a unit test for this function?",
    "How do I profile a slow Python script?",
    "What are the trade-offs between microservices and monoliths?",
    "How does TLS handshake work?",
    "What is CORS and how do I configure it correctly?",
    "Explain consistent hashing in simple terms.",
    "How do I implement rate limiting in my API?",
];

const ASSISTANT_RESPONSES: &[&str] = &[
    "Great question! Rust's ownership system is built around three rules: \
     each value has exactly one owner, there can only be one owner at a time, \
     and when the owner goes out of scope the value is dropped. This eliminates \
     entire classes of bugs like use-after-free and double-free at compile time.",
    "Async/await is cooperative multitasking: tasks yield control voluntarily \
     when waiting for I/O, letting a single thread handle many concurrent \
     operations efficiently. Threads are preemptive and run in parallel on \
     multiple cores, making them better for CPU-bound work. For I/O-heavy \
     workloads, async is typically more efficient.",
    "For large Python projects, use a src layout, configure pyproject.toml, \
     and organise code into domain-driven packages. Keep business logic separate \
     from I/O. Use type hints throughout and run mypy in CI. Consider poetry or \
     hatch for dependency management.",
    "REST API best practices: use nouns for resources, not verbs; use HTTP \
     methods semantically (GET/POST/PUT/PATCH/DELETE); version your API from day \
     one; return consistent error shapes; paginate collections; use HTTPS; \
     document with OpenAPI.",
    "I've reviewed the code. Main issues: the nested loop is O(n²) — consider \
     a hash set for O(n) membership tests; error handling swallows context — use \
     anyhow or thiserror; the function does too much — split into smaller units \
     with clear names. Happy to show rewritten versions of any section.",
    "Out-of-memory in Docker usually means the container hit its memory limit. \
     Check with `docker stats` and inspect the OOM killer logs via \
     `dmesg | grep oom`. Common causes: uncapped caches, memory leaks, or a \
     limit set too low for the workload. Increase the limit or investigate the \
     leak with a memory profiler.",
    "For Rust CI/CD I recommend GitHub Actions with the `dtolnay/rust-toolchain` \
     action. Run `cargo test`, `cargo clippy -- -D warnings`, and \
     `cargo fmt -- --check` on every PR. Cache `~/.cargo/registry` and the \
     `target` directory to keep build times fast. Add a release profile job \
     that uploads artifacts on tag pushes.",
    "The CAP theorem states that a distributed system can provide at most two \
     of: Consistency (every read sees the latest write), Availability (every \
     request gets a response), and Partition tolerance (the system keeps working \
     despite network splits). Since partitions are unavoidable in practice, you \
     choose between CP (consistent but may reject requests) and AP (available \
     but may return stale data).",
    "SQL databases enforce a schema and ACID transactions, making them ideal for \
     structured data with complex relationships. NoSQL databases sacrifice some \
     consistency guarantees for horizontal scalability and flexible schemas. \
     Choose SQL for financial records, NoSQL for high-throughput event streams \
     or document stores.",
    "Go uses a tri-colour mark-and-sweep garbage collector running concurrently \
     with the program. It aims for sub-millisecond pause times. The GC scans \
     the heap, marks live objects starting from roots (globals, stack), then \
     sweeps unreachable memory. You can tune it with GOGC — lower values collect \
     more aggressively at the cost of CPU.",
];

// ── Corpus generation ─────────────────────────────────────────────────────────

fn make_conversation(index: usize, multi_turn_depth: usize) -> Conversation {
    let system_idx = index % SYSTEM_PROMPTS.len();
    let mut messages = vec![Message {
        role: "system".to_owned(),
        content: MessageContent::Text(SYSTEM_PROMPTS[system_idx].to_owned()),
        name: None,
        tool_call_id: None,
    }];

    // Optionally inject a RAG context block into the first user message
    let use_rag = index.is_multiple_of(4);
    let rag_idx = index % RAG_CONTEXTS.len();

    for turn in 0..=multi_turn_depth {
        let q_idx = (index * 3 + turn * 7) % USER_QUESTIONS.len();
        let a_idx = (index * 5 + turn * 3) % ASSISTANT_RESPONSES.len();

        let user_content = if turn == 0 && use_rag {
            format!(
                "{}\n\nUsing the above context: {}",
                RAG_CONTEXTS[rag_idx], USER_QUESTIONS[q_idx]
            )
        } else {
            USER_QUESTIONS[q_idx].to_owned()
        };

        messages.push(Message {
            role: "user".to_owned(),
            content: MessageContent::Text(user_content),
            name: None,
            tool_call_id: None,
        });
        messages.push(Message {
            role: "assistant".to_owned(),
            content: MessageContent::Text(ASSISTANT_RESPONSES[a_idx].to_owned()),
            name: None,
            tool_call_id: None,
        });
    }

    Conversation {
        id: None,
        application: Some(["support", "coding", "research"][index % 3].to_owned()),
        model: ["gpt-4", "gpt-4o", "gpt-3.5-turbo"][index % 3].to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages,
        metadata: None,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;

    println!("Stowken Compression Benchmark");
    println!("{}", "=".repeat(60));
    println!("Corpus: {TOTAL_CONVERSATIONS} conversations, cl100k_base tokenizer");
    println!("Pattern mix:");
    println!(
        "  {} system prompt templates (high reuse)",
        SYSTEM_PROMPTS.len()
    );
    println!(
        "  {} RAG context blocks (25% of conversations)",
        RAG_CONTEXTS.len()
    );
    println!("  {} user question templates", USER_QUESTIONS.len());
    println!(
        "  {} assistant response templates",
        ASSISTANT_RESPONSES.len()
    );
    println!("  Multi-turn depth: 1–3 turns per conversation");
    println!();

    // ── Ingest ────────────────────────────────────────────────────────────────

    let t0 = Instant::now();
    let mut total_new = 0u64;
    let mut total_deduped = 0u64;
    let mut total_bytes_saved = 0u64;

    for i in 0..TOTAL_CONVERSATIONS {
        let depth = match i % 3 {
            0 => 0, // single-turn  (33%)
            1 => 1, // two-turn     (33%)
            _ => 2, // three-turn   (34%)
        };
        let conv = make_conversation(i, depth);
        let r = vault.store(conv).await?;
        total_new += r.new_segments;
        total_deduped += r.deduped_segments;
        total_bytes_saved += r.bytes_saved;
    }

    let ingest_ms = t0.elapsed().as_millis();

    // ── Stats ─────────────────────────────────────────────────────────────────

    let stats = vault.stats().await?;
    let seg_stats = vault.segment_stats().await?;

    // Naive cost: every token reference stored as a raw u32 (4 bytes)
    // This is what you'd pay without any dedup or compression.
    let naive_bytes = stats.naive_bytes;
    let actual_bytes = stats.storage_bytes;
    let total_saved = naive_bytes.saturating_sub(actual_bytes);
    let combined_savings_pct = if naive_bytes > 0 {
        100.0 * total_saved as f64 / naive_bytes as f64
    } else {
        0.0
    };

    // Dedup-only cost: deduplicated but uncompressed raw tokens
    let raw_unique_bytes = (stats.unique_segments as f64
        * (stats.total_tokens as f64 / stats.total_segments.max(1) as f64)
        * 4.0) as u64;
    let dedup_savings_pct = if naive_bytes > 0 {
        100.0 * naive_bytes.saturating_sub(raw_unique_bytes) as f64 / naive_bytes as f64
    } else {
        0.0
    };
    let compression_savings_pct = if raw_unique_bytes > 0 {
        100.0 * raw_unique_bytes.saturating_sub(actual_bytes) as f64 / raw_unique_bytes as f64
    } else {
        0.0
    };

    // ── Report ────────────────────────────────────────────────────────────────

    println!("Ingest Results");
    println!("{}", "-".repeat(60));
    println!(
        "  Time          : {}ms ({:.1}ms/conv)",
        ingest_ms,
        ingest_ms as f64 / TOTAL_CONVERSATIONS as f64
    );
    println!("  Conversations : {}", stats.total_conversations);
    println!("  Total tokens  : {}", fmt_num(stats.total_tokens));
    println!();

    println!("Segment Deduplication");
    println!("{}", "-".repeat(60));
    println!("  Total refs    : {}", fmt_num(total_new + total_deduped));
    println!(
        "  Unique stored : {}  ({:.1}%)",
        fmt_num(stats.unique_segments),
        100.0 * stats.unique_segments as f64 / (total_new + total_deduped).max(1) as f64
    );
    println!(
        "  Deduped refs  : {}  ({:.1}%)",
        fmt_num(total_deduped),
        100.0 * total_deduped as f64 / (total_new + total_deduped).max(1) as f64
    );
    println!();

    println!("Compression");
    println!("{}", "-".repeat(60));
    println!(
        "  Naive storage : {}  (all refs, raw u32)",
        fmt_bytes(naive_bytes)
    );
    println!(
        "  After dedup   : {}  ({:.1}% saved by dedup)",
        fmt_bytes(raw_unique_bytes),
        dedup_savings_pct
    );
    println!(
        "  After zstd    : {}  ({:.1}% saved by zstd)",
        fmt_bytes(actual_bytes),
        compression_savings_pct
    );
    println!();
    println!("  ┌─────────────────────────────────────────────┐");
    println!(
        "  │  Combined savings: {:>6.1}%  ({} → {})  │",
        combined_savings_pct,
        fmt_bytes(naive_bytes),
        fmt_bytes(actual_bytes)
    );
    println!("  └─────────────────────────────────────────────┘");
    println!();

    println!("Per-Segment-Type Breakdown");
    println!("{}", "-".repeat(60));
    println!(
        "  {:<16}  {:>8}  {:>8}  {:>8}  {:>9}",
        "Type", "Unique", "Refs", "Dedup%", "Tokens"
    );
    println!("  {}", "-".repeat(56));
    for s in &seg_stats {
        println!(
            "  {:<16}  {:>8}  {:>8}  {:>7.1}%  {:>9}",
            s.segment_type.to_string(),
            fmt_num(s.unique_count),
            fmt_num(s.total_references),
            s.dedup_ratio * 100.0,
            fmt_num(s.total_tokens)
        );
    }
    println!();

    println!("Bytes Saved by Deduplication (segment refs avoided)");
    println!("{}", "-".repeat(60));
    println!("  {}", fmt_bytes(total_bytes_saved));
    println!();

    println!("Compression Ratio");
    println!("{}", "-".repeat(60));
    println!(
        "  zstd ratio    : {:.3}x  (compressed/raw)",
        stats.compression_ratio
    );
    println!("  Space savings : {:.1}%", stats.savings_percentage);

    Ok(())
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1_048_576 {
        format!("{:.2} MB", b as f64 / 1_048_576.0)
    } else if b >= 1_024 {
        format!("{:.1} KB", b as f64 / 1_024.0)
    } else {
        format!("{} B", b)
    }
}

fn fmt_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}
