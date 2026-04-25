//! Decompression correctness verification.
//!
//! Stores conversations with known content, retrieves them, and asserts
//! byte-for-byte identity between the original and recovered token sequences.
//!
//! Run with:
//!   cargo run --example decompression_verify --release

use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Stowken Decompression Verification");
    println!("{}", "=".repeat(60));

    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;
    let mut passed = 0usize;
    let mut failed = 0usize;

    // ── Test 1: pre-tokenized tokens round-trip ───────────────────────────────
    {
        let tokens_sys: Vec<u32> = (0u32..200).map(|i| i * 17 % 50_000).collect();
        let tokens_user: Vec<u32> = (0u32..500).map(|i| i * 31 % 50_000).collect();
        let tokens_asst: Vec<u32> = (0u32..300).map(|i| i * 7 % 50_000 + 1).collect();

        let conv = Conversation {
            id: Some("test-pretoken".to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![
                msg("system",    MessageContent::Tokens(tokens_sys.clone())),
                msg("user",      MessageContent::Tokens(tokens_user.clone())),
                msg("assistant", MessageContent::Tokens(tokens_asst.clone())),
            ],
            metadata: None,
        };

        let store_result = vault.store(conv).await?;
        let retrieved = vault.retrieve(&store_result.id).await?;

        let sys_ok  = retrieved.segments[0].tokens == tokens_sys;
        let user_ok = retrieved.segments[1].tokens == tokens_user;
        let asst_ok = retrieved.segments[2].tokens == tokens_asst;
        let all_ok  = sys_ok && user_ok && asst_ok;

        print_result("Pre-tokenized tokens (200/500/300 tokens)", all_ok,
            &format!("system={} user={} assistant={}", ok(sys_ok), ok(user_ok), ok(asst_ok)));
        if all_ok { passed += 1; } else { failed += 1; }
    }

    // ── Test 2: text content tokenized then recovered via tiktoken ────────────
    {
        let system_text = "You are a helpful assistant specialised in Rust programming.";
        let user_text   = "What is the difference between Box<T> and Arc<T>?";
        let asst_text   = "Box<T> provides exclusive ownership of heap-allocated data with \
                           no runtime overhead. Arc<T> (Atomically Reference Counted) allows \
                           shared ownership across multiple owners using atomic reference counting, \
                           making it safe to share across threads. Use Box when you need single \
                           ownership; use Arc when you need to share data across threads or owners.";

        let conv = Conversation {
            id: Some("test-text".to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![
                msg("system",    MessageContent::Text(system_text.to_owned())),
                msg("user",      MessageContent::Text(user_text.to_owned())),
                msg("assistant", MessageContent::Text(asst_text.to_owned())),
            ],
            metadata: None,
        };

        let store_result = vault.store(conv).await?;
        let retrieved = vault.retrieve(&store_result.id).await?;

        // Re-tokenize with tiktoken to get the expected token sequences
        let tokenizer = stowken::get_tokenizer("cl100k_base").expect("cl100k_base");
        let expected_sys  = tokenizer.tokenize(system_text);
        let expected_user = tokenizer.tokenize(user_text);
        let expected_asst = tokenizer.tokenize(asst_text);

        let sys_ok  = retrieved.segments[0].tokens == expected_sys;
        let user_ok = retrieved.segments[1].tokens == expected_user;
        let asst_ok = retrieved.segments[2].tokens == expected_asst;
        let all_ok  = sys_ok && user_ok && asst_ok;

        print_result("Text → tokenize → store → retrieve → tokens match", all_ok,
            &format!("system={} user={} assistant={}", ok(sys_ok), ok(user_ok), ok(asst_ok)));

        // Also verify detokenization recovers the original text
        let recovered_sys  = tokenizer.detokenize(&retrieved.segments[0].tokens);
        let recovered_user = tokenizer.detokenize(&retrieved.segments[1].tokens);
        let recovered_asst = tokenizer.detokenize(&retrieved.segments[2].tokens);
        let text_ok = recovered_sys == system_text
            && recovered_user == user_text
            && recovered_asst == asst_text;

        print_result("Detokenize → original text recovered exactly", text_ok, "");

        if all_ok { passed += 1; } else { failed += 1; }
        if text_ok { passed += 1; } else { failed += 1; }
    }

    // ── Test 3: large token sequence (edge values) ────────────────────────────
    {
        // Include boundary values: 0, u32::MAX, and representative vocab IDs
        let large: Vec<u32> = (0u32..10_000)
            .map(|i| match i {
                0 => 0,
                9_999 => u32::MAX,
                i => i * 9973 % 100_277, // cl100k vocab range
            })
            .collect();

        let conv = Conversation {
            id: Some("test-large".to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![msg("user", MessageContent::Tokens(large.clone()))],
            metadata: None,
        };

        let r = vault.store(conv).await?;
        let retrieved = vault.retrieve(&r.id).await?;
        let ok_flag = retrieved.segments[0].tokens == large;

        print_result("Large sequence (10k tokens, boundary values incl. u32::MAX)", ok_flag,
            &format!("stored {} tokens, recovered {}", large.len(), retrieved.segments[0].tokens.len()));
        if ok_flag { passed += 1; } else { failed += 1; }
    }

    // ── Test 4: deduped segment recovers correctly from both conversations ────
    {
        let shared_sys = "You are a shared system prompt used across many conversations.";
        let make = |user: &str, id: &str| Conversation {
            id: Some(id.to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![
                msg("system", MessageContent::Text(shared_sys.to_owned())),
                msg("user",   MessageContent::Text(user.to_owned())),
            ],
            metadata: None,
        };

        let r1 = vault.store(make("First unique question.", "dedup-a")).await?;
        let r2 = vault.store(make("Second unique question.", "dedup-b")).await?;

        // Both conversations share the system prompt segment
        assert_eq!(r2.new_segments, 1, "only user segment should be new");
        assert_eq!(r2.deduped_segments, 1, "system prompt should be deduped");

        let c1 = vault.retrieve(&r1.id).await?;
        let c2 = vault.retrieve(&r2.id).await?;

        let tokenizer = stowken::get_tokenizer("cl100k_base").unwrap();
        let expected_sys = tokenizer.tokenize(shared_sys);

        let both_sys_ok = c1.segments[0].tokens == expected_sys
            && c2.segments[0].tokens == expected_sys;
        let hashes_match = c1.segments[0].hash == c2.segments[0].hash;

        print_result("Deduped segment recovers correctly from both conversations", both_sys_ok,
            &format!("hashes_identical={}", ok(hashes_match)));
        if both_sys_ok { passed += 1; } else { failed += 1; }
    }

    // ── Test 5: compression disabled — tokens still round-trip ───────────────
    {
        let tokens: Vec<u32> = (0u32..1_000).map(|i| i % 50_000).collect();
        let vault_no_compress = Stowken::new(
            MemoryBackend::new(),
            StowkenConfig { enable_compression: false, ..Default::default() },
        ).await?;

        let conv = Conversation {
            id: None,
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            application: None,
            messages: vec![msg("user", MessageContent::Tokens(tokens.clone()))],
            metadata: None,
        };

        let r = vault_no_compress.store(conv).await?;
        let retrieved = vault_no_compress.retrieve(&r.id).await?;
        let ok_flag = retrieved.segments[0].tokens == tokens;

        print_result("No-compression mode — tokens still round-trip exactly", ok_flag, "");
        if ok_flag { passed += 1; } else { failed += 1; }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("{}", "=".repeat(60));
    let total = passed + failed;
    if failed == 0 {
        println!("  ALL {total} TESTS PASSED — decompression is lossless");
    } else {
        println!("  {passed}/{total} passed, {failed} FAILED");
        std::process::exit(1);
    }

    Ok(())
}

fn msg(role: &str, content: MessageContent) -> Message {
    Message { role: role.to_owned(), content, name: None, tool_call_id: None }
}

fn ok(b: bool) -> &'static str { if b { "✓" } else { "✗" } }

fn print_result(label: &str, passed: bool, detail: &str) {
    let status = if passed { "PASS" } else { "FAIL" };
    if detail.is_empty() {
        println!("  [{status}] {label}");
    } else {
        println!("  [{status}] {label}");
        println!("        {detail}");
    }
}
