//! End-to-end dictionary training demonstration.
//!
//! Phase 1 — warm-up corpus (no dictionary):
//!   Store 200 conversations with natural-language-like token distributions.
//!
//! Phase 2 — train & apply:
//!   Call `train_and_apply_dictionary()` on the stored corpus.
//!
//! Phase 3 — post-dictionary corpus (same distribution):
//!   Store 200 more conversations, now compressed with the dictionary.
//!
//! Phase 4 — auto-train demonstration:
//!   Show the auto-train trigger firing in the background.
//!
//! Reports per-segment compressed sizes at each phase so the improvement
//! is directly visible.

use std::time::Instant;

use stowken::{
    storage::{MemoryBackend, StorageBackend},
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

// ── Corpus helpers ────────────────────────────────────────────────────────────

/// Zipf-ish token distribution matching natural English BPE output.
fn natural_tokens(len: usize, seed: u64) -> Vec<u32> {
    (0..len as u64).map(|i| {
        let x = lcg(seed.wrapping_add(i.wrapping_mul(6364136223846793005)));
        let frac = (x as f64) / (u64::MAX as f64);
        (frac.powf(3.0) * 100_277.0) as u32
    }).collect()
}

fn lcg(x: u64) -> u64 {
    x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn conv(id: &str, sys: Vec<u32>, user: Vec<u32>, asst: Vec<u32>) -> Conversation {
    Conversation {
        id: Some(id.to_owned()),
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        application: None,
        messages: vec![
            Message { role: "system".to_owned(),    content: MessageContent::Tokens(sys),  name: None, tool_call_id: None },
            Message { role: "user".to_owned(),      content: MessageContent::Tokens(user), name: None, tool_call_id: None },
            Message { role: "assistant".to_owned(), content: MessageContent::Tokens(asst), name: None, tool_call_id: None },
        ],
        metadata: None,
    }
}

/// Store `n` conversations, return (total_raw_bytes, total_compressed_bytes, elapsed_ms).
async fn store_batch(
    vault: &Stowken<MemoryBackend>,
    n: usize,
    id_prefix: &str,
    seed_offset: u64,
) -> (u64, u64, u128) {
    let t = Instant::now();
    let mut raw_bytes = 0u64;
    let mut compressed_bytes = 0u64;

    for i in 0..n {
        let seed = seed_offset + i as u64;
        let sys  = natural_tokens(120, seed * 3);
        let user = natural_tokens(40,  seed * 7);
        let asst = natural_tokens(80,  seed * 11);

        let total_tokens = sys.len() + user.len() + asst.len();
        raw_bytes += (total_tokens * 4) as u64;

        let id = format!("{id_prefix}_{i}");
        let r = vault.store(conv(&id, sys, user, asst)).await.unwrap();

        // compressed = raw - what dedup saved - (raw - actual stored size)
        // approximation: sum raw_size of new segments from StoreResult
        // We'll compute it from retrieved segments after the fact.
        let _ = r;
    }

    // Walk stored conversations to get actual compressed sizes.
    for i in 0..n {
        let id = format!("{id_prefix}_{i}");
        if let Ok(manifest) = vault.backend().get_manifest(&id).await {
            for seg_ref in &manifest.segments {
                if let Ok(stored) = vault.backend().get_segment(&seg_ref.hash).await {
                    compressed_bytes += stored.compressed_size as u64;
                }
            }
        }
    }

    (raw_bytes, compressed_bytes, t.elapsed().as_millis())
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WARM_UP: usize = 200;
    const POST_DICT: usize = 200;
    const SAMPLE_COUNT: usize = 150;

    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;

    println!("Token distribution: Zipf-ish natural language (cl100k_base-like)");
    println!("Segment sizes: system=120, user=40, assistant=80 tokens");
    println!();

    // ── Phase 1: warm-up without dictionary ──────────────────────────────────
    println!("── Phase 1: storing {WARM_UP} conversations (no dictionary) ───────────────");
    let (raw1, comp1, ms1) = store_batch(&vault, WARM_UP, "warmup", 0).await;
    let ratio1 = comp1 as f64 / raw1 as f64;
    let saving1 = (1.0 - ratio1) * 100.0;
    println!("  Raw bytes       : {:>10}  ({:.1} KB)", raw1, raw1 as f64 / 1024.0);
    println!("  Compressed      : {:>10}  ({:.1} KB)", comp1, comp1 as f64 / 1024.0);
    println!("  Compression ratio: {ratio1:.4}  ({saving1:.1}% savings)");
    println!("  Bytes/token (avg): {:.3}", comp1 as f64 / (raw1 as f64 / 4.0));
    println!("  Time             : {ms1}ms");
    println!();

    // ── Phase 2: train and apply dictionary ──────────────────────────────────
    println!("── Phase 2: training dictionary on {SAMPLE_COUNT} segments ─────────────────────");
    let t = Instant::now();
    let n = vault.train_and_apply_dictionary("cl100k_base", SAMPLE_COUNT).await?;
    let train_ms = t.elapsed().as_millis();
    println!("  Samples used     : {n}");
    println!("  Training time    : {train_ms}ms");
    println!();

    // ── Phase 3: post-dictionary corpus ──────────────────────────────────────
    println!("── Phase 3: storing {POST_DICT} conversations (with dictionary) ─────────────");
    let (raw3, comp3, ms3) = store_batch(&vault, POST_DICT, "postdict", 10_000).await;
    let ratio3 = comp3 as f64 / raw3 as f64;
    let saving3 = (1.0 - ratio3) * 100.0;
    println!("  Raw bytes        : {:>10}  ({:.1} KB)", raw3, raw3 as f64 / 1024.0);
    println!("  Compressed       : {:>10}  ({:.1} KB)", comp3, comp3 as f64 / 1024.0);
    println!("  Compression ratio: {ratio3:.4}  ({saving3:.1}% savings)");
    println!("  Bytes/token (avg): {:.3}", comp3 as f64 / (raw3 as f64 / 4.0));
    println!("  Time             : {ms3}ms");
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────
    let improvement_pp = saving3 - saving1;
    let size_reduction = comp1 as f64 / comp3 as f64;
    println!("── Summary ──────────────────────────────────────────────────────────────────");
    println!("  Without dictionary : {saving1:.1}% savings  ({:.3} bytes/token)",
        comp1 as f64 / (raw1 as f64 / 4.0));
    println!("  With dictionary    : {saving3:.1}% savings  ({:.3} bytes/token)",
        comp3 as f64 / (raw3 as f64 / 4.0));
    println!("  Improvement        : +{improvement_pp:.1}pp");
    println!("  Compressed size    : {:.2}× smaller with dictionary", size_reduction);
    println!();

    // ── Per-segment size breakdown ────────────────────────────────────────────
    println!("── Per-segment breakdown (first 5 conversations each phase) ─────────────────");
    println!("{:>8}  {:>14}  {:>14}  {:>14}  {:>14}",
        "phase", "segment", "raw (bytes)", "no-dict (B)", "dict (B)");
    println!("{}", "-".repeat(70));

    for (prefix, label) in [("warmup", "no-dict"), ("postdict", "dict")] {
        for i in 0..5 {
            let id = format!("{prefix}_{i}");
            let manifest = vault.backend().get_manifest(&id).await?;
            for seg_ref in &manifest.segments {
                let stored = vault.backend().get_segment(&seg_ref.hash).await?;
                let raw = seg_ref.token_count * 4;
                let seg_type = format!("{:?}", seg_ref.segment_type);
                let (col_a, col_b) = if label == "no-dict" {
                    (format!("{}", stored.compressed_size), "-".to_owned())
                } else {
                    ("-".to_owned(), format!("{}", stored.compressed_size))
                };
                println!("{:>8}  {:>14}  {:>14}  {:>14}  {:>14}",
                    label, seg_type, raw, col_a, col_b);
            }
        }
        println!();
    }

    // ── Auto-train demonstration ──────────────────────────────────────────────
    println!("── Auto-train demonstration ──────────────────────────────────────────────────");
    let auto_config = StowkenConfig {
        auto_train_dictionary_after: Some(20),
        dictionary_sample_count: 15,
        ..StowkenConfig::default()
    };
    let auto_vault = Stowken::new(MemoryBackend::new(), auto_config).await?;

    println!("  Config: auto_train_dictionary_after=20, sample_count=15");
    println!("  Storing 50 conversations — dictionary should auto-train after segment #20...");

    for i in 0..50usize {
        let sys  = natural_tokens(120, i as u64 * 5);
        let user = natural_tokens(40,  i as u64 * 9);
        let asst = natural_tokens(80,  i as u64 * 13);
        auto_vault.store(conv(&format!("auto_{i}"), sys, user, asst)).await?;
    }

    // Give background task time to finish.
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // Verify the pipeline is working correctly post-auto-train.
    let check_tokens = natural_tokens(100, 99_999);
    auto_vault.store(conv("check", check_tokens.clone(),
        natural_tokens(20, 88_888), natural_tokens(20, 77_777))).await?;
    let retrieved = auto_vault.retrieve("check").await?;
    let sys_seg = &retrieved.segments[0];
    let ok = sys_seg.tokens == check_tokens;
    println!("  Roundtrip after auto-train: {}", if ok { "✓ PASS" } else { "✗ FAIL" });

    // Show final compressed size of a post-auto-train segment.
    let stored = auto_vault.backend().get_segment(&sys_seg.hash).await?;
    let raw_bytes = sys_seg.token_count * 4;
    println!("  Post-auto-train segment: {} raw → {} compressed ({:.1}% savings)",
        raw_bytes, stored.compressed_size,
        (1.0 - stored.compressed_size as f64 / raw_bytes as f64) * 100.0);

    Ok(())
}
