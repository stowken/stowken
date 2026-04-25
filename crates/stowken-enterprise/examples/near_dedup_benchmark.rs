//! Benchmark: near-dedup savings vs exact-only dedup.
//!
//! Simulates a realistic corpus of system prompts with per-user interpolation:
//!   - 5 base system prompt templates (each ~100 tokens)
//!   - 20 users per template → each user has a unique token or two changed
//!   - 4 conversations per user (same prompt, different user questions)
//!
//! Total: 5 × 20 × 4 = 400 conversations.
//!
//! Reports:
//!   - Exact-only dedup baseline (core Stowken)
//!   - Near-dedup savings on top (EnterpriseStowken)
//!   - Combined savings %

use std::time::Instant;

use stowken::{storage::MemoryBackend, types::StowkenConfig, Stowken};
use stowken_enterprise::enterprise_vault::EnterpriseStowken;

fn make_conv(
    id: &str,
    sys_tokens: Vec<u32>,
    user_tokens: Vec<u32>,
) -> stowken::types::Conversation {
    use stowken::types::{Message, MessageContent};
    stowken::types::Conversation {
        id: Some(id.to_owned()),
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        application: None,
        messages: vec![
            Message {
                role: "system".to_owned(),
                content: MessageContent::Tokens(sys_tokens),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_owned(),
                content: MessageContent::Tokens(user_tokens),
                name: None,
                tool_call_id: None,
            },
        ],
        metadata: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Corpus parameters ─────────────────────────────────────────────────────
    const TEMPLATES: usize = 5;
    const USERS_PER_TEMPLATE: usize = 20;
    const CONVS_PER_USER: usize = 4;
    const SYS_TOKENS: usize = 100;
    const USER_TOKENS: usize = 30;

    // Build base system prompt templates (deterministic token sequences)
    let base_templates: Vec<Vec<u32>> = (0..TEMPLATES)
        .map(|t| (0u32..SYS_TOKENS as u32).map(|i| i + t as u32 * 1000).collect())
        .collect();

    // Per-user variants: change 2 tokens in positions 10 and 50 (e.g. interpolated name)
    let mut all_convs: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
    let mut conv_idx = 0usize;

    for (t, base) in base_templates.iter().enumerate() {
        for u in 0..USERS_PER_TEMPLATE {
            let mut user_sys = base.clone();
            // Unique interpolation: positions 10 and 50 carry user-specific token IDs
            user_sys[10] = 900_000 + (t * USERS_PER_TEMPLATE + u) as u32;
            user_sys[50] = 800_000 + (t * USERS_PER_TEMPLATE + u) as u32;

            for c in 0..CONVS_PER_USER {
                let user_msg: Vec<u32> = (0u32..USER_TOKENS as u32)
                    .map(|i| 700_000 + (conv_idx * USER_TOKENS) as u32 + i)
                    .collect();
                let id = format!("t{t}_u{u}_c{c}");
                all_convs.push((id, user_sys.clone(), user_msg));
                conv_idx += 1;
            }
        }
    }

    let total_raw_bytes: u64 = all_convs
        .iter()
        .map(|(_, s, u)| ((s.len() + u.len()) * 4) as u64)
        .sum();

    println!("Corpus: {} conversations, {:.1} KB raw token data",
        all_convs.len(),
        total_raw_bytes as f64 / 1024.0);
    println!();

    // ── Baseline: core Stowken (exact dedup only) ──────────────────────────
    let core_vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;
    let t = Instant::now();
    let mut core_new = 0u64;
    let mut core_deduped = 0u64;
    let mut core_bytes_saved = 0u64;

    for (id, sys, user) in &all_convs {
        let r = core_vault.store(make_conv(id, sys.clone(), user.clone())).await?;
        core_new += r.new_segments;
        core_deduped += r.deduped_segments;
        core_bytes_saved += r.bytes_saved;
    }
    let core_ms = t.elapsed().as_millis();
    let core_saving_pct = core_bytes_saved as f64 / total_raw_bytes as f64 * 100.0;
    let core_stored_bytes = total_raw_bytes - core_bytes_saved;

    println!("── Exact dedup only (core Stowken) ──────────────────────────────────");
    println!("  New segments   : {core_new}");
    println!("  Deduped segs   : {core_deduped}");
    println!("  Raw bytes saved: {:.1} KB  ({core_saving_pct:.1}%)",
        core_bytes_saved as f64 / 1024.0);
    println!("  Bytes stored   : {:.1} KB", core_stored_bytes as f64 / 1024.0);
    println!("  Time           : {core_ms}ms");
    println!();

    // ── Near-dedup: EnterpriseStowken (exact + MinHash LSH) ─────────────────────
    let nd_vault = EnterpriseStowken::new(
        MemoryBackend::new(),
        StowkenConfig::default(),
        0.80,
    ).await?;
    let t = Instant::now();
    let mut nd_new = 0u64;
    let mut nd_exact_deduped = 0u64;
    let mut nd_near_deduped = 0u64;
    let mut nd_bytes_saved = 0u64;
    let mut nd_near_bytes_saved = 0u64;

    for (id, sys, user) in &all_convs {
        let r = nd_vault.store(make_conv(id, sys.clone(), user.clone())).await?;
        nd_new += r.core.new_segments;
        nd_exact_deduped += r.core.deduped_segments;
        nd_near_deduped += r.near_dedup_segments;
        nd_bytes_saved += r.core.bytes_saved;
        nd_near_bytes_saved += r.near_dedup_bytes_saved;
    }
    let nd_ms = t.elapsed().as_millis();
    let total_saved = nd_bytes_saved + nd_near_bytes_saved;
    let nd_saving_pct = total_saved as f64 / total_raw_bytes as f64 * 100.0;
    let nd_stored_bytes = total_raw_bytes - total_saved;

    let stats = nd_vault.near_dedup_stats();

    println!("── Near-dedup (EnterpriseStowken, threshold=0.80) ─────────────────────────");
    println!("  New segments       : {nd_new}");
    println!("  Exact deduped      : {nd_exact_deduped}");
    println!("  Near-dedup linked  : {nd_near_deduped}");
    println!("  Exact bytes saved  : {:.1} KB  ({:.1}%)",
        nd_bytes_saved as f64 / 1024.0,
        nd_bytes_saved as f64 / total_raw_bytes as f64 * 100.0);
    println!("  Near-dup bytes saved: {:.1} KB  ({:.1}%)",
        nd_near_bytes_saved as f64 / 1024.0,
        nd_near_bytes_saved as f64 / total_raw_bytes as f64 * 100.0);
    println!("  Total bytes saved  : {:.1} KB  ({nd_saving_pct:.1}%)",
        total_saved as f64 / 1024.0);
    println!("  Bytes stored       : {:.1} KB", nd_stored_bytes as f64 / 1024.0);
    println!("  LSH hits           : {}", stats.near_dup_hits);
    println!("  Time               : {nd_ms}ms");
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────
    let improvement = nd_saving_pct - core_saving_pct;
    println!("── Summary ──────────────────────────────────────────────────────────────");
    println!("  Exact-only savings : {core_saving_pct:.1}%");
    println!("  Near-dedup savings : {nd_saving_pct:.1}%");
    println!("  Improvement        : +{improvement:.1}pp");
    println!("  Storage reduction  : {:.1} KB → {:.1} KB  ({:.1}x smaller)",
        core_stored_bytes as f64 / 1024.0,
        nd_stored_bytes as f64 / 1024.0,
        core_stored_bytes as f64 / (nd_stored_bytes as f64).max(1.0));

    Ok(())
}
