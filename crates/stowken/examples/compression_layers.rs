//! Comprehensive compression layer benchmark.
//!
//! Tests each compression layer in isolation and in combination across four
//! realistic token distribution profiles:
//!
//!   - natural_language   : cl100k_base English text (Zipf, many low-ID tokens)
//!   - code               : mixed identifiers and keywords (flatter distribution)
//!   - repetitive_system  : short system prompt repeated 100×
//!   - high_entropy       : random u32 IDs across full vocab (worst case)
//!
//! Layers tested:
//!   1. Raw u32           — baseline (4 bytes/token, no compression)
//!   2. Varint only       — variable-width encoding, no zstd
//!   3. zstd only         — raw u32 bytes → zstd (no varint pre-processing)
//!   4. Varint + zstd     — the default pipeline (level 3)
//!   5. Varint + zstd L9  — max compression level
//!   6. Varint + zstd + dict — with a trained dictionary (110 KiB)
//!
//! Each cell shows compressed size and savings % vs raw u32.

use std::time::Instant;

use stowken::compression::{
    dictionary::DictionaryCompressor,
    varint,
};

// ── Corpus generators ─────────────────────────────────────────────────────────

/// Simulate natural language tokens: Zipf distribution, vocab 100K.
/// ~60% of tokens are < 5000, modelling high-frequency English words.
fn natural_language(len: usize, seed: u64) -> Vec<u32> {
    (0..len as u64).map(|i| {
        let x = lcg(seed.wrapping_add(i * 6364136223846793005));
        // Zipf-like: cube-root mapping concentrates values in low range
        let frac = (x as f64) / (u64::MAX as f64);
        (frac.powf(3.0) * 100_277.0) as u32
    }).collect()
}

/// Simulate code tokens: more uniform distribution (identifiers span the vocab).
fn code_tokens(len: usize, seed: u64) -> Vec<u32> {
    (0..len as u64).map(|i| {
        let x = lcg(seed.wrapping_add(i * 2862933555777941757));
        // Sqrt mapping — flatter than natural language but still somewhat skewed
        let frac = (x as f64) / (u64::MAX as f64);
        (frac.sqrt() * 100_277.0) as u32
    }).collect()
}

/// Repetitive system prompt: same 50 tokens repeated to fill `len`.
fn repetitive(len: usize, seed: u64) -> Vec<u32> {
    let base: Vec<u32> = (0..50).map(|i| ((seed as u32).wrapping_add(i * 17)) % 5000).collect();
    (0..len).map(|i| base[i % base.len()]).collect()
}

/// Worst-case: random u32 across full u32 range.
fn high_entropy(len: usize, seed: u64) -> Vec<u32> {
    (0..len as u64).map(|i| {
        lcg(seed.wrapping_add(i * 9999999999999)) as u32
    }).collect()
}

fn lcg(x: u64) -> u64 {
    x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

// ── Compression methods ───────────────────────────────────────────────────────

fn raw_u32(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

fn varint_only(tokens: &[u32]) -> Vec<u8> {
    varint::encode_tokens(tokens)
}

fn zstd_only(level: i32) -> impl Fn(&[u32]) -> Vec<u8> {
    move |tokens| {
        let raw = raw_u32(tokens);
        let c = DictionaryCompressor::new(level);
        c.compress(&raw).unwrap()
    }
}

fn varint_zstd(level: i32) -> impl Fn(&[u32]) -> Vec<u8> {
    move |tokens| {
        let encoded = varint::encode_tokens(tokens);
        let c = DictionaryCompressor::new(level);
        c.compress(&encoded).unwrap()
    }
}

fn varint_zstd_dict(dict: Vec<u8>, level: i32) -> impl Fn(&[u32]) -> Vec<u8> {
    move |tokens| {
        let encoded = varint::encode_tokens(tokens);
        let c = DictionaryCompressor::with_dictionary(dict.clone(), level);
        c.compress(&encoded).unwrap()
    }
}

fn pct(compressed: usize, raw: usize) -> f64 {
    (1.0 - compressed as f64 / raw as f64) * 100.0
}

fn bytes_per_token(compressed: usize, n_tokens: usize) -> f64 {
    compressed as f64 / n_tokens as f64
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    const TOKENS_PER_CORPUS: usize = 10_000;

    // ── Build corpora ─────────────────────────────────────────────────────────
    let corpora: Vec<(&str, Vec<u32>)> = vec![
        ("natural_language ", natural_language(TOKENS_PER_CORPUS, 42)),
        ("code              ", code_tokens(TOKENS_PER_CORPUS, 99)),
        ("repetitive_system ", repetitive(TOKENS_PER_CORPUS, 7)),
        ("high_entropy      ", high_entropy(TOKENS_PER_CORPUS, 13)),
    ];

    // ── Train a shared dictionary on natural-language samples ─────────────────
    // 200 segments of 100 tokens each — representative of real system prompts.
    println!("Training zstd dictionary on 200 × 100-token natural-language samples…");
    let t = Instant::now();
    let dict_samples: Vec<Vec<u8>> = (0..200u64)
        .map(|seed| varint::encode_tokens(&natural_language(100, seed * 1337)))
        .collect();
    let dict = DictionaryCompressor::train_dictionary(&dict_samples, 112_640)
        .expect("dictionary training failed");
    println!("  Dictionary size: {:.1} KiB  (trained in {}ms)\n", dict.len() as f64 / 1024.0, t.elapsed().as_millis());

    // ── Define methods ────────────────────────────────────────────────────────
    let methods: Vec<(&str, Box<dyn Fn(&[u32]) -> Vec<u8>>)> = vec![
        ("raw u32            ", Box::new(raw_u32)),
        ("varint only        ", Box::new(varint_only)),
        ("zstd-only  L3      ", Box::new(zstd_only(3))),
        ("varint+zstd L3     ", Box::new(varint_zstd(3))),
        ("varint+zstd L9     ", Box::new(varint_zstd(9))),
        ("varint+zstd+dict L3", Box::new(varint_zstd_dict(dict.clone(), 3))),
        ("varint+zstd+dict L9", Box::new(varint_zstd_dict(dict.clone(), 9))),
    ];

    // ── Results table ─────────────────────────────────────────────────────────
    let raw_bytes = TOKENS_PER_CORPUS * 4;
    println!("Corpus: {TOKENS_PER_CORPUS} tokens each  (raw = {} bytes = {:.1} KiB)\n", raw_bytes, raw_bytes as f64 / 1024.0);

    let col_w = 22;
    print!("{:<20}", "Method");
    for (name, _) in &corpora { print!("{:>col_w$}", name.trim()); }
    println!();
    print!("{:<20}", "");
    for _ in &corpora { print!("{:>col_w$}", "(bytes / saved%)"); }
    println!();
    println!("{}", "-".repeat(20 + col_w * corpora.len()));

    for (mname, compress_fn) in &methods {
        print!("{:<20}", mname.trim());
        for (_, tokens) in &corpora {
            let compressed = compress_fn(tokens);
            let sz = compressed.len();
            let saving = pct(sz, raw_bytes);
            let bpt = bytes_per_token(sz, TOKENS_PER_CORPUS);
            if saving <= 0.0 {
                print!("{:>col_w$}", format!("{sz} ({saving:+.0}% ↑)"));
            } else {
                print!("{:>col_w$}", format!("{sz} ({saving:.0}% ↓)"));
            }
            let _ = bpt;
        }
        println!();
    }

    println!();

    // ── Bytes-per-token table ─────────────────────────────────────────────────
    println!("Bytes per token:\n");
    print!("{:<20}", "Method");
    for (name, _) in &corpora { print!("{:>col_w$}", name.trim()); }
    println!();
    println!("{}", "-".repeat(20 + col_w * corpora.len()));

    for (mname, compress_fn) in &methods {
        print!("{:<20}", mname.trim());
        for (_, tokens) in &corpora {
            let compressed = compress_fn(tokens);
            let bpt = bytes_per_token(compressed.len(), TOKENS_PER_CORPUS);
            print!("{:>col_w$}", format!("{bpt:.3}"));
        }
        println!();
    }

    println!();

    // ── Speed benchmark ───────────────────────────────────────────────────────
    println!("Compression speed (ns/token, 500 iterations, natural_language corpus):\n");
    let nl_tokens = &corpora[0].1;
    for (mname, compress_fn) in &methods {
        let t = Instant::now();
        for _ in 0..500 {
            let _ = compress_fn(nl_tokens);
        }
        let total_ns = t.elapsed().as_nanos();
        let ns_per_token = total_ns as f64 / (500 * TOKENS_PER_CORPUS) as f64;
        let ms_per_seg = total_ns as f64 / 500_000.0 / 1000.0;  // ms per 10K-token segment
        println!("  {:<22} {:>8.2} ns/token   ({:.2}ms / 10K-token segment)",
            mname.trim(), ns_per_token, ms_per_seg);
    }

    println!();

    // ── Short-segment deep-dive ───────────────────────────────────────────────
    // zstd struggles on short segments without a dictionary — show the crossover.
    println!("Short-segment behaviour (natural_language, varint+zstd vs varint+zstd+dict):\n");
    println!("{:>10}  {:>16}  {:>20}  {:>20}  {:>10}",
        "tokens", "raw (bytes)", "varint+zstd L3", "varint+dict L3", "dict gain");
    println!("{}", "-".repeat(82));

    let nodic = |tokens: &[u32]| {
        let enc = varint::encode_tokens(tokens);
        DictionaryCompressor::new(3).compress(&enc).unwrap()
    };
    let withdict = {
        let d = dict.clone();
        move |tokens: &[u32]| {
            let enc = varint::encode_tokens(tokens);
            DictionaryCompressor::with_dictionary(d.clone(), 3).compress(&enc).unwrap()
        }
    };

    for n in [10, 25, 50, 100, 200, 500, 1000, 2000, 5000] {
        let tokens = natural_language(n, 77);
        let raw = n * 4;
        let nd = nodic(&tokens).len();
        let wd = withdict(&tokens).len();
        let gain_pct = (nd as f64 - wd as f64) / nd as f64 * 100.0;
        println!("{:>10}  {:>16}  {:>16} ({:>4.0}%)  {:>16} ({:>4.0}%)  {:>+9.1}%",
            n, raw,
            nd, pct(nd, raw),
            wd, pct(wd, raw),
            gain_pct);
    }

    println!();

    // ── Level comparison ─────────────────────────────────────────────────────
    println!("zstd level comparison (varint+zstd, natural_language 10K tokens):\n");
    println!("{:>8}  {:>14}  {:>10}  {:>14}", "level", "size (bytes)", "saved %", "ms / segment");
    println!("{}", "-".repeat(52));
    for level in [1, 3, 6, 9, 12, 15, 19, 22] {
        let compress = varint_zstd(level);
        let t = Instant::now();
        let mut sz = 0;
        for _ in 0..200 { sz = compress(nl_tokens).len(); }
        let ms = t.elapsed().as_millis() as f64 / 200.0;
        println!("{:>8}  {:>14}  {:>9.1}%  {:>12.2}ms", level, sz, pct(sz, raw_bytes), ms);
    }
}
