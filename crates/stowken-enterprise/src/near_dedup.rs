//! MinHash LSH near-duplicate detection and delta storage.
//!
//! # Design
//!
//! Exact dedup (SHA-256 hash equality) misses segments that are *almost* the
//! same — system prompts with an interpolated user name, RAG chunks with minor
//! formatting tweaks, tool calls with slightly different arguments.
//!
//! This module adds a second dedup layer:
//!
//! 1. **MinHash** — each token sequence is reduced to a 128-element signature
//!    using token-level n-gram shingles and 128 independent hash functions.
//! 2. **LSH banding** — the signature is split into `BANDS` bands of
//!    `ROWS_PER_BAND` values.  Two segments are candidates if any band hash
//!    collides.  With BANDS=16 and ROWS=8 the detection probability for
//!    similarity ≥ 0.85 is > 99%.
//! 3. **Delta encoding** — near-duplicates are stored as `(canonical_hash,
//!    edit_script)` instead of full token arrays, typically saving 80-95% of
//!    the variant's storage cost.
//!
//! # Usage
//!
//! ```rust
//! use stowken_enterprise::near_dedup::{NearDedupIndex, NearDedupAction};
//!
//! let mut idx = NearDedupIndex::new(0.85);
//!
//! let canonical: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! let canonical_hash = stowken::types::SegmentHash("abc".into());
//! idx.insert(canonical_hash.clone(), &canonical);
//!
//! let variant: Vec<u32> = vec![1, 2, 3, 4, 99, 6, 7, 8, 9, 10]; // one token changed
//! match idx.process(&variant) {
//!     NearDedupAction::StoreAsNew => println!("no near-dup found"),
//!     NearDedupAction::LinkToCanonical { canonical_hash, delta, similarity } => {
//!         println!("near-dup of {canonical_hash} ({similarity:.0}%)");
//!         let _ = delta; // store this instead of full tokens
//!     }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use stowken::types::{SegmentHash, Token};

// ── Tuning constants ──────────────────────────────────────────────────────────

/// Token-level shingle (n-gram) size.
///
/// Bigrams (size 2) are used rather than trigrams: each changed token corrupts
/// only 2 shingles instead of 3, keeping Jaccard high enough for reliable
/// detection of the common case (1–5 token changes in 50–200 token segments).
const NGRAM_SIZE: usize = 2;
/// Total number of MinHash functions in the signature.
const NUM_HASHES: usize = 128;
/// Number of LSH bands.  Higher → fewer false negatives at given threshold.
const BANDS: usize = 16;
/// Hash functions per band.  `BANDS * ROWS_PER_BAND == NUM_HASHES`.
const ROWS_PER_BAND: usize = NUM_HASHES / BANDS; // 8

// ── MinHash signature ─────────────────────────────────────────────────────────

/// A 128-element MinHash signature for a token sequence.
#[derive(Debug, Clone)]
pub struct MinHashSignature {
    pub values: [u64; NUM_HASHES],
}

/// Compute the MinHash signature of a token sequence.
///
/// Uses token-level n-gram shingles and 128 independent hash functions.
/// Sequences shorter than `NGRAM_SIZE` get a degenerate whole-sequence shingle.
pub fn compute_signature(tokens: &[Token]) -> MinHashSignature {
    let mut values = [u64::MAX; NUM_HASHES];

    if tokens.len() < NGRAM_SIZE {
        // Short sequence: treat the whole thing as one shingle
        for (i, val) in values.iter_mut().enumerate() {
            *val = hash_ngram(tokens, i as u64);
        }
    } else {
        for ngram in tokens.windows(NGRAM_SIZE) {
            for (i, val) in values.iter_mut().enumerate() {
                let h = hash_ngram(ngram, i as u64);
                if h < *val {
                    *val = h;
                }
            }
        }
    }

    MinHashSignature { values }
}

/// Estimate Jaccard similarity between two MinHash signatures (0.0–1.0).
pub fn jaccard_estimate(a: &MinHashSignature, b: &MinHashSignature) -> f64 {
    let matches = a.values.iter().zip(b.values.iter()).filter(|(x, y)| x == y).count();
    matches as f64 / NUM_HASHES as f64
}

fn hash_ngram(ngram: &[Token], seed: u64) -> u64 {
    let mut h = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    for &tok in ngram {
        h ^= u64::from(tok).wrapping_mul(0x6c62_272e_07bb_0142);
        h = h.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    h ^ (h >> 31)
}

// ── LSH band index ────────────────────────────────────────────────────────────

/// Hash a single LSH band to a bucket key.
fn band_hash(sig: &MinHashSignature, band: usize) -> u64 {
    let start = band * ROWS_PER_BAND;
    let mut h: u64 = (band as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    for &v in &sig.values[start..start + ROWS_PER_BAND] {
        h = h.wrapping_add(v).rotate_left(17).wrapping_mul(0x6c62_272e_07bb_0142);
    }
    h
}

// ── NearDedupIndex ────────────────────────────────────────────────────────────

/// Tracks signatures and LSH band buckets for all indexed segments.
pub struct NearDedupIndex {
    /// Similarity threshold (0.0–1.0) for declaring a near-duplicate.
    threshold: f64,
    /// band index → bucket key → list of candidate segment hashes.
    bands: Vec<HashMap<u64, Vec<SegmentHash>>>,
    /// Segment hash → stored signature (for candidate verification).
    signatures: HashMap<SegmentHash, MinHashSignature>,
    /// Segment hash → stored tokens (needed for delta computation).
    tokens: HashMap<SegmentHash, Vec<Token>>,
    /// Running stats.
    pub stats: NearDedupStats,
}

/// Aggregate statistics from the near-dedup index.
#[derive(Debug, Default, Clone)]
pub struct NearDedupStats {
    pub segments_indexed: u64,
    pub near_dup_hits: u64,
    pub exact_dup_hits: u64,
    /// Estimated token bytes saved by near-dedup (variant tokens avoided).
    pub bytes_saved: u64,
}

/// The outcome of processing a segment through the near-dedup index.
pub enum NearDedupAction {
    /// No near-duplicate found; store this segment as a new unique segment.
    StoreAsNew,
    /// A near-duplicate was found; store only the delta.
    LinkToCanonical {
        canonical_hash: SegmentHash,
        delta: SegmentDelta,
        /// Estimated Jaccard similarity (0.0–1.0).
        similarity: f64,
    },
}

impl NearDedupIndex {
    /// Create a new index with the given similarity threshold.
    ///
    /// Recommended threshold: `0.85` — catches near-duplicates while keeping
    /// false-positive rates negligible.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            bands: (0..BANDS).map(|_| HashMap::new()).collect(),
            signatures: HashMap::new(),
            tokens: HashMap::new(),
            stats: NearDedupStats::default(),
        }
    }

    /// Index a segment.  Call this after deciding to store a segment as new.
    pub fn insert(&mut self, hash: SegmentHash, tokens: &[Token]) {
        let sig = compute_signature(tokens);
        for band in 0..BANDS {
            let bh = band_hash(&sig, band);
            self.bands[band].entry(bh).or_default().push(hash.clone());
        }
        self.signatures.insert(hash.clone(), sig);
        self.tokens.insert(hash, tokens.to_vec());
        self.stats.segments_indexed += 1;
    }

    /// Check whether `tokens` is a near-duplicate of any indexed segment.
    ///
    /// Returns `StoreAsNew` if no match is found, or `LinkToCanonical` with
    /// the best-matching canonical and the computed delta.
    pub fn process(&mut self, tokens: &[Token]) -> NearDedupAction {
        let sig = compute_signature(tokens);
        let candidates = self.candidates(&sig);

        let mut best_hash: Option<SegmentHash> = None;
        let mut best_sim = 0.0f64;

        for candidate in candidates {
            if let Some(stored_sig) = self.signatures.get(&candidate) {
                let sim = jaccard_estimate(&sig, stored_sig);
                if sim >= self.threshold && sim > best_sim {
                    best_sim = sim;
                    best_hash = Some(candidate);
                }
            }
        }

        if let Some(canonical_hash) = best_hash {
            let canonical_tokens = self.tokens[&canonical_hash].clone();
            let delta = compute_delta(&canonical_tokens, tokens, canonical_hash.clone());
            let bytes_saved = (tokens.len() as u64) * 4;
            self.stats.near_dup_hits += 1;
            self.stats.bytes_saved += bytes_saved;
            NearDedupAction::LinkToCanonical {
                canonical_hash,
                delta,
                similarity: best_sim,
            }
        } else {
            NearDedupAction::StoreAsNew
        }
    }

    /// Collect LSH candidate hashes (segments sharing any band bucket).
    fn candidates(&self, sig: &MinHashSignature) -> Vec<SegmentHash> {
        let mut seen: HashMap<&SegmentHash, ()> = HashMap::new();
        let mut out = Vec::new();
        for band in 0..BANDS {
            let bh = band_hash(sig, band);
            if let Some(bucket) = self.bands[band].get(&bh) {
                for hash in bucket {
                    if seen.insert(hash, ()).is_none() {
                        out.push(hash.clone());
                    }
                }
            }
        }
        out
    }

    /// Number of segments currently indexed.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }
}

// ── Delta encoding ────────────────────────────────────────────────────────────

/// Stored representation of a near-duplicate segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDelta {
    /// Content hash of the canonical (stored) segment.
    pub canonical_hash: SegmentHash,
    /// Edit operations to reconstruct this segment from the canonical one.
    pub operations: Vec<DeltaOp>,
    /// Serialised byte size of this delta (for savings reporting).
    pub delta_bytes: u32,
}

/// An individual edit operation in a token-level diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaOp {
    /// Keep `length` tokens starting at `offset` in the canonical segment.
    Keep { offset: u32, length: u32 },
    /// Insert new tokens at this position in the output.
    Insert { tokens: Vec<Token> },
    /// Delete `length` tokens starting at `offset` from the canonical segment.
    Delete { offset: u32, length: u32 },
}

/// Compute the token-level delta to transform `canonical` into `variant`.
pub fn compute_delta(canonical: &[Token], variant: &[Token], canonical_hash: SegmentHash) -> SegmentDelta {
    let operations = token_diff(canonical, variant);
    let delta_bytes = estimate_delta_bytes(&operations);
    SegmentDelta { canonical_hash, operations, delta_bytes }
}

/// Apply a delta to a canonical token sequence to recover the variant.
pub fn apply_delta(canonical: &[Token], delta: &SegmentDelta) -> Vec<Token> {
    let mut out = Vec::new();
    for op in &delta.operations {
        match op {
            DeltaOp::Keep { offset, length } => {
                let start = *offset as usize;
                let end = (start + *length as usize).min(canonical.len());
                out.extend_from_slice(&canonical[start..end]);
            }
            DeltaOp::Insert { tokens } => out.extend_from_slice(tokens),
            DeltaOp::Delete { .. } => {}
        }
    }
    out
}

fn estimate_delta_bytes(ops: &[DeltaOp]) -> u32 {
    ops.iter().map(|op| match op {
        DeltaOp::Keep { .. } => 8,
        DeltaOp::Delete { .. } => 8,
        DeltaOp::Insert { tokens } => 4 + tokens.len() as u32 * 4,
    }).sum()
}

/// Token-level diff producing a minimal edit script (greedy LCS).
fn token_diff(a: &[Token], b: &[Token]) -> Vec<DeltaOp> {
    let mut ops = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() || j < b.len() {
        // Longest common prefix from current positions
        let mut run = 0usize;
        while i + run < a.len() && j + run < b.len() && a[i + run] == b[j + run] {
            run += 1;
        }
        if run > 0 {
            ops.push(DeltaOp::Keep { offset: i as u32, length: run as u32 });
            i += run;
            j += run;
        } else if i < a.len() && j < b.len() {
            ops.push(DeltaOp::Delete { offset: i as u32, length: 1 });
            ops.push(DeltaOp::Insert { tokens: vec![b[j]] });
            i += 1;
            j += 1;
        } else if i < a.len() {
            ops.push(DeltaOp::Delete { offset: i as u32, length: (a.len() - i) as u32 });
            i = a.len();
        } else {
            ops.push(DeltaOp::Insert { tokens: b[j..].to_vec() });
            j = b.len();
        }
    }
    ops
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(s: &str) -> SegmentHash { SegmentHash(s.to_owned()) }

    #[test]
    fn identical_sequences_have_similarity_one() {
        let tokens: Vec<Token> = (0..50).collect();
        let a = compute_signature(&tokens);
        let b = compute_signature(&tokens);
        assert!((jaccard_estimate(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn disjoint_sequences_have_low_similarity() {
        let a: Vec<Token> = (0..50).collect();
        let b: Vec<Token> = (100..150).collect();
        let sim = jaccard_estimate(&compute_signature(&a), &compute_signature(&b));
        assert!(sim < 0.5, "got {sim}");
    }

    #[test]
    fn delta_roundtrip() {
        let canonical: Vec<Token> = vec![1, 2, 3, 4, 5];
        let variant:   Vec<Token> = vec![1, 2, 99, 4, 5, 6];
        let delta = compute_delta(&canonical, &variant, hash("h1"));
        let recovered = apply_delta(&canonical, &delta);
        assert_eq!(recovered, variant);
    }

    #[test]
    fn identical_sequences_produce_only_keep_ops() {
        let tokens: Vec<Token> = vec![1, 2, 3, 4, 5];
        let delta = compute_delta(&tokens, &tokens, hash("h"));
        assert!(delta.operations.iter().all(|op| matches!(op, DeltaOp::Keep { .. })));
    }

    #[test]
    fn index_detects_near_duplicate() {
        let mut idx = NearDedupIndex::new(0.80);

        let canonical: Vec<Token> = (0u32..100).collect();
        idx.insert(hash("canonical"), &canonical);

        // Variant: change 2 tokens out of 100 — realistic (e.g. interpolated name).
        // With bigrams: Jaccard ≈ (99-4)/(99+4) ≈ 0.92 → P(detect) > 0.9999
        let mut variant = canonical.clone();
        variant[10] = 9999;
        variant[50] = 9998;

        let action = idx.process(&variant);
        assert!(matches!(action, NearDedupAction::LinkToCanonical { .. }),
            "expected near-dup detection");
    }

    #[test]
    fn index_does_not_match_dissimilar_sequence() {
        let mut idx = NearDedupIndex::new(0.85);
        let canonical: Vec<Token> = (0u32..100).collect();
        idx.insert(hash("canonical"), &canonical);

        // Completely different sequence
        let different: Vec<Token> = (1000u32..1100).collect();
        let action = idx.process(&different);
        assert!(matches!(action, NearDedupAction::StoreAsNew),
            "should not match a dissimilar sequence");
    }

    #[test]
    fn index_detects_interpolated_variant() {
        // Simulate a system prompt with an interpolated user name:
        // canonical: "Hello Alice please ..."
        // variant:   "Hello Bob please ..."  (one token swapped)
        let mut idx = NearDedupIndex::new(0.80);

        let canonical: Vec<Token> = {
            let mut v: Vec<Token> = (1u32..=80).collect();
            v[5] = 42; // "Alice" token
            v
        };
        idx.insert(hash("sys-alice"), &canonical);

        let variant: Vec<Token> = {
            let mut v = canonical.clone();
            v[5] = 43; // "Bob" token
            v
        };

        match idx.process(&variant) {
            NearDedupAction::LinkToCanonical { similarity, delta, .. } => {
                assert!(similarity > 0.90, "similarity should be very high: {similarity}");
                // Delta should be tiny: one delete + one insert
                let insert_tokens: usize = delta.operations.iter()
                    .filter_map(|op| if let DeltaOp::Insert { tokens } = op { Some(tokens.len()) } else { None })
                    .sum();
                assert_eq!(insert_tokens, 1, "only one token should differ");
            }
            NearDedupAction::StoreAsNew => panic!("should have detected near-dup"),
        }
    }

    #[test]
    fn delta_savings_are_significant() {
        let canonical: Vec<Token> = (0u32..200).collect();
        let mut variant = canonical.clone();
        // Change 4 tokens out of 200 (2%)
        variant[10] = 9000; variant[50] = 9001; variant[100] = 9002; variant[150] = 9003;

        let delta = compute_delta(&canonical, &variant, hash("c"));
        let full_bytes = (variant.len() * 4) as u32;
        assert!(
            delta.delta_bytes < full_bytes / 4,
            "delta ({} bytes) should be much smaller than full ({} bytes)",
            delta.delta_bytes, full_bytes
        );
    }

    #[test]
    fn lsh_probability_high_for_similar_sequences() {
        // Statistical test: for 200 pairs with ~95% bigram-Jaccard similarity,
        // LSH (BANDS=16, ROWS=8) should detect > 95% of them.
        //
        // Setup: 150-token sequences with 2 non-adjacent changes.
        // Bigram Jaccard ≈ (149-4)/(149+4) ≈ 0.948 → P(detect) > 0.9999 per pair.
        let mut detected = 0u32;
        let trials = 200u32;

        for seed in 0..trials {
            let canonical: Vec<Token> = (0u32..150)
                .map(|i| (i * 1009 + seed * 7) % 50_000)
                .collect();

            // Change 2 tokens at well-separated positions
            let mut variant = canonical.clone();
            let pos1 = ((seed * 13) % 60) as usize;
            let pos2 = (pos1 + 70).min(149);
            variant[pos1] = variant[pos1].wrapping_add(50_001);
            variant[pos2] = variant[pos2].wrapping_add(50_001);

            let sig_a = compute_signature(&canonical);
            let sig_b = compute_signature(&variant);

            let collides = (0..BANDS).any(|band| {
                band_hash(&sig_a, band) == band_hash(&sig_b, band)
            });
            if collides { detected += 1; }
        }

        let detection_rate = detected as f64 / trials as f64;
        assert!(
            detection_rate > 0.95,
            "LSH detection rate {:.1}% below expected >95% for high-similarity sequences",
            detection_rate * 100.0
        );
    }
}
