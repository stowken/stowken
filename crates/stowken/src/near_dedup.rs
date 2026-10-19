//! Near-duplicate detection via MinHash + LSH, plus a compact streaming
//! delta encoding for storing variants against a canonical segment.
//!
//! # Two-layer design
//!
//! Exact dedup catches byte-identical segments. Near-dedup catches segments
//! that are *almost* identical:
//!
//! - System prompts with an interpolated user name
//! - RAG contexts with one different chunk
//! - Tool call results that differ only in formatting
//!
//! The price of catching them is computing one MinHash signature per
//! incoming segment that misses exact dedup, plus a few SQLite probes.
//!
//! # Algorithm
//!
//! 1. Each token sequence becomes a 128-element [`MinHashSignature`] using
//!    token-level bigram shingles and 128 independent hash functions.
//! 2. The signature is split into 16 LSH bands of 8 rows. Two segments are
//!    *candidates* if any band hash collides. With BANDS=16 and ROWS=8 the
//!    detection probability for Jaccard ≥ 0.85 exceeds 99%.
//! 3. The candidate's full signature is loaded from the index; Jaccard is
//!    estimated by counting matching MinHash values.
//! 4. If similarity exceeds the threshold, the variant is stored as a
//!    [`Delta`] against the best-matching canonical instead of a full
//!    token array. A separate, stricter threshold can additionally alias the
//!    variant's *embedding* to the canonical's (see `embed_reuse_threshold`).
//!
//! This module is the algorithm only — signatures, MinHash, LSH bands, and
//! delta encoding. Persistence of the index (signatures + band buckets +
//! embedding aliases) lives in [`crate::near_dedup_index::NearDupIndex`],
//! a self-contained set of append-only logs under the vault root — not in
//! SQLite. Token sequences are never cached in memory; when computing a
//! delta the canonical is fetched from the backend on demand.

use std::io::Read;

use thiserror::Error;

use crate::types::{SegmentHash, Token};

// ── Tuning constants ──────────────────────────────────────────────────────────

/// Token-level shingle (n-gram) size.
pub const NGRAM_SIZE: usize = 2;
/// Total number of MinHash functions in the signature.
pub const NUM_HASHES: usize = 128;
/// Number of LSH bands.
pub const BANDS: usize = 16;
/// Hash functions per band. `BANDS * ROWS_PER_BAND == NUM_HASHES`.
pub const ROWS_PER_BAND: usize = NUM_HASHES / BANDS;

/// Default Jaccard threshold for declaring a near-duplicate. Conservative
/// — catches the common cases (1–10 token edits in 100-token segments)
/// while keeping false-positive risk negligible.
pub const DEFAULT_THRESHOLD: f64 = 0.85;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum NearDedupError {
    #[error("delta encoding failed: {0}")]
    Encode(String),
    #[error("delta decoding failed: {0}")]
    Decode(String),
    #[error("invalid signature size: expected {expected}, got {got}")]
    InvalidSignatureSize { expected: usize, got: usize },
}

// ── MinHash signature ─────────────────────────────────────────────────────────

/// 128-element MinHash signature. Compact (1024 bytes) and trivially
/// serializable for SQLite storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashSignature {
    pub values: [u64; NUM_HASHES],
}

impl MinHashSignature {
    /// Compute the signature of a token sequence.
    pub fn compute(tokens: &[Token]) -> Self {
        let mut values = [u64::MAX; NUM_HASHES];

        if tokens.len() < NGRAM_SIZE {
            // Short sequence: treat the whole thing as one shingle.
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

        Self { values }
    }

    /// Estimate Jaccard similarity against another signature in [0.0, 1.0].
    pub fn jaccard(&self, other: &Self) -> f64 {
        let matches = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / NUM_HASHES as f64
    }

    /// LSH band hash for a given band index.
    pub fn band_hash(&self, band: usize) -> u64 {
        let start = band * ROWS_PER_BAND;
        let mut h: u64 = (band as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        for &v in &self.values[start..start + ROWS_PER_BAND] {
            h = h
                .wrapping_add(v)
                .rotate_left(17)
                .wrapping_mul(0x6c62_272e_07bb_0142);
        }
        h
    }

    /// Encode as 1024 bytes (128 × u64 little-endian) for SQLite BLOB storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(NUM_HASHES * 8);
        for v in &self.values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NearDedupError> {
        let expected = NUM_HASHES * 8;
        if bytes.len() != expected {
            return Err(NearDedupError::InvalidSignatureSize {
                expected,
                got: bytes.len(),
            });
        }
        let mut values = [0u64; NUM_HASHES];
        for (i, val) in values.iter_mut().enumerate() {
            let off = i * 8;
            *val = u64::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
        }
        Ok(Self { values })
    }
}

fn hash_ngram(ngram: &[Token], seed: u64) -> u64 {
    let mut h = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    for &tok in ngram {
        h ^= u64::from(tok).wrapping_mul(0x6c62_272e_07bb_0142);
        h = h.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    h ^ (h >> 31)
}

// ── Delta encoding ────────────────────────────────────────────────────────────

/// Streaming delta op. Reconstructs the variant by walking a single cursor
/// through the canonical token sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaOp {
    /// Advance the canonical cursor by `n` (don't emit anything).
    Skip(u32),
    /// Emit `n` tokens from the canonical at the cursor; advance cursor.
    Copy(u32),
    /// Emit literal tokens (canonical cursor unchanged).
    Insert(Vec<Token>),
}

const OP_SKIP: u8 = 0x01;
const OP_COPY: u8 = 0x02;
const OP_INSERT: u8 = 0x03;

/// Encode a delta as a compact byte stream:
///
/// ```text
/// [op_tag][varint length][optional varint × length tokens]
/// ```
///
/// The Skip / Copy / Insert opcodes match the streaming model — there's no
/// explicit cursor in the encoding because reconstruction walks the
/// canonical left-to-right, advancing on Skip and Copy.
pub fn encode_delta(ops: &[DeltaOp]) -> Vec<u8> {
    let mut out = Vec::new();
    for op in ops {
        match op {
            DeltaOp::Skip(n) => {
                out.push(OP_SKIP);
                write_varint_u32(*n, &mut out);
            }
            DeltaOp::Copy(n) => {
                out.push(OP_COPY);
                write_varint_u32(*n, &mut out);
            }
            DeltaOp::Insert(tokens) => {
                out.push(OP_INSERT);
                write_varint_u32(tokens.len() as u32, &mut out);
                for t in tokens {
                    write_varint_u32(*t, &mut out);
                }
            }
        }
    }
    out
}

pub fn decode_delta(bytes: &[u8]) -> Result<Vec<DeltaOp>, NearDedupError> {
    let mut ops = Vec::new();
    let mut cursor = std::io::Cursor::new(bytes);
    while (cursor.position() as usize) < bytes.len() {
        let mut tag = [0u8; 1];
        cursor
            .read_exact(&mut tag)
            .map_err(|e| NearDedupError::Decode(e.to_string()))?;
        match tag[0] {
            OP_SKIP => {
                let n = read_varint_u32(&mut cursor)?;
                ops.push(DeltaOp::Skip(n));
            }
            OP_COPY => {
                let n = read_varint_u32(&mut cursor)?;
                ops.push(DeltaOp::Copy(n));
            }
            OP_INSERT => {
                let n = read_varint_u32(&mut cursor)?;
                let mut tokens = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    tokens.push(read_varint_u32(&mut cursor)?);
                }
                ops.push(DeltaOp::Insert(tokens));
            }
            other => {
                return Err(NearDedupError::Decode(format!(
                    "unknown delta op tag 0x{other:02x}"
                )))
            }
        }
    }
    Ok(ops)
}

/// Apply a delta to a canonical token sequence to recover the variant.
pub fn apply_delta(canonical: &[Token], ops: &[DeltaOp]) -> Result<Vec<Token>, NearDedupError> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for op in ops {
        match op {
            DeltaOp::Skip(n) => {
                let next = cursor + *n as usize;
                if next > canonical.len() {
                    return Err(NearDedupError::Decode(format!(
                        "delta Skip overflows canonical: cursor {} + {} > {}",
                        cursor,
                        n,
                        canonical.len()
                    )));
                }
                cursor = next;
            }
            DeltaOp::Copy(n) => {
                let next = cursor + *n as usize;
                if next > canonical.len() {
                    return Err(NearDedupError::Decode(format!(
                        "delta Copy overflows canonical: cursor {} + {} > {}",
                        cursor,
                        n,
                        canonical.len()
                    )));
                }
                out.extend_from_slice(&canonical[cursor..next]);
                cursor = next;
            }
            DeltaOp::Insert(tokens) => out.extend_from_slice(tokens),
        }
    }
    Ok(out)
}

/// Compute a token-level diff that transforms `canonical` into `variant`.
///
/// Greedy LCS: scan for the longest common prefix from the current
/// position; if zero, emit a single Skip+Insert step and try again. Not
/// optimal, but produces correct deltas in O(canonical * variant) worst
/// case and tight ones in the common "few-token-changes" case.
pub fn compute_delta(canonical: &[Token], variant: &[Token]) -> Vec<DeltaOp> {
    let mut ops: Vec<DeltaOp> = Vec::new();
    let mut i = 0usize; // cursor in canonical
    let mut j = 0usize; // cursor in variant

    while i < canonical.len() || j < variant.len() {
        // Longest common prefix from current positions.
        let mut run = 0usize;
        while i + run < canonical.len()
            && j + run < variant.len()
            && canonical[i + run] == variant[j + run]
        {
            run += 1;
        }

        if run > 0 {
            push_or_extend_copy(&mut ops, run as u32);
            i += run;
            j += run;
            continue;
        }

        // No common prefix — emit one Skip and one Insert step. Look-ahead
        // by 1 in each so we make progress.
        if i < canonical.len() && j < variant.len() {
            push_or_extend_skip(&mut ops, 1);
            push_or_extend_insert(&mut ops, vec![variant[j]]);
            i += 1;
            j += 1;
        } else if i < canonical.len() {
            push_or_extend_skip(&mut ops, (canonical.len() - i) as u32);
            i = canonical.len();
        } else {
            push_or_extend_insert(&mut ops, variant[j..].to_vec());
            j = variant.len();
        }
    }

    ops
}

fn push_or_extend_copy(ops: &mut Vec<DeltaOp>, n: u32) {
    if let Some(DeltaOp::Copy(prev)) = ops.last_mut() {
        *prev += n;
    } else {
        ops.push(DeltaOp::Copy(n));
    }
}

fn push_or_extend_skip(ops: &mut Vec<DeltaOp>, n: u32) {
    if let Some(DeltaOp::Skip(prev)) = ops.last_mut() {
        *prev += n;
    } else {
        ops.push(DeltaOp::Skip(n));
    }
}

fn push_or_extend_insert(ops: &mut Vec<DeltaOp>, tokens: Vec<Token>) {
    if let Some(DeltaOp::Insert(prev)) = ops.last_mut() {
        prev.extend(tokens);
    } else {
        ops.push(DeltaOp::Insert(tokens));
    }
}

// ── Hash conversion helpers ───────────────────────────────────────────────────

/// Decode a hex segment hash to its raw 32 bytes for compact frame embedding.
pub fn segment_hash_to_bytes(hash: &SegmentHash) -> Result<[u8; 32], NearDedupError> {
    let raw =
        hex::decode(&hash.0).map_err(|e| NearDedupError::Encode(format!("hex decode: {e}")))?;
    if raw.len() != 32 {
        return Err(NearDedupError::Encode(format!(
            "expected 32-byte hash, got {}",
            raw.len()
        )));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// Encode 32 raw bytes back to a hex SegmentHash string.
pub fn bytes_to_segment_hash(bytes: &[u8; 32]) -> SegmentHash {
    SegmentHash(hex::encode(bytes))
}

// ── Varint helpers (reused from compression::varint but inlined here so
//   the delta encoding has no external coupling) ─────────────────────────────

fn write_varint_u32(mut value: u32, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint_u32(cursor: &mut std::io::Cursor<&[u8]>) -> Result<u32, NearDedupError> {
    let mut shift: u32 = 0;
    let mut result: u32 = 0;
    loop {
        let mut byte = [0u8; 1];
        cursor
            .read_exact(&mut byte)
            .map_err(|e| NearDedupError::Decode(format!("varint truncated: {e}")))?;
        let b = byte[0];
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return Err(NearDedupError::Decode("varint overflows u32".into()));
        }
    }
    Ok(result)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(s: &str) -> SegmentHash {
        // Build a 64-char hex string padded with zeros for tests.
        SegmentHash(format!("{:0>64}", s))
    }

    #[test]
    fn signature_byte_roundtrip() {
        let tokens: Vec<Token> = (0u32..200).collect();
        let sig = MinHashSignature::compute(&tokens);
        let bytes = sig.to_bytes();
        assert_eq!(bytes.len(), 1024);
        let recovered = MinHashSignature::from_bytes(&bytes).unwrap();
        assert_eq!(sig, recovered);
    }

    #[test]
    fn signature_invalid_size_errors() {
        assert!(matches!(
            MinHashSignature::from_bytes(&[0u8; 1023]),
            Err(NearDedupError::InvalidSignatureSize { .. })
        ));
    }

    #[test]
    fn jaccard_identical_is_one() {
        let tokens: Vec<Token> = (0..50).collect();
        let a = MinHashSignature::compute(&tokens);
        let b = MinHashSignature::compute(&tokens);
        assert!((a.jaccard(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint_is_low() {
        let a: Vec<Token> = (0..50).collect();
        let b: Vec<Token> = (1000..1050).collect();
        let sim = MinHashSignature::compute(&a).jaccard(&MinHashSignature::compute(&b));
        assert!(sim < 0.5, "got {sim}");
    }

    #[test]
    fn delta_roundtrip_simple() {
        let canonical: Vec<Token> = vec![1, 2, 3, 4, 5];
        let variant: Vec<Token> = vec![1, 2, 99, 4, 5, 6];
        let ops = compute_delta(&canonical, &variant);
        let recovered = apply_delta(&canonical, &ops).unwrap();
        assert_eq!(recovered, variant);
    }

    #[test]
    fn delta_identical_is_pure_copy() {
        let tokens: Vec<Token> = (0..100).collect();
        let ops = compute_delta(&tokens, &tokens);
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], DeltaOp::Copy(100)));
    }

    #[test]
    fn delta_byte_roundtrip() {
        let canonical: Vec<Token> = (0u32..200).collect();
        let mut variant = canonical.clone();
        variant[10] = 99_999;
        variant[150] = 42_000;
        let ops = compute_delta(&canonical, &variant);
        let bytes = encode_delta(&ops);
        let decoded = decode_delta(&bytes).unwrap();
        assert_eq!(ops, decoded);
        let recovered = apply_delta(&canonical, &decoded).unwrap();
        assert_eq!(recovered, variant);
    }

    #[test]
    fn delta_savings_are_significant() {
        let canonical: Vec<Token> = (0u32..200).collect();
        let mut variant = canonical.clone();
        variant[10] = 9000;
        variant[100] = 9001;
        let ops = compute_delta(&canonical, &variant);
        let bytes = encode_delta(&ops);
        let full = canonical.len() * 4;
        assert!(
            bytes.len() < full / 4,
            "delta {} bytes vs full {} bytes",
            bytes.len(),
            full
        );
    }

    #[test]
    fn apply_delta_overflow_errors() {
        let canonical: Vec<Token> = vec![1, 2, 3];
        let bad_ops = vec![DeltaOp::Copy(10)];
        assert!(matches!(
            apply_delta(&canonical, &bad_ops),
            Err(NearDedupError::Decode(_))
        ));
    }

    #[test]
    fn band_hash_collision_for_similar_sequences() {
        // 150 tokens with 2 changes; bigram-Jaccard ≈ 0.95 → at least one
        // band should collide (P > 99.99%).
        let canonical: Vec<Token> = (0u32..150).collect();
        let mut variant = canonical.clone();
        variant[20] = 99_999;
        variant[80] = 88_888;

        let sig_a = MinHashSignature::compute(&canonical);
        let sig_b = MinHashSignature::compute(&variant);
        let any_collision = (0..BANDS).any(|b| sig_a.band_hash(b) == sig_b.band_hash(b));
        assert!(
            any_collision,
            "expected at least one band collision for high-similarity sequences"
        );
    }

    #[test]
    fn segment_hash_byte_roundtrip() {
        let h = hash("deadbeef");
        let bytes = segment_hash_to_bytes(&h).unwrap();
        let recovered = bytes_to_segment_hash(&bytes);
        assert_eq!(h, recovered);
    }
}
