//! Exact segment deduplication via SHA-256 content hashing.

use crate::types::{SegmentHash, Token};
use sha2::{Digest, Sha256};

/// Hash a token sequence to a `SegmentHash` (SHA-256, hex-encoded).
///
/// Tokens are hashed in little-endian byte order so the result is
/// independent of host endianness and stable across architectures.
pub fn hash_segment(tokens: &[Token]) -> SegmentHash {
    let mut hasher = Sha256::new();
    for &token in tokens {
        hasher.update(token.to_le_bytes());
    }
    SegmentHash(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_tokens_same_hash() {
        let a = hash_segment(&[1, 2, 3]);
        let b = hash_segment(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_tokens_different_hash() {
        let a = hash_segment(&[1, 2, 3]);
        let b = hash_segment(&[1, 2, 4]);
        assert_ne!(a, b);
    }

    #[test]
    fn order_matters() {
        let a = hash_segment(&[1, 2, 3]);
        let b = hash_segment(&[3, 2, 1]);
        assert_ne!(a, b);
    }

    #[test]
    fn empty_sequence_hashes_consistently() {
        let a = hash_segment(&[]);
        let b = hash_segment(&[]);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_is_hex_encoded_sha256() {
        let h = hash_segment(&[0]);
        assert_eq!(h.0.len(), 64);
        assert!(h.0.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
