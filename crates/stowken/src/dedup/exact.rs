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

/// Build a tenant-scoped hash by prefixing the raw hash with the tenant ID.
///
/// Two tenants with identical segments do NOT share storage, preventing
/// information leakage through hash probing.
pub fn hash_segment_tenant(tokens: &[Token], tenant_id: &str) -> SegmentHash {
    let mut hasher = Sha256::new();
    hasher.update(tenant_id.as_bytes());
    hasher.update(b":");
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
    fn tenant_isolation() {
        let tokens = &[1u32, 2, 3];
        let a = hash_segment_tenant(tokens, "tenant-a");
        let b = hash_segment_tenant(tokens, "tenant-b");
        let plain = hash_segment(tokens);
        assert_ne!(a, b, "different tenants must have different hashes");
        assert_ne!(a, plain, "tenant hash must differ from plain hash");
    }

    #[test]
    fn tenant_deterministic() {
        let tokens = &[42u32, 100, 999];
        let a = hash_segment_tenant(tokens, "my-tenant");
        let b = hash_segment_tenant(tokens, "my-tenant");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_is_hex_encoded_sha256() {
        let h = hash_segment(&[0]);
        // SHA-256 output is 32 bytes → 64 hex chars
        assert_eq!(h.0.len(), 64);
        assert!(h.0.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
