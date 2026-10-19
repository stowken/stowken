//! Variable-width integer encoding for token IDs.
//!
//! Encoding scheme optimised for the Zipf distribution of token values:
//!
//! | Token value              | Bytes | First-byte pattern              |
//! |--------------------------|-------|---------------------------------|
//! | 0 – 127                  | 1     | `0xxxxxxx`                      |
//! | 128 – 16 383             | 2     | `10xxxxxx xxxxxxxx`             |
//! | 16 384 – 2 097 151       | 3     | `110xxxxx xxxxxxxx xxxxxxxx`    |
//! | 2 097 152 – 536 870 911  | 4     | `111xxxxx xxxxxxxx xxxxxxxx xxxxxxxx` |
//! | 536 870 912 – u32::MAX   | 5     | `11111111 XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX` (raw u32 BE) |
//!
//! The 5-byte escape (`0xFF` prefix) handles values beyond the 29-bit range of
//! the 4-byte scheme. All real tokenizer vocabularies fit well within the
//! 4-byte range (~512M), but the encoding must be lossless for any `u32`.

use crate::types::Token;
use thiserror::Error;

/// Errors produced during varint decoding.
#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("truncated varint stream at byte {0}")]
    Truncated(usize),
    #[error("decoded token value {0} exceeds u32::MAX")]
    Overflow(u64),
}

/// Encode a slice of token IDs into a compact byte stream.
pub fn encode_tokens(tokens: &[Token]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for &tok in tokens {
        encode_one(tok, &mut out);
    }
    out
}

/// Decode a byte stream produced by `encode_tokens` back to token IDs.
pub fn decode_tokens(data: &[u8]) -> Result<Vec<Token>, DecodeError> {
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let (tok, consumed) = decode_one(&data[i..], i)?;
        tokens.push(tok);
        i += consumed;
    }
    Ok(tokens)
}

/// The maximum value that fits in the 4-byte encoding (29 payload bits).
const MAX_4BYTE: u32 = (1 << 29) - 1; // 536_870_911

/// Write a single token's varint encoding into `out`.
fn encode_one(tok: Token, out: &mut Vec<u8>) {
    if tok < 128 {
        // 1 byte: 0xxxxxxx
        out.push(tok as u8);
    } else if tok < 16_384 {
        // 2 bytes: 10xxxxxx xxxxxxxx
        let high = 0b1000_0000u8 | ((tok >> 8) as u8 & 0x3F);
        let low = (tok & 0xFF) as u8;
        out.push(high);
        out.push(low);
    } else if tok < 2_097_152 {
        // 3 bytes: 110xxxxx xxxxxxxx xxxxxxxx
        let high = 0b1100_0000u8 | ((tok >> 16) as u8 & 0x1F);
        let mid = ((tok >> 8) & 0xFF) as u8;
        let low = (tok & 0xFF) as u8;
        out.push(high);
        out.push(mid);
        out.push(low);
    } else if tok <= MAX_4BYTE {
        // 4 bytes: 111xxxxx xxxxxxxx xxxxxxxx xxxxxxxx  (29 payload bits)
        let high = 0b1110_0000u8 | ((tok >> 24) as u8 & 0x1F);
        let b2 = ((tok >> 16) & 0xFF) as u8;
        let b3 = ((tok >> 8) & 0xFF) as u8;
        let b4 = (tok & 0xFF) as u8;
        out.push(high);
        out.push(b2);
        out.push(b3);
        out.push(b4);
    } else {
        // 5-byte escape for values > 536_870_911:
        //   0xFF  followed by raw big-endian u32
        out.push(0xFF);
        out.extend_from_slice(&tok.to_be_bytes());
    }
}

/// Decode one varint from `data`, returning `(token, bytes_consumed)`.
fn decode_one(data: &[u8], base_offset: usize) -> Result<(Token, usize), DecodeError> {
    if data.is_empty() {
        return Err(DecodeError::Truncated(base_offset));
    }
    let first = data[0];
    if first & 0b1000_0000 == 0 {
        // 1-byte: 0xxxxxxx
        Ok((u32::from(first), 1))
    } else if first & 0b0100_0000 == 0 {
        // 2-byte: 10xxxxxx xxxxxxxx
        if data.len() < 2 {
            return Err(DecodeError::Truncated(base_offset));
        }
        let val = ((u32::from(first & 0x3F)) << 8) | u32::from(data[1]);
        Ok((val, 2))
    } else if first & 0b0010_0000 == 0 {
        // 3-byte: 110xxxxx xxxxxxxx xxxxxxxx
        if data.len() < 3 {
            return Err(DecodeError::Truncated(base_offset));
        }
        let val =
            ((u32::from(first & 0x1F)) << 16) | (u32::from(data[1]) << 8) | u32::from(data[2]);
        Ok((val, 3))
    } else if first == 0xFF {
        // 5-byte escape: raw big-endian u32
        if data.len() < 5 {
            return Err(DecodeError::Truncated(base_offset));
        }
        let val = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        Ok((val, 5))
    } else {
        // 4-byte: 111xxxxx xxxxxxxx xxxxxxxx xxxxxxxx  (29 payload bits)
        if data.len() < 4 {
            return Err(DecodeError::Truncated(base_offset));
        }
        let val = ((u32::from(first & 0x1F)) << 24)
            | (u32::from(data[1]) << 16)
            | (u32::from(data[2]) << 8)
            | u32::from(data[3]);
        Ok((val, 4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(tokens: &[Token]) {
        let encoded = encode_tokens(tokens);
        let decoded = decode_tokens(&encoded).expect("decode failed");
        assert_eq!(
            tokens,
            decoded.as_slice(),
            "roundtrip mismatch for {tokens:?}"
        );
    }

    #[test]
    fn boundary_values() {
        roundtrip(&[0]);
        roundtrip(&[1]);
        roundtrip(&[127]);
        roundtrip(&[128]);
        roundtrip(&[16_383]);
        roundtrip(&[16_384]);
        roundtrip(&[2_097_151]);
        roundtrip(&[2_097_152]);
        roundtrip(&[u32::MAX]);
    }

    #[test]
    fn empty_sequence() {
        roundtrip(&[]);
    }

    #[test]
    fn mixed_sequence() {
        let tokens: Vec<Token> = vec![0, 64, 127, 128, 1000, 16_383, 16_384, 100_000, u32::MAX];
        roundtrip(&tokens);
    }

    #[test]
    fn long_sequence() {
        let tokens: Vec<Token> = (0..10_000).map(|i| i % 150_000).collect();
        roundtrip(&tokens);
    }

    #[test]
    fn encoding_sizes() {
        // Verify expected byte widths
        assert_eq!(encode_tokens(&[0]).len(), 1);
        assert_eq!(encode_tokens(&[127]).len(), 1);
        assert_eq!(encode_tokens(&[128]).len(), 2);
        assert_eq!(encode_tokens(&[16_383]).len(), 2);
        assert_eq!(encode_tokens(&[16_384]).len(), 3);
        assert_eq!(encode_tokens(&[2_097_151]).len(), 3);
        assert_eq!(encode_tokens(&[2_097_152]).len(), 4);
        assert_eq!(encode_tokens(&[536_870_911]).len(), 4);
        // Values beyond the 4-byte range use the 5-byte escape
        assert_eq!(encode_tokens(&[536_870_912]).len(), 5);
        assert_eq!(encode_tokens(&[u32::MAX]).len(), 5);
    }

    #[test]
    fn truncated_stream_errors() {
        // A 2-byte encoding truncated to 1 byte
        let partial = vec![0b1000_0000u8]; // first byte of a 2-byte token, missing second
        assert!(matches!(
            decode_tokens(&partial),
            Err(DecodeError::Truncated(_))
        ));
    }

    #[test]
    fn high_freq_tokens_are_small() {
        // Top tokens (0–127) should encode in 1 byte each
        let top_tokens: Vec<Token> = (0u32..128).collect();
        let encoded = encode_tokens(&top_tokens);
        assert_eq!(encoded.len(), 128, "each token 0–127 should cost 1 byte");
    }
}
