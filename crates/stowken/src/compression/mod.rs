//! Token compression: varint encoding + zstd compression.
//!
//! # Frame format
//!
//! Every compressed segment starts with a 1-byte version tag identifying the
//! codec used. Decompression dispatches on this byte so future codecs can be
//! introduced without breaking previously stored data.
//!
//! | Byte | Codec                         |
//! |-----:|-------------------------------|
//! | `0x01` | `varint`                    |
//! | `0x02` | `varint + zstd`             |
//!
//! The version byte is followed by the codec-specific payload. New codecs
//! (e.g. dictionary-tagged zstd, bit-packed encodings) get a new byte; old
//! frames remain readable indefinitely.

pub mod varint;

use crate::types::Token;
use thiserror::Error;

/// Format version tag at the head of every compressed frame.
///
/// All values must round-trip through [`decompress`] regardless of which
/// version tag the producer wrote.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameVersion {
    /// `varint` only (no zstd).
    VarintOnly = 0x01,
    /// `varint` then zstd.
    VarintZstd = 0x02,
}

impl FrameVersion {
    fn from_byte(byte: u8) -> Result<Self, CompressionError> {
        match byte {
            0x01 => Ok(Self::VarintOnly),
            0x02 => Ok(Self::VarintZstd),
            other => Err(CompressionError::UnsupportedVersion(other)),
        }
    }
}

/// Errors from compression / decompression.
#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("compression failed: {0}")]
    Compress(String),
    #[error("decompression failed: {0}")]
    Decompress(String),
    #[error("empty frame (no version byte)")]
    EmptyFrame,
    #[error("unsupported frame version: 0x{0:02x}")]
    UnsupportedVersion(u8),
}

/// Full compression/decompression pipeline: varint encode → zstd compress.
#[derive(Clone)]
pub struct CompressionPipeline {
    enabled: bool,
    level: i32,
}

impl CompressionPipeline {
    pub fn new(enabled: bool, compression_level: i32) -> Self {
        Self { enabled, level: compression_level }
    }

    /// Compress a token sequence into a versioned frame.
    pub fn compress(&self, tokens: &[Token]) -> Result<Vec<u8>, CompressionError> {
        let varint_bytes = varint::encode_tokens(tokens);
        if self.enabled {
            let zstd_bytes = zstd::bulk::compress(&varint_bytes, self.level)
                .map_err(|e| CompressionError::Compress(e.to_string()))?;
            let mut out = Vec::with_capacity(zstd_bytes.len() + 1);
            out.push(FrameVersion::VarintZstd as u8);
            out.extend_from_slice(&zstd_bytes);
            Ok(out)
        } else {
            let mut out = Vec::with_capacity(varint_bytes.len() + 1);
            out.push(FrameVersion::VarintOnly as u8);
            out.extend_from_slice(&varint_bytes);
            Ok(out)
        }
    }

    /// Decompress a versioned frame back to a token sequence.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<Token>, CompressionError> {
        let (version_byte, payload) = data.split_first().ok_or(CompressionError::EmptyFrame)?;
        let version = FrameVersion::from_byte(*version_byte)?;

        let varint_bytes = match version {
            FrameVersion::VarintOnly => payload.to_vec(),
            FrameVersion::VarintZstd => zstd::stream::decode_all(payload)
                .map_err(|e| CompressionError::Decompress(e.to_string()))?,
        };
        varint::decode_tokens(&varint_bytes)
            .map_err(|e| CompressionError::Decompress(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_zstd() {
        let pipeline = CompressionPipeline::new(true, 3);
        let tokens: Vec<Token> = (0u32..1000).collect();
        let frame = pipeline.compress(&tokens).unwrap();
        assert_eq!(frame[0], FrameVersion::VarintZstd as u8);
        assert_eq!(pipeline.decompress(&frame).unwrap(), tokens);
    }

    #[test]
    fn roundtrip_without_zstd() {
        let pipeline = CompressionPipeline::new(false, 3);
        let tokens: Vec<Token> = (0u32..1000).collect();
        let frame = pipeline.compress(&tokens).unwrap();
        assert_eq!(frame[0], FrameVersion::VarintOnly as u8);
        assert_eq!(pipeline.decompress(&frame).unwrap(), tokens);
    }

    #[test]
    fn cross_pipeline_decompresses_old_frames() {
        // A frame produced with compression OFF must decompress on a pipeline
        // with compression ON, and vice versa — version byte selects the codec.
        let zstd_pipeline = CompressionPipeline::new(true, 3);
        let plain_pipeline = CompressionPipeline::new(false, 3);
        let tokens: Vec<Token> = (0u32..200).collect();

        let zstd_frame = zstd_pipeline.compress(&tokens).unwrap();
        let plain_frame = plain_pipeline.compress(&tokens).unwrap();

        assert_eq!(plain_pipeline.decompress(&zstd_frame).unwrap(), tokens);
        assert_eq!(zstd_pipeline.decompress(&plain_frame).unwrap(), tokens);
    }

    #[test]
    fn empty_frame_errors() {
        let pipeline = CompressionPipeline::new(true, 3);
        assert!(matches!(pipeline.decompress(&[]), Err(CompressionError::EmptyFrame)));
    }

    #[test]
    fn unknown_version_errors() {
        let pipeline = CompressionPipeline::new(true, 3);
        let bad = [0xFFu8, 0, 1, 2, 3];
        assert!(matches!(
            pipeline.decompress(&bad),
            Err(CompressionError::UnsupportedVersion(0xFF))
        ));
    }
}
