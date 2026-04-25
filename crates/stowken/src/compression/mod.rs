//! Token compression: varint encoding + zstd dictionary compression.

pub mod dictionary;
pub mod varint;

use crate::types::Token;
use dictionary::{CompressionError, DictionaryCompressor};

/// Full compression/decompression pipeline: varint encode → zstd compress.
#[derive(Clone)]
pub struct CompressionPipeline {
    compressor: DictionaryCompressor,
    enabled: bool,
}

impl CompressionPipeline {
    pub fn new(enabled: bool, compression_level: i32) -> Self {
        Self {
            compressor: DictionaryCompressor::new(compression_level),
            enabled,
        }
    }

    pub fn with_dictionary(dict_bytes: Vec<u8>, enabled: bool, compression_level: i32) -> Self {
        Self {
            compressor: DictionaryCompressor::with_dictionary(dict_bytes, compression_level),
            enabled,
        }
    }

    /// Compress a token sequence.
    pub fn compress(&self, tokens: &[Token]) -> Result<Vec<u8>, CompressionError> {
        let varint_bytes = varint::encode_tokens(tokens);
        if self.enabled {
            self.compressor.compress(&varint_bytes)
        } else {
            Ok(varint_bytes)
        }
    }

    /// Decompress bytes back to a token sequence.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<Token>, CompressionError> {
        let varint_bytes = if self.enabled {
            self.compressor.decompress(data)?
        } else {
            data.to_vec()
        };
        varint::decode_tokens(&varint_bytes)
            .map_err(|e| CompressionError::Decompress(e.to_string()))
    }
}
