//! zstd dictionary compression wrapper.
//!
//! Applies zstd compression on top of varint-encoded token bytes, optionally
//! using a pre-trained dictionary for improved ratios on token-like data.

use thiserror::Error;

/// Errors from compression / decompression.
#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("compression failed: {0}")]
    Compress(String),
    #[error("decompression failed: {0}")]
    Decompress(String),
    #[error("dictionary training failed: {0}")]
    Training(String),
}

/// Wraps a zstd encoder/decoder, optionally with a pre-trained dictionary.
#[derive(Clone)]
pub struct DictionaryCompressor {
    /// Raw dictionary bytes; `None` when no dictionary is loaded.
    dictionary_bytes: Option<Vec<u8>>,
    compression_level: i32,
}

impl DictionaryCompressor {
    /// Create a compressor without a dictionary.
    pub fn new(compression_level: i32) -> Self {
        Self {
            dictionary_bytes: None,
            compression_level,
        }
    }

    /// Create a compressor with a pre-trained zstd dictionary.
    pub fn with_dictionary(dict_bytes: Vec<u8>, compression_level: i32) -> Self {
        Self {
            dictionary_bytes: Some(dict_bytes),
            compression_level,
        }
    }

    /// Load a dictionary from a file path.
    pub fn from_dict_path(path: &str, compression_level: i32) -> Result<Self, CompressionError> {
        let bytes = std::fs::read(path)
            .map_err(|e| CompressionError::Compress(format!("failed to read dict: {e}")))?;
        Ok(Self::with_dictionary(bytes, compression_level))
    }

    /// Compress `data` using zstd (with optional dictionary).
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if let Some(dict) = &self.dictionary_bytes {
            let mut compressor = zstd::bulk::Compressor::with_dictionary(self.compression_level, dict)
                .map_err(|e| CompressionError::Compress(e.to_string()))?;
            compressor.compress(data)
                .map_err(|e| CompressionError::Compress(e.to_string()))
        } else {
            zstd::bulk::compress(data, self.compression_level)
                .map_err(|e| CompressionError::Compress(e.to_string()))
        }
    }

    /// Decompress `data` (with optional dictionary).
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if let Some(dict) = &self.dictionary_bytes {
            let mut decompressor = zstd::bulk::Decompressor::with_dictionary(dict)
                .map_err(|e| CompressionError::Decompress(e.to_string()))?;
            decompressor.decompress(data, 256 * 1024 * 1024)
                .map_err(|e| CompressionError::Decompress(e.to_string()))
        } else {
            zstd::stream::decode_all(data)
                .map_err(|e| CompressionError::Decompress(e.to_string()))
        }
    }

    /// Train a zstd dictionary from sample data.
    ///
    /// `samples` should be varint-encoded token sequences from a representative corpus.
    /// `dict_size` is the target dictionary size in bytes (default: 112 640 = 110 KiB).
    pub fn train_dictionary(samples: &[Vec<u8>], dict_size: usize) -> Result<Vec<u8>, CompressionError> {
        let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
        zstd::dict::from_samples(&refs, dict_size)
            .map_err(|e| CompressionError::Training(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::varint;

    fn sample_varint_data() -> Vec<u8> {
        let tokens: Vec<u32> = (0u32..1000).map(|i| i % 50_000).collect();
        varint::encode_tokens(&tokens)
    }

    #[test]
    fn roundtrip_without_dictionary() {
        let compressor = DictionaryCompressor::new(3);
        let data = sample_varint_data();
        let compressed = compressor.compress(&data).expect("compress");
        let decompressed = compressor.decompress(&compressed).expect("decompress");
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_with_trained_dictionary() {
        // Build sample corpus
        let samples: Vec<Vec<u8>> = (0u32..50)
            .map(|seed| {
                let tokens: Vec<u32> = (0..200).map(|i| (i + seed) % 30_000).collect();
                varint::encode_tokens(&tokens)
            })
            .collect();

        let dict = DictionaryCompressor::train_dictionary(&samples, 4096).expect("train");
        let compressor = DictionaryCompressor::with_dictionary(dict, 3);

        let data = sample_varint_data();
        let compressed = compressor.compress(&data).expect("compress");
        let decompressed = compressor.decompress(&compressed).expect("decompress");
        assert_eq!(data, decompressed);
    }

    #[test]
    fn compression_actually_reduces_size() {
        let compressor = DictionaryCompressor::new(3);
        // Highly compressible: all the same token repeated
        let tokens: Vec<u32> = vec![42; 1000];
        let data = varint::encode_tokens(&tokens);
        let compressed = compressor.compress(&data).expect("compress");
        assert!(compressed.len() < data.len(), "expected compression to reduce size");
    }

    #[test]
    fn empty_roundtrip() {
        let compressor = DictionaryCompressor::new(3);
        let compressed = compressor.compress(&[]).expect("compress empty");
        let decompressed = compressor.decompress(&compressed).expect("decompress empty");
        assert!(decompressed.is_empty());
    }
}
