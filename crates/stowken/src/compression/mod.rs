//! Token compression: varint encoding + zstd (optionally with a dictionary).
//!
//! # Frame format
//!
//! Every compressed segment starts with a 1-byte version tag identifying the
//! codec used. Decompression dispatches on this byte so future codecs can be
//! introduced without breaking previously stored data.
//!
//! | Byte | Codec                             | Body                                               |
//! |-----:|-----------------------------------|----------------------------------------------------|
//! | 0x01 | varint                            | `varint(tokens)`                                   |
//! | 0x02 | varint + zstd                     | `zstd(varint(tokens))`                             |
//! | 0x03 | varint + zstd with dictionary     | `dict_id(u32 LE) || zstd(varint(tokens), dict)`    |
//! | 0x04 | delta against a canonical segment | `canonical_hash(32 bytes) || delta_ops`            |
//!
//! For 0x03, the 4-byte little-endian dictionary ID directly follows the
//! version byte. The decompressor looks the ID up in the [`DictRegistry`]
//! and decompresses with that dictionary; multiple dictionaries coexist so
//! rotation is non-destructive — old frames stay readable as long as their
//! dict file stays on disk.
//!
//! For 0x04, the 32-byte raw SHA-256 hash of the canonical segment follows
//! the version tag, then a streaming delta in the [`crate::near_dedup`]
//! op format. The decompressor must be configured with a [`CanonicalResolver`]
//! that knows how to load tokens for a given hash; this keeps
//! `CompressionPipeline` decoupled from the storage backend.

pub mod varint;

use std::sync::Arc;

use thiserror::Error;

use crate::dict_registry::{DictError, DictRegistry};
use crate::types::Token;

/// Format version tag at the head of every compressed frame.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameVersion {
    /// `varint` only (no zstd).
    VarintOnly = 0x01,
    /// `varint` then zstd, no dictionary.
    VarintZstd = 0x02,
    /// `varint` then zstd with a dictionary; 4-byte dict_id follows the tag.
    VarintZstdDict = 0x03,
    /// Delta against a canonical segment. 32-byte hash + op stream follows.
    Delta = 0x04,
}

impl FrameVersion {
    fn from_byte(byte: u8) -> Result<Self, CompressionError> {
        match byte {
            0x01 => Ok(Self::VarintOnly),
            0x02 => Ok(Self::VarintZstd),
            0x03 => Ok(Self::VarintZstdDict),
            0x04 => Ok(Self::Delta),
            other => Err(CompressionError::UnsupportedVersion(other)),
        }
    }
}

/// Looks up the tokens of a canonical segment by hash. Implemented by the
/// vault on top of its storage backend; injected into the compression
/// pipeline so frame `0x04` decompression can resolve cross-segment refs
/// without coupling compression to the backend trait.
pub trait CanonicalResolver: Send + Sync {
    fn resolve(
        &self,
        hash: &crate::types::SegmentHash,
    ) -> Result<Vec<crate::types::Token>, CompressionError>;
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
    #[error("frame is truncated (expected at least {expected} bytes, got {got})")]
    Truncated { expected: usize, got: usize },
    #[error("dictionary error: {0}")]
    Dict(#[from] DictError),
    #[error("delta frame requires a canonical resolver but none is attached")]
    NoCanonicalResolver,
    #[error("delta frame error: {0}")]
    Delta(String),
}

/// Full compression/decompression pipeline.
///
/// Holds an optional [`DictRegistry`] (for `0x03` frames) and an optional
/// [`CanonicalResolver`] (for `0x04` frames). New compressions never
/// produce `0x04` directly — that path is driven by the vault's near-dedup
/// layer, which calls `compress_delta` explicitly. The pipeline only needs
/// the resolver to *decompress* `0x04` frames.
#[derive(Clone)]
pub struct CompressionPipeline {
    enabled: bool,
    level: i32,
    registry: Option<Arc<DictRegistry>>,
    resolver: Option<Arc<dyn CanonicalResolver>>,
}

impl CompressionPipeline {
    pub fn new(enabled: bool, compression_level: i32) -> Self {
        Self { enabled, level: compression_level, registry: None, resolver: None }
    }

    /// Attach a dictionary registry. Compression and decompression will use
    /// it for `0x03` frames; `0x01` / `0x02` frames keep working unchanged.
    pub fn with_registry(mut self, registry: Arc<DictRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Attach a canonical resolver so `0x04` (delta) frames can decompress.
    /// Without one, decompressing a `0x04` frame returns
    /// `CompressionError::NoCanonicalResolver`.
    pub fn with_resolver(mut self, resolver: Arc<dyn CanonicalResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Build a `0x04` (delta) frame against the named canonical segment.
    /// Caller is responsible for ensuring the canonical exists in storage —
    /// the pipeline does not check, since checking would require a backend.
    pub fn compress_delta(
        &self,
        canonical_hash: &crate::types::SegmentHash,
        ops: &[crate::near_dedup::DeltaOp],
    ) -> Result<Vec<u8>, CompressionError> {
        let raw_hash = crate::near_dedup::segment_hash_to_bytes(canonical_hash)
            .map_err(|e| CompressionError::Delta(e.to_string()))?;
        let payload = crate::near_dedup::encode_delta(ops);
        let mut out = Vec::with_capacity(33 + payload.len());
        out.push(FrameVersion::Delta as u8);
        out.extend_from_slice(&raw_hash);
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Inspect a frame's canonical hash, if it's a `0x04` frame. Returns
    /// `None` for any other frame version.
    pub fn frame_canonical_hash(
        &self,
        data: &[u8],
    ) -> Result<Option<crate::types::SegmentHash>, CompressionError> {
        let (&version_byte, rest) = data.split_first().ok_or(CompressionError::EmptyFrame)?;
        if FrameVersion::from_byte(version_byte)? != FrameVersion::Delta {
            return Ok(None);
        }
        if rest.len() < 32 {
            return Err(CompressionError::Truncated { expected: 32, got: rest.len() });
        }
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&rest[..32]);
        Ok(Some(crate::near_dedup::bytes_to_segment_hash(&bytes)))
    }

    /// Compress a token sequence into a versioned frame.
    pub fn compress(&self, tokens: &[Token]) -> Result<Vec<u8>, CompressionError> {
        let varint_bytes = varint::encode_tokens(tokens);

        if !self.enabled {
            let mut out = Vec::with_capacity(varint_bytes.len() + 1);
            out.push(FrameVersion::VarintOnly as u8);
            out.extend_from_slice(&varint_bytes);
            return Ok(out);
        }

        // Prefer the active dictionary if a registry has one.
        if let Some(registry) = &self.registry {
            if let Some(active_id) = registry.active_id() {
                let dict_bytes = registry.get_bytes(active_id)?;
                let mut compressor =
                    zstd::bulk::Compressor::with_dictionary(self.level, &dict_bytes)
                        .map_err(|e| CompressionError::Compress(e.to_string()))?;
                let zstd_bytes = compressor
                    .compress(&varint_bytes)
                    .map_err(|e| CompressionError::Compress(e.to_string()))?;

                let mut out = Vec::with_capacity(zstd_bytes.len() + 5);
                out.push(FrameVersion::VarintZstdDict as u8);
                out.extend_from_slice(&active_id.to_le_bytes());
                out.extend_from_slice(&zstd_bytes);
                return Ok(out);
            }
        }

        // No active dict — fall back to plain varint+zstd.
        let zstd_bytes = zstd::bulk::compress(&varint_bytes, self.level)
            .map_err(|e| CompressionError::Compress(e.to_string()))?;
        let mut out = Vec::with_capacity(zstd_bytes.len() + 1);
        out.push(FrameVersion::VarintZstd as u8);
        out.extend_from_slice(&zstd_bytes);
        Ok(out)
    }

    /// Decompress a versioned frame back to a token sequence.
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<Token>, CompressionError> {
        let (&version_byte, rest) = data.split_first().ok_or(CompressionError::EmptyFrame)?;
        let version = FrameVersion::from_byte(version_byte)?;

        // 0x04 takes a different path entirely — its payload is a delta
        // against another segment, not a varint-encoded token stream.
        if version == FrameVersion::Delta {
            return self.decompress_delta(rest);
        }

        let varint_bytes = match version {
            FrameVersion::VarintOnly => rest.to_vec(),
            FrameVersion::VarintZstd => zstd::stream::decode_all(rest)
                .map_err(|e| CompressionError::Decompress(e.to_string()))?,
            FrameVersion::VarintZstdDict => {
                if rest.len() < 4 {
                    return Err(CompressionError::Truncated {
                        expected: 4,
                        got: rest.len(),
                    });
                }
                let (id_bytes, payload) = rest.split_at(4);
                let dict_id = u32::from_le_bytes([
                    id_bytes[0],
                    id_bytes[1],
                    id_bytes[2],
                    id_bytes[3],
                ]);

                let registry = self.registry.as_ref().ok_or_else(|| {
                    CompressionError::Decompress(format!(
                        "frame requires dict_id {dict_id} but no registry is attached"
                    ))
                })?;
                let dict_bytes = registry.get_bytes(dict_id)?;
                let mut decoder = zstd::bulk::Decompressor::with_dictionary(&dict_bytes)
                    .map_err(|e| CompressionError::Decompress(e.to_string()))?;
                decoder
                    .decompress(payload, 256 * 1024 * 1024)
                    .map_err(|e| CompressionError::Decompress(e.to_string()))?
            }
            FrameVersion::Delta => unreachable!("handled above"),
        };

        varint::decode_tokens(&varint_bytes)
            .map_err(|e| CompressionError::Decompress(e.to_string()))
    }

    /// Decompression path for `0x04` frames: read the canonical hash,
    /// resolve to tokens, decode the delta, apply.
    fn decompress_delta(&self, body: &[u8]) -> Result<Vec<Token>, CompressionError> {
        if body.len() < 32 {
            return Err(CompressionError::Truncated { expected: 32, got: body.len() });
        }
        let (hash_bytes, delta_bytes) = body.split_at(32);
        let mut raw_hash = [0u8; 32];
        raw_hash.copy_from_slice(hash_bytes);
        let canonical_hash = crate::near_dedup::bytes_to_segment_hash(&raw_hash);

        let resolver = self.resolver.as_ref().ok_or(CompressionError::NoCanonicalResolver)?;
        let canonical_tokens = resolver.resolve(&canonical_hash)?;

        let ops = crate::near_dedup::decode_delta(delta_bytes)
            .map_err(|e| CompressionError::Delta(e.to_string()))?;
        crate::near_dedup::apply_delta(&canonical_tokens, &ops)
            .map_err(|e| CompressionError::Delta(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict_registry::DictRegistry;

    fn train_dict(seed: u32) -> Vec<u8> {
        let samples: Vec<Vec<u8>> = (0u32..30)
            .map(|i| {
                let tokens: Vec<u32> = (0..200).map(|t| (t + i * seed) % 30_000).collect();
                varint::encode_tokens(&tokens)
            })
            .collect();
        let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
        zstd::dict::from_samples(&refs, 4096).expect("train dict")
    }

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

    #[test]
    fn dict_frame_roundtrip() {
        let registry = Arc::new(DictRegistry::in_memory());
        let dict = train_dict(7);
        let info = registry.register(dict, 30).unwrap();
        registry.activate(info.id).unwrap();

        let pipeline = CompressionPipeline::new(true, 3).with_registry(Arc::clone(&registry));
        let tokens: Vec<Token> = (0u32..500).collect();
        let frame = pipeline.compress(&tokens).unwrap();

        assert_eq!(frame[0], FrameVersion::VarintZstdDict as u8);
        let dict_id_in_frame =
            u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
        assert_eq!(dict_id_in_frame, info.id);

        assert_eq!(pipeline.decompress(&frame).unwrap(), tokens);
    }

    #[test]
    fn rotation_old_dict_frames_remain_readable() {
        // The v0.1 data-loss scenario: train dict A → store with A → train
        // dict B → activate B → store with B → confirm A's frame still
        // decompresses. v0.1 lost the A-compressed data here because dict A
        // was overwritten in memory.
        let registry = Arc::new(DictRegistry::in_memory());

        let dict_a = train_dict(11);
        let info_a = registry.register(dict_a, 30).unwrap();
        registry.activate(info_a.id).unwrap();

        let pipeline = CompressionPipeline::new(true, 3).with_registry(Arc::clone(&registry));
        let tokens_a: Vec<Token> = (0u32..400).collect();
        let frame_a = pipeline.compress(&tokens_a).unwrap();

        // Train and rotate to dict B.
        let dict_b = train_dict(29);
        let info_b = registry.register(dict_b, 30).unwrap();
        registry.activate(info_b.id).unwrap();
        assert_ne!(info_a.id, info_b.id, "test setup: dicts should differ");

        let tokens_b: Vec<Token> = (1000u32..1400).collect();
        let frame_b = pipeline.compress(&tokens_b).unwrap();

        // Both frames must decompress correctly even though A is no longer active.
        assert_eq!(pipeline.decompress(&frame_a).unwrap(), tokens_a, "old dict frame lost!");
        assert_eq!(pipeline.decompress(&frame_b).unwrap(), tokens_b);

        // And the second frame should be tagged with B's ID.
        let id_in_frame_b =
            u32::from_le_bytes([frame_b[1], frame_b[2], frame_b[3], frame_b[4]]);
        assert_eq!(id_in_frame_b, info_b.id);
    }

    #[test]
    fn dict_frame_without_registry_errors_clearly() {
        // Compress with a registry, then try to decompress on a fresh
        // pipeline that has no registry. Should fail with a clear message,
        // not corrupt data or panic.
        let registry = Arc::new(DictRegistry::in_memory());
        let info = registry.register(train_dict(31), 30).unwrap();
        registry.activate(info.id).unwrap();
        let with_reg = CompressionPipeline::new(true, 3).with_registry(Arc::clone(&registry));

        let tokens: Vec<Token> = (0u32..100).collect();
        let frame = with_reg.compress(&tokens).unwrap();

        let without_reg = CompressionPipeline::new(true, 3);
        let err = without_reg.decompress(&frame).unwrap_err();
        match err {
            CompressionError::Decompress(msg) => {
                assert!(msg.contains("no registry"), "unexpected message: {msg}");
            }
            other => panic!("expected Decompress error, got {other:?}"),
        }
    }

    #[test]
    fn delta_frame_roundtrip() {
        use crate::near_dedup::{compute_delta, MinHashSignature};
        use crate::types::SegmentHash;
        use std::sync::Mutex;

        let canonical: Vec<Token> = (0u32..200).collect();
        let mut variant = canonical.clone();
        variant[10] = 9999;
        variant[150] = 8888;

        // The canonical's hash is whatever we like for this test; use a
        // realistic 64-char hex string. The pipeline doesn't validate that
        // the hash matches the tokens — that's the vault's job.
        let canonical_hash = SegmentHash(format!("{:0>64}", "deadbeef"));

        struct StubResolver {
            map: Mutex<std::collections::HashMap<SegmentHash, Vec<Token>>>,
        }
        impl super::CanonicalResolver for StubResolver {
            fn resolve(&self, hash: &SegmentHash) -> Result<Vec<Token>, super::CompressionError> {
                self.map
                    .lock()
                    .unwrap()
                    .get(hash)
                    .cloned()
                    .ok_or_else(|| super::CompressionError::Decompress(format!("no canonical: {}", hash.0)))
            }
        }

        let resolver = Arc::new(StubResolver {
            map: Mutex::new(
                std::iter::once((canonical_hash.clone(), canonical.clone())).collect(),
            ),
        });

        let pipeline = CompressionPipeline::new(true, 3).with_resolver(resolver);
        let ops = compute_delta(&canonical, &variant);
        let frame = pipeline.compress_delta(&canonical_hash, &ops).unwrap();

        assert_eq!(frame[0], FrameVersion::Delta as u8);
        let recovered = pipeline.decompress(&frame).unwrap();
        assert_eq!(recovered, variant);

        // The canonical hash field should round-trip via frame_canonical_hash().
        assert_eq!(
            pipeline.frame_canonical_hash(&frame).unwrap(),
            Some(canonical_hash.clone())
        );

        // And signatures of variant tokens recovered through the delta
        // match signatures of variant tokens directly.
        let direct_sig = MinHashSignature::compute(&variant);
        let recovered_sig = MinHashSignature::compute(&recovered);
        assert_eq!(direct_sig, recovered_sig);
    }

    #[test]
    fn delta_frame_without_resolver_errors_clearly() {
        use crate::near_dedup::compute_delta;
        use crate::types::SegmentHash;

        let canonical: Vec<Token> = (0u32..50).collect();
        let variant: Vec<Token> = (0u32..50).rev().collect();
        let canonical_hash = SegmentHash(format!("{:0>64}", "feedface"));

        let pipeline = CompressionPipeline::new(true, 3);
        let ops = compute_delta(&canonical, &variant);
        let frame = pipeline.compress_delta(&canonical_hash, &ops).unwrap();

        // Decompressing without a resolver should fail with the dedicated error.
        let err = pipeline.decompress(&frame).unwrap_err();
        assert!(matches!(err, CompressionError::NoCanonicalResolver));
    }

    #[test]
    fn truncated_dict_frame_errors() {
        let pipeline = CompressionPipeline::new(true, 3);
        let truncated = [0x03u8, 1, 2]; // dict_id needs 4 bytes
        assert!(matches!(
            pipeline.decompress(&truncated),
            Err(CompressionError::Truncated { .. })
        ));
    }
}
