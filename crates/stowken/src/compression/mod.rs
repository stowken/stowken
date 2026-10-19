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
//! | 0x05 | substring-referenced segment      | sequence of `Inline(tokens)` and `Ref(substring_id)` ops |
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
//!
//! For 0x05, the body is a sequence of opcodes:
//!   - `0x01 || varint(length) || varint(token) × length` — inline tokens
//!   - `0x02 || varint(substring_id)` — reference an entry in the
//!     [`crate::substring_registry::SubstringRegistry`]
//!
//! Decompression streams the ops, emitting inline tokens directly and
//! looking up referenced substrings from the registry. Used by token-level
//! shared-substring dedup — one shared blob can replace the same long
//! token sequence across thousands of otherwise-distinct segments.

pub mod varint;

use std::io::Read;
use std::sync::Arc;

use thiserror::Error;

use crate::dict_registry::{DictError, DictRegistry};
use crate::substring_registry::{SubstringError, SubstringId, SubstringRegistry};
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
    /// Sequence of inline runs and substring references.
    Substring = 0x05,
}

impl FrameVersion {
    fn from_byte(byte: u8) -> Result<Self, CompressionError> {
        match byte {
            0x01 => Ok(Self::VarintOnly),
            0x02 => Ok(Self::VarintZstd),
            0x03 => Ok(Self::VarintZstdDict),
            0x04 => Ok(Self::Delta),
            0x05 => Ok(Self::Substring),
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
    #[error("substring registry error: {0}")]
    Substring(#[from] SubstringError),
    #[error("substring frame requires a registry but none is attached")]
    NoSubstringRegistry,
}

/// Op in a `0x05` substring-referenced frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstringOp {
    /// Emit `tokens` directly.
    Inline(Vec<Token>),
    /// Resolve `id` in the substring registry and emit those tokens.
    Ref(SubstringId),
}

const SUBSTRING_OP_INLINE: u8 = 0x01;
const SUBSTRING_OP_REF: u8 = 0x02;

/// Encode a sequence of substring ops as the body of a `0x05` frame.
/// Inserts the version tag at position 0.
pub fn encode_substring_frame(ops: &[SubstringOp]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(FrameVersion::Substring as u8);
    for op in ops {
        match op {
            SubstringOp::Inline(tokens) => {
                out.push(SUBSTRING_OP_INLINE);
                varint_write_u32(tokens.len() as u32, &mut out);
                for t in tokens {
                    varint_write_u32(*t, &mut out);
                }
            }
            SubstringOp::Ref(id) => {
                out.push(SUBSTRING_OP_REF);
                varint_write_u32(*id, &mut out);
            }
        }
    }
    out
}

/// Decode the body of a `0x05` frame (caller has already stripped the
/// version byte) into a sequence of substring ops.
pub fn decode_substring_frame(body: &[u8]) -> Result<Vec<SubstringOp>, CompressionError> {
    let mut ops = Vec::new();
    let mut cursor = std::io::Cursor::new(body);
    while (cursor.position() as usize) < body.len() {
        let mut tag = [0u8; 1];
        cursor
            .read_exact(&mut tag)
            .map_err(|e| CompressionError::Decompress(format!("substring frame: {e}")))?;
        match tag[0] {
            SUBSTRING_OP_INLINE => {
                let len = varint_read_u32(&mut cursor)?;
                let mut tokens = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    tokens.push(varint_read_u32(&mut cursor)?);
                }
                ops.push(SubstringOp::Inline(tokens));
            }
            SUBSTRING_OP_REF => {
                let id = varint_read_u32(&mut cursor)?;
                ops.push(SubstringOp::Ref(id));
            }
            other => {
                return Err(CompressionError::Decompress(format!(
                    "unknown substring op tag 0x{other:02x}"
                )));
            }
        }
    }
    Ok(ops)
}

fn varint_write_u32(mut v: u32, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn varint_read_u32(cursor: &mut std::io::Cursor<&[u8]>) -> Result<u32, CompressionError> {
    let mut shift: u32 = 0;
    let mut result: u32 = 0;
    loop {
        let mut byte = [0u8; 1];
        cursor
            .read_exact(&mut byte)
            .map_err(|e| CompressionError::Decompress(format!("varint truncated: {e}")))?;
        let b = byte[0];
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return Err(CompressionError::Decompress("varint overflows u32".into()));
        }
    }
    Ok(result)
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
    substrings: Option<Arc<SubstringRegistry>>,
}

impl CompressionPipeline {
    pub fn new(enabled: bool, compression_level: i32) -> Self {
        Self {
            enabled,
            level: compression_level,
            registry: None,
            resolver: None,
            substrings: None,
        }
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

    /// Attach a substring registry. Required for `0x05` frame decompression
    /// and used by the vault to build `0x05` frames at write time.
    pub fn with_substring_registry(mut self, substrings: Arc<SubstringRegistry>) -> Self {
        self.substrings = Some(substrings);
        self
    }

    /// Read access to the attached substring registry.
    pub fn substring_registry(&self) -> Option<&Arc<SubstringRegistry>> {
        self.substrings.as_ref()
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
            return Err(CompressionError::Truncated {
                expected: 32,
                got: rest.len(),
            });
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

        // 0x04 and 0x05 take different paths entirely — their payloads
        // aren't varint-encoded token streams.
        if version == FrameVersion::Delta {
            return self.decompress_delta(rest);
        }
        if version == FrameVersion::Substring {
            return self.decompress_substring_frame(rest);
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
                let dict_id =
                    u32::from_le_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]);

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
            FrameVersion::Delta | FrameVersion::Substring => unreachable!("handled above"),
        };

        varint::decode_tokens(&varint_bytes)
            .map_err(|e| CompressionError::Decompress(e.to_string()))
    }

    /// Decompression path for `0x05` frames: stream the ops, emitting
    /// inline tokens directly and looking up substring refs from the
    /// registry.
    fn decompress_substring_frame(&self, body: &[u8]) -> Result<Vec<Token>, CompressionError> {
        let registry = self
            .substrings
            .as_ref()
            .ok_or(CompressionError::NoSubstringRegistry)?;
        let ops = decode_substring_frame(body)?;
        // Pre-allocate based on op count — a rough upper bound is enough
        // to avoid most reallocations.
        let mut out = Vec::with_capacity(ops.len() * 16);
        for op in ops {
            match op {
                SubstringOp::Inline(tokens) => out.extend_from_slice(&tokens),
                SubstringOp::Ref(id) => {
                    let tokens = registry.get_tokens(id)?;
                    out.extend_from_slice(tokens.as_slice());
                }
            }
        }
        Ok(out)
    }

    /// Greedy substring-match over `tokens` using the attached
    /// substring registry. Returns the encoded `0x05` frame *only if*
    /// it ends up shorter than `fallback_len` (typically the size of a
    /// fresh full compression of the same tokens). Otherwise returns
    /// `None` so the caller can fall back to a normal frame.
    pub fn try_compress_substring_frame(
        &self,
        tokens: &[Token],
        fallback_len: usize,
    ) -> Result<Option<Vec<u8>>, CompressionError> {
        let registry = match &self.substrings {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };

        let mut ops: Vec<SubstringOp> = Vec::new();
        let mut inline_buffer: Vec<Token> = Vec::new();
        let mut i = 0usize;
        let mut any_ref = false;

        while i < tokens.len() {
            if let Some((id, length)) = registry.find_longest_match_at(&tokens[i..]) {
                if !inline_buffer.is_empty() {
                    ops.push(SubstringOp::Inline(std::mem::take(&mut inline_buffer)));
                }
                ops.push(SubstringOp::Ref(id));
                any_ref = true;
                i += length;
            } else {
                inline_buffer.push(tokens[i]);
                i += 1;
            }
        }
        if !inline_buffer.is_empty() {
            ops.push(SubstringOp::Inline(inline_buffer));
        }

        // Skip 0x05 entirely if we never matched a substring — a
        // fully-inline frame is strictly worse than 0x02.
        if !any_ref {
            return Ok(None);
        }

        let frame = encode_substring_frame(&ops);
        if frame.len() < fallback_len {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    /// Decompression path for `0x04` frames: read the canonical hash,
    /// resolve to tokens, decode the delta, apply.
    fn decompress_delta(&self, body: &[u8]) -> Result<Vec<Token>, CompressionError> {
        if body.len() < 32 {
            return Err(CompressionError::Truncated {
                expected: 32,
                got: body.len(),
            });
        }
        let (hash_bytes, delta_bytes) = body.split_at(32);
        let mut raw_hash = [0u8; 32];
        raw_hash.copy_from_slice(hash_bytes);
        let canonical_hash = crate::near_dedup::bytes_to_segment_hash(&raw_hash);

        let resolver = self
            .resolver
            .as_ref()
            .ok_or(CompressionError::NoCanonicalResolver)?;
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
        assert!(matches!(
            pipeline.decompress(&[]),
            Err(CompressionError::EmptyFrame)
        ));
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
        let dict_id_in_frame = u32::from_le_bytes([frame[1], frame[2], frame[3], frame[4]]);
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
        assert_eq!(
            pipeline.decompress(&frame_a).unwrap(),
            tokens_a,
            "old dict frame lost!"
        );
        assert_eq!(pipeline.decompress(&frame_b).unwrap(), tokens_b);

        // And the second frame should be tagged with B's ID.
        let id_in_frame_b = u32::from_le_bytes([frame_b[1], frame_b[2], frame_b[3], frame_b[4]]);
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
                self.map.lock().unwrap().get(hash).cloned().ok_or_else(|| {
                    super::CompressionError::Decompress(format!("no canonical: {}", hash.0))
                })
            }
        }

        let resolver = Arc::new(StubResolver {
            map: Mutex::new(std::iter::once((canonical_hash.clone(), canonical.clone())).collect()),
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
    fn substring_frame_op_roundtrip() {
        let ops = vec![
            SubstringOp::Inline(vec![100, 200, 300]),
            SubstringOp::Ref(7),
            SubstringOp::Inline(vec![400, 500]),
            SubstringOp::Ref(42),
        ];
        let frame = encode_substring_frame(&ops);
        assert_eq!(frame[0], FrameVersion::Substring as u8);
        let decoded = decode_substring_frame(&frame[1..]).unwrap();
        assert_eq!(decoded, ops);
    }

    #[test]
    fn substring_frame_decompress_roundtrip() {
        use crate::substring_registry::SubstringRegistry;

        let registry = Arc::new(SubstringRegistry::in_memory());
        let chunk_a: Vec<Token> = (0u32..32).collect();
        let chunk_b: Vec<Token> = (1000u32..1064).collect();
        let id_a = registry.register(chunk_a.clone(), 50).unwrap().id;
        let id_b = registry.register(chunk_b.clone(), 30).unwrap().id;

        let pipeline =
            CompressionPipeline::new(true, 3).with_substring_registry(Arc::clone(&registry));

        // Encode a frame that interleaves a Ref, an Inline run, and another Ref.
        let ops = vec![
            SubstringOp::Ref(id_a),
            SubstringOp::Inline(vec![99_999, 88_888, 77_777]),
            SubstringOp::Ref(id_b),
        ];
        let frame = encode_substring_frame(&ops);

        let recovered = pipeline.decompress(&frame).unwrap();
        let mut expected = chunk_a.clone();
        expected.extend_from_slice(&[99_999, 88_888, 77_777]);
        expected.extend_from_slice(&chunk_b);
        assert_eq!(recovered, expected);
    }

    #[test]
    fn substring_frame_errors_without_registry() {
        let pipeline = CompressionPipeline::new(true, 3);
        let frame = encode_substring_frame(&[SubstringOp::Ref(1)]);
        let err = pipeline.decompress(&frame).unwrap_err();
        assert!(matches!(err, CompressionError::NoSubstringRegistry));
    }

    #[test]
    fn try_compress_substring_frame_returns_none_when_no_match() {
        use crate::substring_registry::SubstringRegistry;
        let registry = Arc::new(SubstringRegistry::in_memory());
        registry.register((0u32..32).collect(), 5).unwrap();
        let pipeline = CompressionPipeline::new(true, 3).with_substring_registry(registry);
        // No registered substring matches this prefix.
        let tokens: Vec<Token> = (90_000u32..90_064).collect();
        let result = pipeline
            .try_compress_substring_frame(&tokens, 1024)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_compress_substring_frame_returns_none_when_not_smaller() {
        use crate::substring_registry::SubstringRegistry;
        let registry = Arc::new(SubstringRegistry::in_memory());
        registry.register((0u32..32).collect(), 5).unwrap();
        let pipeline = CompressionPipeline::new(true, 3).with_substring_registry(registry);
        let tokens: Vec<Token> = (0u32..32).collect();
        // Pretend the fallback is ridiculously small — substring frame
        // can't beat that.
        let result = pipeline.try_compress_substring_frame(&tokens, 1).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_compress_substring_frame_uses_match_when_smaller() {
        use crate::substring_registry::SubstringRegistry;
        let registry = Arc::new(SubstringRegistry::in_memory());
        let chunk: Vec<Token> = (0u32..200).collect();
        registry.register(chunk.clone(), 5).unwrap();
        let pipeline =
            CompressionPipeline::new(true, 3).with_substring_registry(Arc::clone(&registry));

        // Build a token sequence: one chunk + a small inline tail.
        let mut tokens = chunk.clone();
        tokens.extend_from_slice(&[55_555, 66_666]);

        // Fallback len of 1024 is more than enough for the substring frame
        // to beat (frame is ~5 bytes for the Ref + a tiny Inline run).
        let result = pipeline
            .try_compress_substring_frame(&tokens, 1024)
            .unwrap();
        let frame = result.expect("substring frame should be smaller than fallback");
        assert_eq!(frame[0], FrameVersion::Substring as u8);

        // Round-trip: decompressing the produced frame must give the
        // exact input back.
        let recovered = pipeline.decompress(&frame).unwrap();
        assert_eq!(recovered, tokens);
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
