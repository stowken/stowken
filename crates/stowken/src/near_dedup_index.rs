//! Per-vault near-duplicate index: MinHash signatures + LSH band buckets,
//! plus a variant→canonical embedding-alias map.
//!
//! # Why this is not in SQLite
//!
//! The v0.4 implementation kept the LSH index in two SQLite tables. That
//! couples near-dedup to the metadata DB, doesn't survive being shipped to
//! a distributed store, and pays a SQL round-trip per band probe on the
//! ingest hot path. This module replaces it with self-contained append-only
//! logs under the vault root:
//!
//! ```text
//! near_dedup/
//! ├── signatures.log    # [32-byte hash][1024-byte signature] records
//! ├── aliases.log       # [32-byte variant][32-byte canonical] records
//! └── tombstones.log    # [32-byte hash] records (removals)
//! ```
//!
//! Append-only keeps insert O(1) on the ingest hot path — no manifest
//! rewrite per segment. The whole index is materialised in RAM at `open`
//! (candidate lookup must not pay disk I/O per probe); band buckets are
//! *derived* from the signatures at load, not persisted. Memory-backed
//! vaults skip persistence entirely.
//!
//! Removals append a tombstone rather than rewriting; dead log bytes are
//! reclaimed only by a full `reindex` (which calls [`NearDupIndex::clear`]
//! then re-inserts). Compaction-in-place is a future optimization.
//!
//! # Memory
//!
//! Signatures live in RAM at [`MinHashSignature`]'s width (1024 bytes each
//! at the current `NUM_HASHES = 128`). For very large vaults this dominates;
//! a future tunable can shrink `NUM_HASHES`/value width without touching
//! this module's interface.
//!
//! # Embedding aliases
//!
//! When the vault decides a variant is close enough to a canonical to share
//! its embedding (a *higher* bar than the storage-delta threshold), it
//! records the mapping here. The embed path reads it back to copy the
//! canonical's vector instead of paying for a fresh embedding call — and,
//! critically, to keep delta-stored variants searchable at all.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use thiserror::Error;

use crate::near_dedup::{
    bytes_to_segment_hash, segment_hash_to_bytes, MinHashSignature, NearDedupError, BANDS,
    NUM_HASHES,
};
use crate::types::SegmentHash;

/// Bytes per serialized MinHash signature (128 × u64, little-endian).
const SIGNATURE_BYTES: usize = NUM_HASHES * 8;
/// One `signatures.log` record: raw hash followed by its signature.
const SIG_RECORD: usize = 32 + SIGNATURE_BYTES;
/// One `aliases.log` record: variant hash followed by canonical hash.
const ALIAS_RECORD: usize = 64;
/// One `tombstones.log` record: a single raw hash.
const TOMB_RECORD: usize = 32;

const SIG_LOG: &str = "signatures.log";
const ALIAS_LOG: &str = "aliases.log";
const TOMB_LOG: &str = "tombstones.log";

#[derive(Debug, Error)]
pub enum NearDupIndexError {
    #[error("near-dedup index I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("near-dedup signature error: {0}")]
    Signature(#[from] NearDedupError),
}

pub type NearDupIndexResult<T> = Result<T, NearDupIndexError>;

/// Shared near-duplicate index for one vault.
#[derive(Clone)]
pub struct NearDupIndex {
    inner: Arc<RwLock<IndexState>>,
}

struct IndexState {
    /// Where the log files live. `None` for in-memory indexes.
    root: Option<PathBuf>,
    /// segment hash → full signature.
    signatures: HashMap<SegmentHash, MinHashSignature>,
    /// (band index, band hash) → segment hashes that fall in that bucket.
    bands: HashMap<(u32, u64), Vec<SegmentHash>>,
    /// variant hash → canonical hash whose embedding it reuses.
    aliases: HashMap<SegmentHash, SegmentHash>,
}

impl IndexState {
    fn index_bands(&mut self, hash: &SegmentHash, sig: &MinHashSignature) {
        for band in 0..BANDS {
            let key = (band as u32, sig.band_hash(band));
            let bucket = self.bands.entry(key).or_default();
            if !bucket.contains(hash) {
                bucket.push(hash.clone());
            }
        }
    }

    fn deindex_bands(&mut self, hash: &SegmentHash, sig: &MinHashSignature) {
        for band in 0..BANDS {
            let key = (band as u32, sig.band_hash(band));
            if let Some(bucket) = self.bands.get_mut(&key) {
                bucket.retain(|h| h != hash);
                if bucket.is_empty() {
                    self.bands.remove(&key);
                }
            }
        }
    }
}

impl NearDupIndex {
    /// Construct an in-memory index (no persistence).
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(RwLock::new(IndexState {
                root: None,
                signatures: HashMap::new(),
                bands: HashMap::new(),
                aliases: HashMap::new(),
            })),
        }
    }

    /// Open (or create) a filesystem-backed index rooted at `root`. Replays
    /// the append-only logs and rebuilds the band buckets in RAM.
    pub fn open<P: AsRef<Path>>(root: P) -> NearDupIndexResult<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;

        let mut signatures: HashMap<SegmentHash, MinHashSignature> = HashMap::new();
        let mut aliases: HashMap<SegmentHash, SegmentHash> = HashMap::new();

        // Replay signatures.log.
        for rec in read_records(&root.join(SIG_LOG), SIG_RECORD)? {
            let hash = bytes_to_segment_hash(rec[..32].try_into().expect("32 bytes"));
            let sig = MinHashSignature::from_bytes(&rec[32..])?;
            signatures.insert(hash, sig);
        }
        // Replay aliases.log (last write wins).
        for rec in read_records(&root.join(ALIAS_LOG), ALIAS_RECORD)? {
            let variant = bytes_to_segment_hash(rec[..32].try_into().expect("32 bytes"));
            let canonical = bytes_to_segment_hash(rec[32..].try_into().expect("32 bytes"));
            aliases.insert(variant, canonical);
        }
        // Apply tombstones: drop the hash as signature, as alias variant, and
        // as alias canonical.
        for rec in read_records(&root.join(TOMB_LOG), TOMB_RECORD)? {
            let hash = bytes_to_segment_hash(rec[..32].try_into().expect("32 bytes"));
            signatures.remove(&hash);
            aliases.remove(&hash);
            aliases.retain(|_, c| *c != hash);
        }

        let mut state = IndexState {
            root: Some(root),
            signatures: HashMap::new(),
            bands: HashMap::new(),
            aliases,
        };
        let sigs: Vec<(SegmentHash, MinHashSignature)> = signatures.into_iter().collect();
        for (hash, sig) in sigs {
            state.index_bands(&hash, &sig);
            state.signatures.insert(hash, sig);
        }

        Ok(Self {
            inner: Arc::new(RwLock::new(state)),
        })
    }

    /// Insert a segment's signature. Idempotent — re-inserting an existing
    /// hash is a no-op (a hash's signature is deterministic).
    pub fn insert(&self, hash: &SegmentHash, sig: &MinHashSignature) -> NearDupIndexResult<()> {
        let mut state = self.inner.write().unwrap();
        if state.signatures.contains_key(hash) {
            return Ok(());
        }
        if let Some(root) = state.root.clone() {
            let mut rec = Vec::with_capacity(SIG_RECORD);
            rec.extend_from_slice(&segment_hash_to_bytes(hash)?);
            rec.extend_from_slice(&sig.to_bytes());
            append_record(&root.join(SIG_LOG), &rec)?;
        }
        state.index_bands(hash, sig);
        state.signatures.insert(hash.clone(), sig.clone());
        Ok(())
    }

    /// Return the deduplicated set of candidate hashes sharing at least one
    /// LSH band bucket with the supplied band hashes.
    pub fn candidates(&self, band_hashes: &[u64]) -> Vec<SegmentHash> {
        let state = self.inner.read().unwrap();
        let mut out: Vec<SegmentHash> = Vec::new();
        let mut seen: HashSet<&SegmentHash> = HashSet::new();
        for (band, bh) in band_hashes.iter().enumerate() {
            if let Some(bucket) = state.bands.get(&(band as u32, *bh)) {
                for h in bucket {
                    if seen.insert(h) {
                        out.push(h.clone());
                    }
                }
            }
        }
        out
    }

    /// Fetch signatures for a set of hashes (those present in the index).
    pub fn signatures_batch(
        &self,
        hashes: &[SegmentHash],
    ) -> HashMap<SegmentHash, MinHashSignature> {
        let state = self.inner.read().unwrap();
        let mut out = HashMap::with_capacity(hashes.len());
        for h in hashes {
            if let Some(sig) = state.signatures.get(h) {
                out.insert(h.clone(), sig.clone());
            }
        }
        out
    }

    /// Single-signature lookup.
    pub fn signature(&self, hash: &SegmentHash) -> Option<MinHashSignature> {
        self.inner.read().unwrap().signatures.get(hash).cloned()
    }

    /// Record that `variant` should reuse `canonical`'s embedding.
    pub fn set_embedding_alias(
        &self,
        variant: &SegmentHash,
        canonical: &SegmentHash,
    ) -> NearDupIndexResult<()> {
        let mut state = self.inner.write().unwrap();
        if state.aliases.get(variant) == Some(canonical) {
            return Ok(());
        }
        if let Some(root) = state.root.clone() {
            let mut rec = Vec::with_capacity(ALIAS_RECORD);
            rec.extend_from_slice(&segment_hash_to_bytes(variant)?);
            rec.extend_from_slice(&segment_hash_to_bytes(canonical)?);
            append_record(&root.join(ALIAS_LOG), &rec)?;
        }
        state.aliases.insert(variant.clone(), canonical.clone());
        Ok(())
    }

    /// The canonical whose embedding `variant` reuses, if any.
    pub fn embedding_alias(&self, variant: &SegmentHash) -> Option<SegmentHash> {
        self.inner.read().unwrap().aliases.get(variant).cloned()
    }

    /// Remove a segment from the index (signature, bands, and any alias rows
    /// referencing it as variant or canonical). Appends a tombstone.
    pub fn remove(&self, hash: &SegmentHash) -> NearDupIndexResult<()> {
        let mut state = self.inner.write().unwrap();
        if let Some(root) = state.root.clone() {
            append_record(&root.join(TOMB_LOG), &segment_hash_to_bytes(hash)?)?;
        }
        if let Some(sig) = state.signatures.remove(hash) {
            state.deindex_bands(hash, &sig);
        }
        state.aliases.remove(hash);
        state.aliases.retain(|_, c| c != hash);
        Ok(())
    }

    /// Number of indexed signatures.
    pub fn len(&self) -> u64 {
        self.inner.read().unwrap().signatures.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().signatures.is_empty()
    }

    /// Drop everything and truncate the logs. Used by `reindex` before a
    /// full rebuild.
    pub fn clear(&self) -> NearDupIndexResult<()> {
        let mut state = self.inner.write().unwrap();
        state.signatures.clear();
        state.bands.clear();
        state.aliases.clear();
        if let Some(root) = state.root.clone() {
            for log in [SIG_LOG, ALIAS_LOG, TOMB_LOG] {
                let path = root.join(log);
                if path.exists() {
                    std::fs::write(&path, [])?;
                }
            }
        }
        Ok(())
    }
}

/// Append a single record durably (`O(1)`).
fn append_record(path: &Path, rec: &[u8]) -> NearDupIndexResult<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(rec)?;
    f.sync_data()?;
    Ok(())
}

/// Read a fixed-size-record log, dropping a torn trailing partial record
/// (e.g. from a crash mid-append).
fn read_records(path: &Path, record_size: usize) -> NearDupIndexResult<Vec<Vec<u8>>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read(path)?;
    let n = raw.len() / record_size;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * record_size;
        out.push(raw[start..start + record_size].to_vec());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Token;

    fn sig_of(range: std::ops::Range<u32>) -> MinHashSignature {
        let tokens: Vec<Token> = range.collect();
        MinHashSignature::compute(&tokens)
    }

    // Real segment hashes are 64-char SHA-256 hex; the packed-log format
    // round-trips them through `segment_hash_to_bytes`, so fixtures must be
    // valid hex too.
    fn hash(seed: u8) -> SegmentHash {
        SegmentHash(hex::encode([seed; 32]))
    }

    #[test]
    fn insert_then_candidate_lookup() {
        let idx = NearDupIndex::in_memory();
        let h = hash(1);
        let sig = sig_of(0..150);
        idx.insert(&h, &sig).unwrap();

        let bands: Vec<u64> = (0..BANDS).map(|b| sig.band_hash(b)).collect();
        assert!(idx.candidates(&bands).contains(&h));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn similar_segment_is_a_candidate() {
        let idx = NearDupIndex::in_memory();
        let canonical = hash(2);
        idx.insert(&canonical, &sig_of(0..150)).unwrap();

        let mut variant_tokens: Vec<Token> = (0..150).collect();
        variant_tokens[20] = 99_999;
        variant_tokens[80] = 88_888;
        let variant_sig = MinHashSignature::compute(&variant_tokens);
        let bands: Vec<u64> = (0..BANDS).map(|b| variant_sig.band_hash(b)).collect();

        assert!(
            idx.candidates(&bands).contains(&canonical),
            "high-similarity variant should surface the canonical"
        );
    }

    #[test]
    fn alias_roundtrip() {
        let idx = NearDupIndex::in_memory();
        let v = hash(3);
        let c = hash(2);
        assert_eq!(idx.embedding_alias(&v), None);
        idx.set_embedding_alias(&v, &c).unwrap();
        assert_eq!(idx.embedding_alias(&v), Some(c));
    }

    #[test]
    fn persists_and_reopens() {
        let dir = std::env::temp_dir().join(format!("stowken_neardup_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let h1 = hash(4);
        let h2 = hash(5);
        let v = hash(3);
        {
            let idx = NearDupIndex::open(&dir).unwrap();
            idx.insert(&h1, &sig_of(0..150)).unwrap();
            idx.insert(&h2, &sig_of(500..700)).unwrap();
            idx.set_embedding_alias(&v, &h1).unwrap();
            assert_eq!(idx.len(), 2);
        }
        {
            let idx = NearDupIndex::open(&dir).unwrap();
            assert_eq!(idx.len(), 2);
            assert!(idx.signature(&h1).is_some());
            assert_eq!(idx.embedding_alias(&v), Some(h1.clone()));

            let sig = idx.signature(&h1).unwrap();
            let bands: Vec<u64> = (0..BANDS).map(|b| sig.band_hash(b)).collect();
            assert!(idx.candidates(&bands).contains(&h1));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tombstone_survives_reopen() {
        let dir = std::env::temp_dir().join(format!("stowken_neardup_tomb_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let h = hash(6);
        {
            let idx = NearDupIndex::open(&dir).unwrap();
            idx.insert(&h, &sig_of(0..150)).unwrap();
            idx.remove(&h).unwrap();
            assert_eq!(idx.len(), 0);
        }
        {
            let idx = NearDupIndex::open(&dir).unwrap();
            assert_eq!(
                idx.len(),
                0,
                "tombstone should suppress the re-replayed insert"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
