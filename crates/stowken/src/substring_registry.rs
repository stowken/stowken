//! Per-vault registry of promoted token substrings for v0.5 token-level
//! shared-substring dedup.
//!
//! # Why
//!
//! Segment-level dedup catches identical segments. Near-dedup catches
//! segments that are *almost* identical. Neither catches the case where
//! the same long boilerplate appears verbatim across thousands of
//! otherwise-distinct segments — e.g., "as a large language model, I
//! cannot..." showing up in 50,000 unrelated assistant turns. Substring
//! dedup promotes these recurring chunks to shared blobs and rewrites
//! incoming segments to reference them.
//!
//! # Layout
//!
//! For a filesystem-backed vault:
//!
//! ```text
//! substrings/
//! ├── blobs.dat         # concatenated varint-encoded token blobs
//! └── index.json        # array of {id, offset, length_bytes, info...}
//! ```
//!
//! Substrings are packed into a single `blobs.dat` rather than one file
//! per substring, because typical promotions are tiny (~32 bytes each)
//! and a per-substring file pays a 4 KiB cluster cost on most
//! filesystems — the directory would balloon to 100× the actual data.
//! Memory-backed vaults skip persistence entirely.
//!
//! # Discovery is NOT here
//!
//! This module is the storage and lookup layer only. The vault's
//! `train_substrings` method does the actual discovery (rolling-hash
//! window counting) and calls `register` for each promoted substring.
//!
//! # Lookup index
//!
//! For the write path (segment → 0x05 frame), the registry maintains an
//! in-memory hash index keyed by the rolling hash of each substring's
//! first [`MIN_LENGTH`] tokens. `find_longest_match_at` does a one-pass
//! lookup in O(candidates) per position.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::Token;

/// Sequential 32-bit substring identifier (1, 2, 3, ...).
pub type SubstringId = u32;

/// Minimum substring length we promote. Shorter than this, the per-ref
/// overhead (varint id, op tag, framing) eats the savings.
pub const MIN_LENGTH: usize = 16;

/// Errors from substring operations.
#[derive(Debug, Error)]
pub enum SubstringError {
    #[error("substring I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("substring {0} not found in registry")]
    NotFound(SubstringId),
    #[error("substring tokens are too short ({0} < {min} required)", min = MIN_LENGTH)]
    TooShort(usize),
    #[error("registry serialization error: {0}")]
    Serialization(String),
    #[error("varint decode error: {0}")]
    Varint(String),
}

pub type SubstringResult<T> = Result<T, SubstringError>;

/// Public-facing description of one registered substring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstringInfo {
    pub id: SubstringId,
    pub length: u32,
    pub created_at: DateTime<Utc>,
    /// How many distinct segments the discovery pass observed this
    /// substring in. Stored for inspection — not used at lookup time.
    pub source_occurrences: u32,
}

/// On-disk index entry. Layered on top of `SubstringInfo` with the
/// extra fields needed to slice the packed blob file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    #[serde(flatten)]
    info: SubstringInfo,
    /// Offset into `blobs.dat`.
    offset: u64,
    /// Length in bytes inside `blobs.dat` (varint-encoded token stream).
    bytes: u32,
}

/// Shared registry of substrings for one vault.
#[derive(Clone)]
pub struct SubstringRegistry {
    inner: Arc<RwLock<RegistryState>>,
}

struct RegistryState {
    /// Where substring files live. `None` for in-memory registries.
    root: Option<PathBuf>,
    /// id → tokens (lazy-loaded for filesystem registries).
    cache: HashMap<SubstringId, Arc<Vec<Token>>>,
    /// All known substrings (by info sidecar).
    known: HashMap<SubstringId, SubstringInfo>,
    /// Hash-of-first-MIN_LENGTH-tokens → list of substring ids that start
    /// with that window. Used by `find_longest_match_at`.
    window_index: HashMap<u64, Vec<SubstringId>>,
    /// Next sequential id for the next `register` call.
    next_id: SubstringId,
}

impl SubstringRegistry {
    /// Construct an in-memory registry (no persistence).
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryState {
                root: None,
                cache: HashMap::new(),
                known: HashMap::new(),
                window_index: HashMap::new(),
                next_id: 1,
            })),
        }
    }

    /// Open or create a filesystem-backed registry rooted at `root`.
    ///
    /// Reads `index.json` and `blobs.dat`, populates the in-memory cache
    /// and window index in one pass. `find_longest_match_at` is on the
    /// write hot path — it must not pay disk I/O per match attempt, so
    /// every promoted substring is materialised in RAM at open.
    pub fn open<P: AsRef<Path>>(root: P) -> SubstringResult<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;

        let mut known: HashMap<SubstringId, SubstringInfo> = HashMap::new();
        let mut cache: HashMap<SubstringId, Arc<Vec<Token>>> = HashMap::new();
        let mut window_index: HashMap<u64, Vec<SubstringId>> = HashMap::new();
        let mut max_seen_id: SubstringId = 0;

        let index_path = root.join("index.json");
        let blobs_path = root.join("blobs.dat");

        if index_path.exists() {
            let entries: Vec<IndexEntry> = {
                let raw = std::fs::read(&index_path)?;
                serde_json::from_slice(&raw)
                    .map_err(|e| SubstringError::Serialization(e.to_string()))?
            };
            let blobs = if blobs_path.exists() {
                std::fs::read(&blobs_path)?
            } else {
                Vec::new()
            };

            for entry in entries {
                let start = entry.offset as usize;
                let end = start + entry.bytes as usize;
                if end > blobs.len() {
                    return Err(SubstringError::Serialization(format!(
                        "substring {} index entry overruns blobs.dat",
                        entry.info.id
                    )));
                }
                let tokens = decode_varint_tokens(&blobs[start..end])?;
                if tokens.len() >= MIN_LENGTH {
                    let h = hash_window(&tokens[..MIN_LENGTH]);
                    window_index.entry(h).or_default().push(entry.info.id);
                    cache.insert(entry.info.id, Arc::new(tokens));
                }
                max_seen_id = max_seen_id.max(entry.info.id);
                known.insert(entry.info.id, entry.info);
            }
        }

        let next_id = max_seen_id.checked_add(1).unwrap_or(1);

        Ok(Self {
            inner: Arc::new(RwLock::new(RegistryState {
                root: Some(root),
                cache,
                known,
                window_index,
                next_id,
            })),
        })
    }

    /// Register a new substring. Assigns the next sequential id.
    pub fn register(
        &self,
        tokens: Vec<Token>,
        source_occurrences: u32,
    ) -> SubstringResult<SubstringInfo> {
        if tokens.len() < MIN_LENGTH {
            return Err(SubstringError::TooShort(tokens.len()));
        }

        let mut state = self.inner.write().unwrap();
        let id = state.next_id;
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| SubstringError::Serialization("substring id space exhausted".into()))?;

        let info = SubstringInfo {
            id,
            length: tokens.len() as u32,
            created_at: Utc::now(),
            source_occurrences,
        };

        // Persist before mutating in-memory state.
        if let Some(root) = state.root.clone() {
            let blobs_path = root.join("blobs.dat");
            let tokens_bytes = encode_varint_tokens(&tokens);

            // Determine append offset by stat'ing the existing blobs.dat.
            let offset = if blobs_path.exists() {
                std::fs::metadata(&blobs_path)?.len()
            } else {
                0
            };

            // Append the blob.
            use std::io::Write;
            let mut blobs_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&blobs_path)?;
            blobs_file.write_all(&tokens_bytes)?;
            blobs_file.sync_data()?;

            // Build the new index entry and rewrite index.json. Rewriting
            // the whole index is fine — registries stay small (a few
            // thousand entries at most) and rewrites only happen during
            // training, not on the hot path.
            let new_entry = IndexEntry {
                info: info.clone(),
                offset,
                bytes: tokens_bytes.len() as u32,
            };
            let mut entries = read_index(&root)?;
            entries.push(new_entry);
            write_index(&root, &entries)?;
        }

        // Update window index for fast write-path matching.
        let h = hash_window(&tokens[..MIN_LENGTH]);
        state.window_index.entry(h).or_default().push(id);

        state.cache.insert(id, Arc::new(tokens));
        state.known.insert(id, info.clone());
        Ok(info)
    }

    /// Get the tokens for a known substring. The cache is populated
    /// eagerly at `open` and `register`, so this is a single map lookup
    /// in the common case.
    pub fn get_tokens(&self, id: SubstringId) -> SubstringResult<Arc<Vec<Token>>> {
        let state = self.inner.read().unwrap();
        state
            .cache
            .get(&id)
            .map(Arc::clone)
            .ok_or(SubstringError::NotFound(id))
    }

    /// Find the longest substring that matches the prefix of `tokens` at
    /// position 0. Returns `(id, length_consumed)` if any candidate
    /// matches; `None` otherwise.
    ///
    /// Lookup is O(candidates_in_window_bucket) — typically very small
    /// because the `MIN_LENGTH`-token rolling hash is highly selective.
    pub fn find_longest_match_at(&self, tokens: &[Token]) -> Option<(SubstringId, usize)> {
        if tokens.len() < MIN_LENGTH {
            return None;
        }
        let h = hash_window(&tokens[..MIN_LENGTH]);

        let state = self.inner.read().unwrap();
        let candidates = state.window_index.get(&h)?;

        let mut best: Option<(SubstringId, usize)> = None;
        for &id in candidates {
            // Use cached tokens directly; if missing, skip rather than
            // page in (the lookup is on a hot path; cold candidates
            // rarely beat the cached ones in practice).
            let Some(stored) = state.cache.get(&id) else {
                continue;
            };
            if tokens.len() < stored.len() {
                continue;
            }
            if &tokens[..stored.len()] != stored.as_slice() {
                continue;
            }
            if best.is_none_or(|(_, prev_len)| stored.len() > prev_len) {
                best = Some((id, stored.len()));
            }
        }
        best
    }

    /// All known substrings, sorted by id ascending.
    pub fn list(&self) -> Vec<SubstringInfo> {
        let state = self.inner.read().unwrap();
        let mut out: Vec<SubstringInfo> = state.known.values().cloned().collect();
        out.sort_by_key(|s| s.id);
        out
    }

    /// Number of registered substrings.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().known.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().known.is_empty()
    }

    /// Drop substrings whose ids are NOT in `keep`. Returns the number
    /// of dropped entries.
    ///
    /// On filesystem-backed registries the packed `blobs.dat` is
    /// rebuilt from scratch (with the kept substrings re-concatenated)
    /// and `index.json` is rewritten with new offsets. Atomicity is
    /// best-effort tmp-rename: a crash mid-rebuild leaves the old
    /// blobs.dat in place; the index.json rename is the commit point.
    pub fn retain(&self, keep: &std::collections::HashSet<SubstringId>) -> SubstringResult<u64> {
        let mut state = self.inner.write().unwrap();

        // First, decide what's going. Capture the target ids in
        // creation-time order so the rebuilt blobs.dat preserves
        // sequential layout.
        let dropping: Vec<SubstringId> = state
            .known
            .keys()
            .copied()
            .filter(|id| !keep.contains(id))
            .collect();

        if dropping.is_empty() {
            return Ok(0);
        }

        // Persist first when filesystem-backed.
        if let Some(root) = state.root.clone() {
            // Rebuild blobs.dat by re-encoding kept entries in order.
            let kept_ids: Vec<SubstringId> = {
                let mut ids: Vec<SubstringId> = state
                    .known
                    .keys()
                    .copied()
                    .filter(|id| keep.contains(id))
                    .collect();
                ids.sort();
                ids
            };

            let mut new_entries: Vec<IndexEntry> = Vec::with_capacity(kept_ids.len());
            let mut new_blobs: Vec<u8> = Vec::new();
            for id in &kept_ids {
                let info = state.known.get(id).expect("kept id present in known");
                let tokens = state
                    .cache
                    .get(id)
                    .ok_or(SubstringError::NotFound(*id))?
                    .clone();
                let bytes = encode_varint_tokens(&tokens);
                let offset = new_blobs.len() as u64;
                new_blobs.extend_from_slice(&bytes);
                new_entries.push(IndexEntry {
                    info: info.clone(),
                    offset,
                    bytes: bytes.len() as u32,
                });
            }

            // Atomic-ish rebuild: write tmp files, then rename.
            let blobs_tmp = root.join("blobs.dat.tmp");
            std::fs::write(&blobs_tmp, &new_blobs)?;
            std::fs::rename(&blobs_tmp, root.join("blobs.dat"))?;
            write_index(&root, &new_entries)?;
        }

        // Then mutate in-memory state.
        let dropped_count = dropping.len() as u64;
        for id in &dropping {
            // Drop from cache / known / window_index.
            if let Some(tokens) = state.cache.remove(id) {
                if tokens.len() >= MIN_LENGTH {
                    let h = hash_window(&tokens[..MIN_LENGTH]);
                    if let Some(bucket) = state.window_index.get_mut(&h) {
                        bucket.retain(|other| other != id);
                        if bucket.is_empty() {
                            state.window_index.remove(&h);
                        }
                    }
                }
            }
            state.known.remove(id);
        }
        Ok(dropped_count)
    }
}

// ── Window hash + varint helpers ──────────────────────────────────────────────

/// FNV-1a-ish rolling hash over a token window. Cheap and reasonably
/// well-distributed for our use case (we just need the bucket lookup to
/// be selective enough that the candidate list stays short).
pub fn hash_window(tokens: &[Token]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn read_index(root: &Path) -> SubstringResult<Vec<IndexEntry>> {
    let path = root.join("index.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read(&path)?;
    serde_json::from_slice(&raw).map_err(|e| SubstringError::Serialization(e.to_string()))
}

fn write_index(root: &Path, entries: &[IndexEntry]) -> SubstringResult<()> {
    let path = root.join("index.json");
    let tmp = root.join("index.json.tmp");
    let body = serde_json::to_vec_pretty(entries)
        .map_err(|e| SubstringError::Serialization(e.to_string()))?;
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn encode_varint_tokens(tokens: &[Token]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        write_varint_u32(t, &mut out);
    }
    out
}

fn decode_varint_tokens(bytes: &[u8]) -> SubstringResult<Vec<Token>> {
    let mut out = Vec::new();
    let mut cursor = std::io::Cursor::new(bytes);
    while (cursor.position() as usize) < bytes.len() {
        out.push(read_varint_u32(&mut cursor)?);
    }
    Ok(out)
}

fn write_varint_u32(mut value: u32, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint_u32(cursor: &mut std::io::Cursor<&[u8]>) -> SubstringResult<u32> {
    let mut shift: u32 = 0;
    let mut result: u32 = 0;
    loop {
        let mut byte = [0u8; 1];
        cursor
            .read_exact(&mut byte)
            .map_err(|e| SubstringError::Varint(format!("truncated: {e}")))?;
        let b = byte[0];
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return Err(SubstringError::Varint("varint overflows u32".into()));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn substring(seed: u32, length: usize) -> Vec<Token> {
        (0..length as u32)
            .map(|i| (seed * 17 + i * 3) % 50_000)
            .collect()
    }

    #[test]
    fn register_and_get_in_memory() {
        let r = SubstringRegistry::in_memory();
        let toks = substring(1, 32);
        let info = r.register(toks.clone(), 100).unwrap();
        assert_eq!(info.id, 1);
        assert_eq!(info.length, 32);
        assert_eq!(r.len(), 1);
        let recovered = r.get_tokens(info.id).unwrap();
        assert_eq!(recovered.as_slice(), toks.as_slice());
    }

    #[test]
    fn register_too_short_errors() {
        let r = SubstringRegistry::in_memory();
        let too_short = substring(1, MIN_LENGTH - 1);
        let err = r.register(too_short, 5).unwrap_err();
        assert!(matches!(err, SubstringError::TooShort(_)));
    }

    #[test]
    fn ids_are_sequential() {
        let r = SubstringRegistry::in_memory();
        let i1 = r.register(substring(1, 32), 10).unwrap().id;
        let i2 = r.register(substring(2, 32), 10).unwrap().id;
        let i3 = r.register(substring(3, 32), 10).unwrap().id;
        assert_eq!((i1, i2, i3), (1, 2, 3));
    }

    #[test]
    fn find_longest_match_at_simple() {
        let r = SubstringRegistry::in_memory();
        let s = substring(7, 50);
        let info = r.register(s.clone(), 10).unwrap();

        // Searching at a sequence that begins with s should find a match.
        let mut haystack = s.clone();
        haystack.extend_from_slice(&[99_999, 88_888]);
        let m = r.find_longest_match_at(&haystack);
        assert_eq!(m, Some((info.id, 50)));
    }

    #[test]
    fn find_longest_match_picks_longest() {
        let r = SubstringRegistry::in_memory();
        let short = substring(11, 20);
        // Make a longer substring whose first MIN_LENGTH tokens match `short`.
        let mut long = short.clone();
        long.extend((0..40u32).map(|i| (i + 1000) % 50_000));

        let i_short = r.register(short.clone(), 10).unwrap().id;
        let i_long = r.register(long.clone(), 5).unwrap().id;

        // At a position starting with `long`, the longer match should win.
        let m = r.find_longest_match_at(&long);
        assert_eq!(m, Some((i_long, long.len())));

        // At a position starting with `short` followed by garbage, the
        // shorter match should win.
        let mut mostly_short = short.clone();
        mostly_short.extend_from_slice(&[55_555, 66_666]);
        let m2 = r.find_longest_match_at(&mostly_short);
        assert_eq!(m2, Some((i_short, short.len())));
    }

    #[test]
    fn find_no_match_at_short_input() {
        let r = SubstringRegistry::in_memory();
        r.register(substring(1, 32), 10).unwrap();
        let short_input = vec![1u32; MIN_LENGTH - 1];
        assert_eq!(r.find_longest_match_at(&short_input), None);
    }

    #[test]
    fn find_no_match_at_unknown_window() {
        let r = SubstringRegistry::in_memory();
        r.register(substring(1, 32), 10).unwrap();
        let other: Vec<Token> = (90_000u32..90_032).collect();
        assert_eq!(r.find_longest_match_at(&other), None);
    }

    #[test]
    fn filesystem_persistence_round_trip() {
        let dir = TempDir::new().unwrap();
        let toks = substring(13, 64);
        let id;
        {
            let r = SubstringRegistry::open(dir.path()).unwrap();
            let info = r.register(toks.clone(), 25).unwrap();
            id = info.id;
        }

        // Reopen.
        let r2 = SubstringRegistry::open(dir.path()).unwrap();
        assert_eq!(r2.len(), 1);
        let recovered = r2.get_tokens(id).unwrap();
        assert_eq!(recovered.as_slice(), toks.as_slice());

        // The window index should also have been rebuilt — find_longest_match
        // needs it.
        let m = r2.find_longest_match_at(&toks);
        assert_eq!(m, Some((id, toks.len())));
    }

    #[test]
    fn next_id_continues_across_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let r = SubstringRegistry::open(dir.path()).unwrap();
            r.register(substring(1, 32), 10).unwrap();
            r.register(substring(2, 32), 10).unwrap();
        }
        let r2 = SubstringRegistry::open(dir.path()).unwrap();
        let info = r2.register(substring(3, 32), 10).unwrap();
        assert_eq!(info.id, 3, "next_id continuity broken across reopen");
    }
}
