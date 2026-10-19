//! Per-vault registry of zstd compression dictionaries.
//!
//! # Why a registry
//!
//! In v0.1, applying a new dictionary replaced the old one in memory. Frames
//! compressed with the old dict were no longer decompressible — silent data
//! loss on every rotation. The fix at the format level (frame byte `0x03`
//! carrying a 4-byte dict_id) means each frame names its own dictionary;
//! the decompressor selects by ID. Rotation becomes safe because old IDs
//! stay valid as long as the dict file stays on disk.
//!
//! # Layout
//!
//! For a filesystem-backed vault with dictionaries enabled, the on-disk
//! layout under `<vault-root>` is:
//!
//! ```text
//! dictionaries/
//! ├── 1234567890.zstd     # raw dictionary bytes (matches dict_id)
//! ├── 1234567890.json     # DictInfo: created_at, sample_count, size
//! ├── 9876543210.zstd
//! ├── 9876543210.json
//! └── active              # one line: "9876543210"
//! ```
//!
//! Memory-backed vaults skip persistence — the registry exists in RAM only.
//!
//! # Concurrency
//!
//! All state lives behind a single `RwLock`. Reads (decompression) are
//! frequent and parallel; writes (registration, activation) are
//! administrative and rare. The byte cache is `Arc<Vec<u8>>` so reads can
//! hand out cheap references without holding the lock.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 32-bit zstd dictionary identifier. Embedded by the zstd dict-builder when
/// training; we use it directly as the registry key so the frame's
/// dict_id field matches what zstd itself records.
pub type DictId = u32;

/// Errors from registry operations.
#[derive(Debug, Error)]
pub enum DictError {
    #[error("dictionary I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dictionary {0} not found in registry")]
    NotFound(DictId),
    #[error("dictionary {0} is already registered")]
    DuplicateId(DictId),
    #[error("dictionary bytes are missing the zstd dict header magic")]
    InvalidDictBytes,
    #[error("dictionary bytes have no embedded ID and one was not supplied")]
    NoEmbeddedId,
    #[error("dictionary {0} has no info sidecar")]
    MissingInfo(DictId),
    #[error("registry serialization error: {0}")]
    Serialization(String),
}

pub type DictResult<T> = Result<T, DictError>;

/// Public-facing description of one registered dictionary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictInfo {
    pub id: DictId,
    pub created_at: DateTime<Utc>,
    pub sample_count: u32,
    pub size_bytes: u64,
    /// True if this is the dictionary the registry will use for new compressions.
    pub is_active: bool,
}

/// Shared registry of dictionaries for one vault.
#[derive(Clone)]
pub struct DictRegistry {
    inner: Arc<RwLock<RegistryState>>,
}

struct RegistryState {
    /// Where dict files live. `None` means in-memory only.
    root: Option<PathBuf>,
    /// id → bytes (lazy: filesystem-backed registries populate on first use).
    cache: HashMap<DictId, Arc<Vec<u8>>>,
    /// Inventory of all known dicts (info sidecars).
    known: HashMap<DictId, DictInfo>,
    /// The dict marked as the writer for new compressions.
    active: Option<DictId>,
}

impl DictRegistry {
    /// Construct an in-memory registry (no persistence).
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryState {
                root: None,
                cache: HashMap::new(),
                known: HashMap::new(),
                active: None,
            })),
        }
    }

    /// Construct a filesystem-backed registry rooted at `root`.
    /// Creates the directory if it doesn't exist and loads any existing
    /// `*.json` info sidecars + the `active` marker.
    pub fn open<P: AsRef<Path>>(root: P) -> DictResult<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;

        let mut known: HashMap<DictId, DictInfo> = HashMap::new();
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let info: DictInfo = serde_json::from_slice(&raw)
                .map_err(|e| DictError::Serialization(e.to_string()))?;
            known.insert(info.id, info);
        }

        let active_path = root.join("active");
        let active = if active_path.exists() {
            let raw = std::fs::read_to_string(&active_path)?;
            raw.trim()
                .parse::<DictId>()
                .ok()
                .filter(|id| known.contains_key(id))
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(RegistryState {
                root: Some(root),
                cache: HashMap::new(),
                known,
                active,
            })),
        })
    }

    /// Register a new dictionary. The bytes must include a real (non-zero)
    /// embedded zstd dict_id; otherwise `NoEmbeddedId` is returned.
    /// Does NOT activate — call `activate` afterwards if desired.
    pub fn register(&self, bytes: Vec<u8>, sample_count: u32) -> DictResult<DictInfo> {
        let id = extract_dict_id(&bytes)?;

        let mut state = self.inner.write().unwrap();
        if state.known.contains_key(&id) {
            return Err(DictError::DuplicateId(id));
        }

        let info = DictInfo {
            id,
            created_at: Utc::now(),
            sample_count,
            size_bytes: bytes.len() as u64,
            is_active: false,
        };

        // Persist before mutating in-memory state.
        if let Some(root) = state.root.clone() {
            std::fs::write(root.join(format!("{id}.zstd")), &bytes)?;
            let info_json = serde_json::to_vec_pretty(&info)
                .map_err(|e| DictError::Serialization(e.to_string()))?;
            std::fs::write(root.join(format!("{id}.json")), info_json)?;
        }

        state.cache.insert(id, Arc::new(bytes));
        state.known.insert(id, info.clone());
        Ok(info)
    }

    /// Mark `id` as the dictionary new compressions will use. The dict must
    /// already be registered.
    pub fn activate(&self, id: DictId) -> DictResult<()> {
        let mut state = self.inner.write().unwrap();
        if !state.known.contains_key(&id) {
            return Err(DictError::NotFound(id));
        }

        // Flip the is_active flag in the inventory.
        for info in state.known.values_mut() {
            info.is_active = info.id == id;
        }
        state.active = Some(id);

        if let Some(root) = state.root.clone() {
            // Persist the new active marker first.
            std::fs::write(root.join("active"), format!("{id}\n"))?;
            // Rewrite info sidecars so on-disk state stays consistent. The
            // is_active flag is denormalised but cheap to keep current and
            // useful for `stowken dict list` after a reopen.
            let snapshot: Vec<DictInfo> = state.known.values().cloned().collect();
            for info in snapshot {
                let info_path = root.join(format!("{}.json", info.id));
                let info_json = serde_json::to_vec_pretty(&info)
                    .map_err(|e| DictError::Serialization(e.to_string()))?;
                std::fs::write(info_path, info_json)?;
            }
        }
        Ok(())
    }

    /// Active dict ID, if any.
    pub fn active_id(&self) -> Option<DictId> {
        self.inner.read().unwrap().active
    }

    /// Get the bytes for a known dict, faulting in from disk on cache miss.
    pub fn get_bytes(&self, id: DictId) -> DictResult<Arc<Vec<u8>>> {
        // Fast path: cache hit under a read lock.
        {
            let state = self.inner.read().unwrap();
            if let Some(bytes) = state.cache.get(&id) {
                return Ok(Arc::clone(bytes));
            }
        }

        // Slow path: load from disk and insert.
        let mut state = self.inner.write().unwrap();
        if let Some(bytes) = state.cache.get(&id) {
            return Ok(Arc::clone(bytes));
        }
        if !state.known.contains_key(&id) {
            return Err(DictError::NotFound(id));
        }
        let root = state.root.as_ref().ok_or(DictError::NotFound(id))?.clone();
        let bytes = std::fs::read(root.join(format!("{id}.zstd")))?;
        let arc = Arc::new(bytes);
        state.cache.insert(id, Arc::clone(&arc));
        Ok(arc)
    }

    /// All known dicts, sorted by creation time ascending.
    pub fn list(&self) -> Vec<DictInfo> {
        let state = self.inner.read().unwrap();
        let mut out: Vec<DictInfo> = state.known.values().cloned().collect();
        out.sort_by_key(|d| d.created_at);
        out
    }

    /// Number of distinct dicts in the registry.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().known.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().known.is_empty()
    }
}

/// Read the embedded dict_id from a zstd dictionary.
///
/// Per RFC 8478, a dictionary starts with magic `0xEC30A437` (LE), followed
/// by a 4-byte little-endian `dict_id`. ID 0 means "no dict-ID, manually
/// managed" — the registry rejects these to avoid ambiguous lookups.
fn extract_dict_id(bytes: &[u8]) -> DictResult<DictId> {
    if bytes.len() < 8 {
        return Err(DictError::InvalidDictBytes);
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0xEC30_A437 {
        return Err(DictError::InvalidDictBytes);
    }
    let id = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if id == 0 {
        return Err(DictError::NoEmbeddedId);
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::varint;
    use tempfile::TempDir;

    /// Train a minimal real zstd dict from synthetic samples. Used to exercise
    /// the registry with bytes that have a real embedded dict_id.
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
    fn extract_id_from_real_dict() {
        let dict = train_dict(7);
        let id = extract_dict_id(&dict).unwrap();
        assert!(id != 0, "trained dict should have a non-zero embedded id");
    }

    #[test]
    fn extract_id_rejects_garbage() {
        assert!(matches!(
            extract_dict_id(b"not a real dict"),
            Err(DictError::InvalidDictBytes)
        ));
    }

    #[test]
    fn in_memory_register_get_activate() {
        let reg = DictRegistry::in_memory();
        let dict = train_dict(11);
        let info = reg.register(dict.clone(), 30).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.active_id().is_none());

        reg.activate(info.id).unwrap();
        assert_eq!(reg.active_id(), Some(info.id));

        let bytes = reg.get_bytes(info.id).unwrap();
        assert_eq!(bytes.as_slice(), dict.as_slice());
    }

    #[test]
    fn duplicate_register_errors() {
        let reg = DictRegistry::in_memory();
        let dict = train_dict(13);
        reg.register(dict.clone(), 30).unwrap();
        let err = reg.register(dict, 30).unwrap_err();
        assert!(matches!(err, DictError::DuplicateId(_)));
    }

    #[test]
    fn activate_unknown_errors() {
        let reg = DictRegistry::in_memory();
        let err = reg.activate(42).unwrap_err();
        assert!(matches!(err, DictError::NotFound(42)));
    }

    #[test]
    fn filesystem_persistence_round_trip() {
        let dir = TempDir::new().unwrap();
        let dict = train_dict(17);
        let id;

        {
            let reg = DictRegistry::open(dir.path()).unwrap();
            let info = reg.register(dict.clone(), 30).unwrap();
            reg.activate(info.id).unwrap();
            id = info.id;
        }

        // Reopen — registry should rediscover the dict and the active marker.
        let reg2 = DictRegistry::open(dir.path()).unwrap();
        assert_eq!(reg2.len(), 1);
        assert_eq!(reg2.active_id(), Some(id));
        let bytes = reg2.get_bytes(id).unwrap();
        assert_eq!(bytes.as_slice(), dict.as_slice());
    }

    #[test]
    fn multiple_dicts_coexist() {
        let dir = TempDir::new().unwrap();
        let reg = DictRegistry::open(dir.path()).unwrap();

        let d1 = train_dict(19);
        let d2 = train_dict(23);
        let i1 = reg.register(d1.clone(), 30).unwrap().id;
        let i2 = reg.register(d2.clone(), 30).unwrap().id;
        assert_ne!(
            i1, i2,
            "different training corpora should produce different ids"
        );

        reg.activate(i2).unwrap();

        // Both dicts must remain readable; activation only affects writes.
        assert_eq!(reg.get_bytes(i1).unwrap().as_slice(), d1.as_slice());
        assert_eq!(reg.get_bytes(i2).unwrap().as_slice(), d2.as_slice());
        assert_eq!(reg.active_id(), Some(i2));
    }
}
