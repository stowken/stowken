//! Python bindings for Stowken (PyO3 + maturin).
//!
//! Build:  maturin develop   (editable install)
//!         maturin build --release  (wheel)
//!
//! Usage:
//!   import stowken
//!   vault = stowken.Stowken()                       # in-memory
//!   vault = stowken.Stowken.open("/var/lib/sk")     # filesystem-backed
//!   result = vault.store({"model": "gpt-4", "messages": [...]})
//!   conv   = vault.retrieve(result.id)              # includes tokens

use std::sync::OnceLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;

extern crate stowken as sk;

use sk::{
    storage::{FilesystemBackend, MemoryBackend},
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

// ── Shared runtime ───────────────────────────────────────────────────────────

/// One Tokio runtime shared across every PyO3 call. Constructing a runtime
/// per-call (the previous behaviour) made every store/retrieve dominate-cost
/// even on local data.
fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        Runtime::new().expect("failed to construct shared tokio runtime")
    })
}

fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ── Python-visible result types ───────────────────────────────────────────────

/// Result returned by :meth:`Stowken.store`.
#[pyclass]
#[derive(Clone)]
pub struct StoreResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub total_segments: u64,
    #[pyo3(get)]
    pub new_segments: u64,
    #[pyo3(get)]
    pub deduped_segments: u64,
    #[pyo3(get)]
    pub bytes_saved: u64,
    #[pyo3(get)]
    pub compression_ratio: f64,
}

#[pymethods]
impl StoreResult {
    fn __repr__(&self) -> String {
        format!(
            "StoreResult(id={:?}, segments={}, new={}, deduped={}, bytes_saved={}, compression={:.2}x)",
            self.id, self.total_segments, self.new_segments,
            self.deduped_segments, self.bytes_saved, self.compression_ratio
        )
    }
}

/// Storage statistics returned by :meth:`Stowken.stats`.
#[pyclass]
#[derive(Clone)]
pub struct TokenUsageStats {
    #[pyo3(get)]
    pub total_tokens: u64,
    #[pyo3(get)]
    pub total_conversations: u64,
    #[pyo3(get)]
    pub unique_segments: u64,
    #[pyo3(get)]
    pub total_segments: u64,
    #[pyo3(get)]
    pub dedup_ratio: f64,
    #[pyo3(get)]
    pub compression_ratio: f64,
    #[pyo3(get)]
    pub storage_bytes: u64,
    #[pyo3(get)]
    pub savings_percentage: f64,
}

#[pymethods]
impl TokenUsageStats {
    fn __repr__(&self) -> String {
        format!(
            "TokenUsageStats(conversations={}, tokens={}, dedup={:.1}%, compression={:.2}x, storage={}B)",
            self.total_conversations, self.total_tokens,
            self.dedup_ratio * 100.0, self.compression_ratio, self.storage_bytes
        )
    }
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

fn parse_message(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<Message> {
    let role: String = obj.get_item("role")?.extract()?;

    let content = if let Ok(tokens_raw) = obj.get_item("tokens") {
        let tokens: Vec<u32> = tokens_raw.extract()?;
        MessageContent::Tokens(tokens)
    } else {
        let content_raw = obj.get_item("content")?;
        let text = if let Ok(s) = content_raw.extract::<String>() {
            s
        } else if let Ok(parts) = content_raw.downcast::<PyList>() {
            let mut combined = String::new();
            for part in parts.iter() {
                if let Ok(t) = part.get_item("text").and_then(|v| v.extract::<String>()) {
                    combined.push_str(&t);
                }
            }
            combined
        } else {
            String::new()
        };
        MessageContent::Text(text)
    };

    let name: Option<String> = obj.get_item("name").ok().and_then(|v| v.extract().ok());
    let tool_call_id: Option<String> =
        obj.get_item("tool_call_id").ok().and_then(|v| v.extract().ok());

    Ok(Message { role, content, name, tool_call_id })
}

fn parse_conversation(dict: &pyo3::Bound<'_, PyDict>) -> PyResult<Conversation> {
    let model: String = dict
        .get_item("model")?
        .ok_or_else(|| PyRuntimeError::new_err("conversation missing 'model'"))?
        .extract()?;

    let tokenizer: String = dict
        .get_item("tokenizer")?
        .and_then(|v| v.extract().ok())
        .unwrap_or_else(|| "cl100k_base".to_owned());

    let application: Option<String> =
        dict.get_item("application")?.and_then(|v| v.extract().ok());

    let id: Option<String> = dict.get_item("id")?.and_then(|v| v.extract().ok());

    let messages_raw = dict
        .get_item("messages")?
        .ok_or_else(|| PyRuntimeError::new_err("conversation missing 'messages'"))?;
    let messages_list = messages_raw.downcast::<PyList>()?;

    let mut messages = Vec::new();
    for item in messages_list.iter() {
        messages.push(parse_message(&item)?);
    }

    Ok(Conversation { id, model, tokenizer, application, metadata: None, messages })
}

// ── Backend variants ─────────────────────────────────────────────────────────

/// Wraps either an in-memory or filesystem-backed `Stowken`.
///
/// Held inside an `Arc` so we can run the inner future on the shared runtime
/// without lifetime juggling.
enum VaultKind {
    Memory(Stowken<MemoryBackend>),
    Filesystem(Stowken<FilesystemBackend>),
}

// ── Main Python class ─────────────────────────────────────────────────────────

/// Compressed, deduplicated storage for LLM token sequences.
///
/// Default constructor opens an ephemeral in-memory vault. For persistence
/// use :meth:`open` with a directory path.
///
/// Example::
///
///     import stowken
///
///     # Ephemeral
///     vault = stowken.Stowken()
///
///     # Persistent
///     vault = stowken.Stowken.open("/var/lib/stowken")
///
///     vault.store({
///         "model": "gpt-4",
///         "messages": [
///             {"role": "system",    "content": "You are helpful."},
///             {"role": "user",      "content": "Hello!"},
///             {"role": "assistant", "content": "Hi there!"},
///         ],
///     })
#[pyclass(name = "Stowken")]
pub struct PyStowken {
    inner: VaultKind,
}

#[pymethods]
impl PyStowken {
    /// Create a new in-memory vault.
    ///
    /// Args:
    ///     enable_compression (bool): Enable zstd compression. Default ``True``.
    #[new]
    #[pyo3(signature = (enable_compression=true))]
    fn new(_py: Python<'_>, enable_compression: bool) -> PyResult<Self> {
        let config = StowkenConfig { enable_compression };
        let backend = MemoryBackend::new();
        let inner = runtime()
            .block_on(Stowken::new(backend, config))
            .map_err(py_err)?;
        Ok(Self { inner: VaultKind::Memory(inner) })
    }

    /// Open a filesystem-backed vault rooted at `path`.
    ///
    /// Creates the directory and metadata database if they don't exist.
    /// Subsequent calls with the same path reopen the existing vault.
    ///
    /// Args:
    ///     path (str): Directory path for vault storage.
    ///     enable_compression (bool): Default ``True``.
    #[classmethod]
    #[pyo3(signature = (path, enable_compression=true))]
    fn open(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        _py: Python<'_>,
        path: String,
        enable_compression: bool,
    ) -> PyResult<Self> {
        let config = StowkenConfig { enable_compression };
        let db_path = format!("{}/metadata.db", path.trim_end_matches('/'));
        let inner = runtime()
            .block_on(async move {
                let backend = FilesystemBackend::new(&path).await.map_err(|e| e.to_string())?;
                Stowken::open(backend, config, &db_path)
                    .await
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner: VaultKind::Filesystem(inner) })
    }

    /// Store a conversation dict (OpenAI-compatible format).
    fn store(
        &self,
        _py: Python<'_>,
        conversation: &pyo3::Bound<'_, PyDict>,
    ) -> PyResult<StoreResult> {
        let conv = parse_conversation(conversation)?;
        let result = match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.store(conv)).map_err(py_err)?,
            VaultKind::Filesystem(v) => runtime().block_on(v.store(conv)).map_err(py_err)?,
        };
        Ok(StoreResult {
            id: result.id,
            total_segments: result.total_segments,
            new_segments: result.new_segments,
            deduped_segments: result.deduped_segments,
            bytes_saved: result.bytes_saved,
            compression_ratio: result.compression_ratio,
        })
    }

    /// Retrieve a conversation by ID.
    ///
    /// Returns:
    ///     dict: ``id``, ``model``, ``tokenizer``, ``total_tokens``,
    ///     and ``segments``. Each segment dict contains ``segment_type``,
    ///     ``token_count``, ``hash``, and ``tokens`` (the actual `u32`
    ///     token IDs as a Python list).
    fn retrieve(&self, py: Python<'_>, id: &str) -> PyResult<PyObject> {
        let retrieved = match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.retrieve(id)).map_err(py_err)?,
            VaultKind::Filesystem(v) => runtime().block_on(v.retrieve(id)).map_err(py_err)?,
        };

        let dict = PyDict::new_bound(py);
        dict.set_item("id", &retrieved.manifest.id)?;
        dict.set_item("model", &retrieved.manifest.model)?;
        dict.set_item("tokenizer", &retrieved.manifest.tokenizer)?;
        if let Some(app) = &retrieved.manifest.application {
            dict.set_item("application", app)?;
        }
        dict.set_item("total_tokens", retrieved.manifest.total_tokens)?;

        let segs_list = PyList::empty_bound(py);
        for seg in &retrieved.segments {
            let sd = PyDict::new_bound(py);
            sd.set_item("segment_type", seg.segment_type.to_string())?;
            sd.set_item("token_count", seg.token_count)?;
            sd.set_item("hash", &seg.hash.0)?;
            sd.set_item("tokens", &seg.tokens)?;
            segs_list.append(sd)?;
        }
        dict.set_item("segments", segs_list)?;

        Ok(dict.into())
    }

    /// Delete a conversation by ID.
    fn delete(&self, _py: Python<'_>, id: &str) -> PyResult<()> {
        match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.delete(id)).map_err(py_err),
            VaultKind::Filesystem(v) => runtime().block_on(v.delete(id)).map_err(py_err),
        }
    }

    /// Return aggregate storage statistics.
    fn stats(&self, _py: Python<'_>) -> PyResult<TokenUsageStats> {
        let s = match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.stats()).map_err(py_err)?,
            VaultKind::Filesystem(v) => runtime().block_on(v.stats()).map_err(py_err)?,
        };
        Ok(TokenUsageStats {
            total_tokens: s.total_tokens,
            total_conversations: s.total_conversations,
            unique_segments: s.unique_segments,
            total_segments: s.total_segments,
            dedup_ratio: s.dedup_ratio,
            compression_ratio: s.compression_ratio,
            storage_bytes: s.storage_bytes,
            savings_percentage: s.savings_percentage,
        })
    }

    /// Run garbage collection.
    ///
    /// Returns:
    ///     int: number of zero-reference segments deleted.
    fn gc(&self, _py: Python<'_>) -> PyResult<u64> {
        match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.gc()).map_err(py_err),
            VaultKind::Filesystem(v) => runtime().block_on(v.gc()).map_err(py_err),
        }
    }

    /// Rebuild the SQLite metadata index from on-disk manifests.
    ///
    /// Useful after restoring a vault from a backup that didn't include the
    /// index, or if you suspect drift. Returns a dict with
    /// ``conversations_indexed``, ``unique_segments_indexed``, and
    /// ``segments_missing``.
    fn reindex(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.reindex()).map_err(py_err)?,
            VaultKind::Filesystem(v) => runtime().block_on(v.reindex()).map_err(py_err)?,
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("conversations_indexed", report.conversations_indexed)?;
        dict.set_item("unique_segments_indexed", report.unique_segments_indexed)?;
        dict.set_item("segments_missing", report.segments_missing)?;
        Ok(dict.into())
    }

    /// List segment hashes currently eligible for GC, without deleting.
    fn gc_candidates(&self, py: Python<'_>) -> PyResult<PyObject> {
        let hashes = match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.gc_candidates()).map_err(py_err)?,
            VaultKind::Filesystem(v) => runtime().block_on(v.gc_candidates()).map_err(py_err)?,
        };
        let list = PyList::empty_bound(py);
        for h in hashes {
            list.append(h.0)?;
        }
        Ok(list.into())
    }

    /// Return per-segment-type statistics.
    fn segment_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = match &self.inner {
            VaultKind::Memory(v) => runtime().block_on(v.segment_stats()).map_err(py_err)?,
            VaultKind::Filesystem(v) => runtime().block_on(v.segment_stats()).map_err(py_err)?,
        };
        let list = PyList::empty_bound(py);
        for s in stats {
            let d = PyDict::new_bound(py);
            d.set_item("segment_type", s.segment_type.to_string())?;
            d.set_item("unique_count", s.unique_count)?;
            d.set_item("total_references", s.total_references)?;
            d.set_item("dedup_ratio", s.dedup_ratio)?;
            d.set_item("total_tokens", s.total_tokens)?;
            d.set_item("compressed_bytes", s.compressed_bytes)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        match self.stats(py) {
            Ok(s) => format!(
                "Stowken(conversations={}, unique_segments={})",
                s.total_conversations, s.unique_segments
            ),
            Err(_) => "Stowken()".to_owned(),
        }
    }
}

// ── Module registration ────────────────────────────────────────────────────────

#[pymodule]
fn stowken(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStowken>()?;
    m.add_class::<StoreResult>()?;
    m.add_class::<TokenUsageStats>()?;
    Ok(())
}
