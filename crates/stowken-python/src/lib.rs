//! Python bindings for Stowken (PyO3 + maturin).
//!
//! Build:  maturin develop   (editable install)
//!         maturin build --release  (wheel)
//!
//! Usage:
//!   import stowken
//!   vault = stowken.Stowken()
//!   result = vault.store({"model": "gpt-4", "messages": [...]})
//!   stats  = vault.stats()

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// Alias the dependency to avoid ambiguity with this lib's own name.
extern crate stowken as tv;

use tv::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

// ── Runtime ──────────────────────────────────────────────────────────────────

fn rt() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))
}

fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn block<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    py.allow_threads(f)
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

/// Encode a text string to u32 token IDs using UTF-8 byte pairs.
///
/// This is a simple fallback tokenizer: each pair of consecutive UTF-8 bytes
/// is packed into a u32. It is not equivalent to any real LLM tokenizer but
/// lets the vault function without an external tokenizer dependency.
fn parse_message(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<Message> {
    let role: String = obj.get_item("role")?.extract()?;

    // Prefer explicit token arrays; otherwise pass text through so the vault's
    // built-in tiktoken tokenizer handles it automatically.
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

// ── Main Python class ─────────────────────────────────────────────────────────

/// In-memory Stowken instance.
///
/// Example::
///
///     import stowken
///
///     vault = stowken.Stowken()
///     result = vault.store({
///         "model": "gpt-4",
///         "messages": [
///             {"role": "system",    "content": "You are helpful."},
///             {"role": "user",      "content": "Hello!"},
///             {"role": "assistant", "content": "Hi there!"},
///         ]
///     })
///     print(result)
///     stats = vault.stats()
///     print(f"Saved {stats.savings_percentage:.1f}%")
#[pyclass(name = "Stowken")]
pub struct PyStowken {
    inner: Stowken<MemoryBackend>,
}

#[pymethods]
impl PyStowken {
    /// Create a new in-memory vault.
    ///
    /// Args:
    ///     enable_compression (bool): Enable zstd compression. Default ``True``.
    #[new]
    #[pyo3(signature = (enable_compression=true))]
    fn new(py: Python<'_>, enable_compression: bool) -> PyResult<Self> {
        let config = StowkenConfig { enable_compression, dictionary_path: None, tenant_id: None, ..StowkenConfig::default() };
        let backend = MemoryBackend::new();
        let inner = block(py, || {
            rt()?.block_on(Stowken::new(backend, config)).map_err(py_err)
        })?;
        Ok(Self { inner })
    }

    /// Store a conversation dict (OpenAI-compatible format).
    ///
    /// Args:
    ///     conversation (dict): Must include ``model`` (str) and ``messages`` (list).
    ///         Optional keys: ``tokenizer``, ``application``, ``id``.
    ///
    /// Returns:
    ///     StoreResult
    fn store(&self, py: Python<'_>, conversation: &pyo3::Bound<'_, PyDict>) -> PyResult<StoreResult> {
        let conv = parse_conversation(conversation)?;
        let result = block(py, || {
            rt()?.block_on(self.inner.store(conv)).map_err(py_err)
        })?;
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
    ///     dict: ``id``, ``model``, ``tokenizer``, ``total_tokens``, ``segments``
    fn retrieve(&self, py: Python<'_>, id: &str) -> PyResult<PyObject> {
        let retrieved = block(py, || {
            rt()?.block_on(self.inner.retrieve(id)).map_err(py_err)
        })?;

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
            segs_list.append(sd)?;
        }
        dict.set_item("segments", segs_list)?;

        Ok(dict.into())
    }

    /// Delete a conversation by ID.
    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<()> {
        block(py, || rt()?.block_on(self.inner.delete(id)).map_err(py_err))
    }

    /// Return aggregate storage statistics.
    ///
    /// Returns:
    ///     TokenUsageStats
    fn stats(&self, py: Python<'_>) -> PyResult<TokenUsageStats> {
        let s = block(py, || rt()?.block_on(self.inner.stats()).map_err(py_err))?;
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
    ///     int: number of zero-reference segments deleted
    fn gc(&self, py: Python<'_>) -> PyResult<u64> {
        block(py, || rt()?.block_on(self.inner.gc()).map_err(py_err))
    }

    /// Return per-segment-type statistics.
    ///
    /// Returns:
    ///     list[dict]: each dict has ``segment_type``, ``unique_count``,
    ///     ``total_references``, ``dedup_ratio``, ``total_tokens``, ``compressed_bytes``
    fn segment_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats =
            block(py, || rt()?.block_on(self.inner.segment_stats()).map_err(py_err))?;
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
