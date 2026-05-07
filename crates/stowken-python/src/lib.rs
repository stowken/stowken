//! Python bindings for Stowken (PyO3 + maturin).
//!
//! Build:  maturin develop   (editable install)
//!         maturin build --release  (wheel)
//!
//! Each I/O method is exposed in two flavours:
//!   `vault.foo(...)`        — blocking; runs the future on a shared runtime.
//!   `vault.foo_async(...)`  — returns a Python awaitable; safe inside an
//!                             event loop (FastAPI, asyncio workers, etc.).
//!
//! Both flavours share a single Tokio runtime, so mixing them in one process
//! is fine.

use std::sync::{Arc, OnceLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;

extern crate stowken as sk;

use sk::{
    storage::{s3::S3Backend, FilesystemBackend, MemoryBackend},
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

// ── Shared runtime ───────────────────────────────────────────────────────────

/// One Tokio runtime shared across every PyO3 call (sync and async). Calling
/// `block_on` inside an async future would deadlock, so the sync methods
/// only ever use this on a thread that doesn't have an active runtime.
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

// ── Result-shaping helpers (used by both sync and async paths) ───────────────

fn shape_store_result(r: sk::types::StoreResult) -> StoreResult {
    StoreResult {
        id: r.id,
        total_segments: r.total_segments,
        new_segments: r.new_segments,
        deduped_segments: r.deduped_segments,
        bytes_saved: r.bytes_saved,
        compression_ratio: r.compression_ratio,
    }
}

fn shape_token_stats(s: sk::types::TokenUsageStats) -> TokenUsageStats {
    TokenUsageStats {
        total_tokens: s.total_tokens,
        total_conversations: s.total_conversations,
        unique_segments: s.unique_segments,
        total_segments: s.total_segments,
        dedup_ratio: s.dedup_ratio,
        compression_ratio: s.compression_ratio,
        storage_bytes: s.storage_bytes,
        savings_percentage: s.savings_percentage,
    }
}

fn build_retrieve_dict(
    py: Python<'_>,
    retrieved: &sk::types::RetrievedConversation,
) -> PyResult<PyObject> {
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

fn build_conversation_text(
    py: Python<'_>,
    ct: &sk::types::ConversationText,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("conversation_id", &ct.conversation_id)?;
    dict.set_item("model", &ct.model)?;
    if let Some(app) = &ct.application {
        dict.set_item("application", app)?;
    }
    dict.set_item("text", &ct.text)?;
    dict.set_item("created_at", ct.created_at.to_rfc3339())?;
    let turns = PyList::empty_bound(py);
    for t in &ct.turns {
        let td = PyDict::new_bound(py);
        td.set_item("role", &t.role)?;
        td.set_item("text", &t.text)?;
        turns.append(td)?;
    }
    dict.set_item("turns", turns)?;
    Ok(dict.into())
}

fn build_segment_stats_list(
    py: Python<'_>,
    stats: &[sk::types::SegmentTypeStats],
) -> PyResult<PyObject> {
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

fn build_reindex_dict(
    py: Python<'_>,
    report: &sk::types::ReindexStats,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("conversations_indexed", report.conversations_indexed)?;
    dict.set_item("unique_segments_indexed", report.unique_segments_indexed)?;
    dict.set_item("segments_missing", report.segments_missing)?;
    Ok(dict.into())
}

fn build_hash_list(py: Python<'_>, hashes: &[sk::types::SegmentHash]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for h in hashes {
        list.append(&h.0)?;
    }
    Ok(list.into())
}

// ── Backend variants ─────────────────────────────────────────────────────────

/// Wraps an in-memory, filesystem-backed, or S3-backed `Stowken`.
enum VaultKind {
    Memory(Stowken<MemoryBackend>),
    Filesystem(Stowken<FilesystemBackend>),
    S3(Stowken<S3Backend>),
}

// ── Per-method dispatch helpers ──────────────────────────────────────────────
//
// Each helper takes a reference to VaultKind and runs the inner `Stowken`
// method asynchronously, returning the raw Rust result. The sync wrappers
// use `runtime().block_on(...)` and the async wrappers feed the future to
// `pyo3_async_runtimes::tokio::future_into_py`.

async fn vault_store(
    inner: &VaultKind,
    conv: Conversation,
) -> Result<sk::types::StoreResult, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.store(conv).await,
        VaultKind::Filesystem(v) => v.store(conv).await,
        VaultKind::S3(v) => v.store(conv).await,
    }
}

async fn vault_retrieve(
    inner: &VaultKind,
    id: &str,
) -> Result<sk::types::RetrievedConversation, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.retrieve(id).await,
        VaultKind::Filesystem(v) => v.retrieve(id).await,
        VaultKind::S3(v) => v.retrieve(id).await,
    }
}

async fn vault_retrieve_batch(
    inner: &VaultKind,
    ids: &[&str],
) -> Result<Vec<sk::types::ConversationText>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.retrieve_batch(ids).await,
        VaultKind::Filesystem(v) => v.retrieve_batch(ids).await,
        VaultKind::S3(v) => v.retrieve_batch(ids).await,
    }
}

async fn vault_delete(inner: &VaultKind, id: &str) -> Result<(), sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.delete(id).await,
        VaultKind::Filesystem(v) => v.delete(id).await,
        VaultKind::S3(v) => v.delete(id).await,
    }
}

async fn vault_stats(
    inner: &VaultKind,
) -> Result<sk::types::TokenUsageStats, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.stats().await,
        VaultKind::Filesystem(v) => v.stats().await,
        VaultKind::S3(v) => v.stats().await,
    }
}

async fn vault_gc(inner: &VaultKind) -> Result<u64, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.gc().await,
        VaultKind::Filesystem(v) => v.gc().await,
        VaultKind::S3(v) => v.gc().await,
    }
}

async fn vault_gc_candidates(
    inner: &VaultKind,
) -> Result<Vec<sk::types::SegmentHash>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.gc_candidates().await,
        VaultKind::Filesystem(v) => v.gc_candidates().await,
        VaultKind::S3(v) => v.gc_candidates().await,
    }
}

async fn vault_segment_stats(
    inner: &VaultKind,
) -> Result<Vec<sk::types::SegmentTypeStats>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.segment_stats().await,
        VaultKind::Filesystem(v) => v.segment_stats().await,
        VaultKind::S3(v) => v.segment_stats().await,
    }
}

async fn vault_reindex(
    inner: &VaultKind,
) -> Result<sk::types::ReindexStats, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.reindex().await,
        VaultKind::Filesystem(v) => v.reindex().await,
        VaultKind::S3(v) => v.reindex().await,
    }
}

async fn vault_train_substrings(
    inner: &VaultKind,
    samples: usize,
    min_occurrences: u32,
) -> Result<Vec<sk::substring_registry::SubstringInfo>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.train_substrings(samples, min_occurrences).await,
        VaultKind::Filesystem(v) => v.train_substrings(samples, min_occurrences).await,
        VaultKind::S3(v) => v.train_substrings(samples, min_occurrences).await,
    }
}

async fn vault_compact_substrings(
    inner: &VaultKind,
) -> Result<sk::types::SubstringCompactStats, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.compact_substrings().await,
        VaultKind::Filesystem(v) => v.compact_substrings().await,
        VaultKind::S3(v) => v.compact_substrings().await,
    }
}

async fn vault_gc_substrings(
    inner: &VaultKind,
) -> Result<sk::types::SubstringGcStats, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.gc_substrings().await,
        VaultKind::Filesystem(v) => v.gc_substrings().await,
        VaultKind::S3(v) => v.gc_substrings().await,
    }
}

fn vault_list_substrings(inner: &VaultKind) -> Vec<sk::substring_registry::SubstringInfo> {
    match inner {
        VaultKind::Memory(v) => v.list_substrings(),
        VaultKind::Filesystem(v) => v.list_substrings(),
        VaultKind::S3(v) => v.list_substrings(),
    }
}

fn build_substring_info_dict(
    py: Python<'_>,
    info: &sk::substring_registry::SubstringInfo,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", info.id)?;
    dict.set_item("length", info.length)?;
    dict.set_item("created_at", info.created_at.to_rfc3339())?;
    dict.set_item("source_occurrences", info.source_occurrences)?;
    Ok(dict.into())
}

fn build_substring_info_list(
    py: Python<'_>,
    infos: &[sk::substring_registry::SubstringInfo],
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for info in infos {
        list.append(build_substring_info_dict(py, info)?)?;
    }
    Ok(list.into())
}

async fn vault_find_near_duplicates(
    inner: &VaultKind,
    threshold: f64,
    min_refs: u64,
    limit: u64,
) -> Result<Vec<sk::types::NearDuplicateCluster>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.find_near_duplicates(threshold, min_refs, limit).await,
        VaultKind::Filesystem(v) => v.find_near_duplicates(threshold, min_refs, limit).await,
        VaultKind::S3(v) => v.find_near_duplicates(threshold, min_refs, limit).await,
    }
}

async fn vault_near_dedup_index_size(
    inner: &VaultKind,
) -> Result<u64, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.near_dedup_index_size().await,
        VaultKind::Filesystem(v) => v.near_dedup_index_size().await,
        VaultKind::S3(v) => v.near_dedup_index_size().await,
    }
}

fn build_near_dup_clusters(
    py: Python<'_>,
    clusters: &[sk::types::NearDuplicateCluster],
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for cluster in clusters {
        let d = PyDict::new_bound(py);
        d.set_item("canonical", &cluster.canonical.0)?;
        d.set_item("canonical_token_count", cluster.canonical_token_count)?;
        d.set_item("canonical_ref_count", cluster.canonical_ref_count)?;
        let variants = PyList::empty_bound(py);
        for v in &cluster.variants {
            let vd = PyDict::new_bound(py);
            vd.set_item("hash", &v.hash.0)?;
            vd.set_item("similarity", v.similarity)?;
            vd.set_item("token_count", v.token_count)?;
            vd.set_item("ref_count", v.ref_count)?;
            variants.append(vd)?;
        }
        d.set_item("variants", variants)?;
        list.append(d)?;
    }
    Ok(list.into())
}

async fn vault_train_dict(
    inner: &VaultKind,
    samples: usize,
) -> Result<sk::dict_registry::DictInfo, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.train_dictionary(samples).await,
        VaultKind::Filesystem(v) => v.train_dictionary(samples).await,
        VaultKind::S3(v) => v.train_dictionary(samples).await,
    }
}

async fn vault_train_and_activate_dict(
    inner: &VaultKind,
    samples: usize,
) -> Result<sk::dict_registry::DictInfo, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.train_and_activate_dictionary(samples).await,
        VaultKind::Filesystem(v) => v.train_and_activate_dictionary(samples).await,
        VaultKind::S3(v) => v.train_and_activate_dictionary(samples).await,
    }
}

async fn vault_activate_dict(
    inner: &VaultKind,
    dict_id: u32,
) -> Result<(), sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.activate_dictionary(dict_id).await,
        VaultKind::Filesystem(v) => v.activate_dictionary(dict_id).await,
        VaultKind::S3(v) => v.activate_dictionary(dict_id).await,
    }
}

fn vault_list_dicts(inner: &VaultKind) -> Vec<sk::dict_registry::DictInfo> {
    match inner {
        VaultKind::Memory(v) => v.list_dictionaries(),
        VaultKind::Filesystem(v) => v.list_dictionaries(),
        VaultKind::S3(v) => v.list_dictionaries(),
    }
}

fn build_dict_info_dict(
    py: Python<'_>,
    info: &sk::dict_registry::DictInfo,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", info.id)?;
    dict.set_item("created_at", info.created_at.to_rfc3339())?;
    dict.set_item("sample_count", info.sample_count)?;
    dict.set_item("size_bytes", info.size_bytes)?;
    dict.set_item("is_active", info.is_active)?;
    Ok(dict.into())
}

fn build_dict_info_list(
    py: Python<'_>,
    infos: &[sk::dict_registry::DictInfo],
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for info in infos {
        let d = build_dict_info_dict(py, info)?;
        list.append(d)?;
    }
    Ok(list.into())
}

// ── Semantic search bridges (feature: semantic-search) ──────────────────────

#[cfg(feature = "semantic-search")]
mod semantic_bridge {
    use super::*;

    /// Bridges a Python callable (``Callable[[list[str]], list[list[float]]]``)
    /// into Stowken's ``EmbeddingAdapter`` trait. The callable is invoked
    /// inside ``Python::with_gil`` from the spawn_blocking thread that
    /// `Stowken::semantic_search` uses, so this is safe even when called
    /// from an async context.
    pub struct PyEmbeddingAdapter {
        pub callback: PyObject,
        pub dim: usize,
        pub model: String,
    }

    impl sk::types::EmbeddingAdapter for PyEmbeddingAdapter {
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
            Python::with_gil(|py| {
                let list = PyList::new_bound(py, texts);
                let result = self
                    .callback
                    .call1(py, (list,))
                    .map_err(|e| format!("python embedding callback: {e}"))?;
                let rows: Vec<Vec<f32>> = result
                    .extract(py)
                    .map_err(|e| format!("decoding embedding result: {e}"))?;
                Ok(rows)
            })
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn model_name(&self) -> &str {
            &self.model
        }
    }

    /// Bridges a Python callable (``Callable[[str], str]``) into Stowken's
    /// ``SummarizerAdapter`` trait.
    pub struct PySummarizerAdapter {
        pub callback: PyObject,
        pub model: String,
    }

    impl sk::types::SummarizerAdapter for PySummarizerAdapter {
        fn summarize(&self, text: &str) -> Result<String, String> {
            Python::with_gil(|py| {
                let result = self
                    .callback
                    .call1(py, (text,))
                    .map_err(|e| format!("python summarizer callback: {e}"))?;
                result
                    .extract(py)
                    .map_err(|e| format!("decoding summarizer result: {e}"))
            })
        }
        fn model_name(&self) -> &str {
            &self.model
        }
    }
}

#[cfg(feature = "semantic-search")]
fn vault_set_embedding_adapter(
    inner: &VaultKind,
    adapter: semantic_bridge::PyEmbeddingAdapter,
) {
    match inner {
        VaultKind::Memory(v) => v.set_embedding_adapter(adapter),
        VaultKind::Filesystem(v) => v.set_embedding_adapter(adapter),
        VaultKind::S3(v) => v.set_embedding_adapter(adapter),
    }
}

#[cfg(feature = "semantic-search")]
fn vault_set_summarizer(
    inner: &VaultKind,
    adapter: semantic_bridge::PySummarizerAdapter,
) {
    match inner {
        VaultKind::Memory(v) => v.set_summarizer(adapter),
        VaultKind::Filesystem(v) => v.set_summarizer(adapter),
        VaultKind::S3(v) => v.set_summarizer(adapter),
    }
}

#[cfg(feature = "semantic-search")]
fn vault_use_concat_summary(inner: &VaultKind, max_chars: usize) {
    match inner {
        VaultKind::Memory(v) => v.use_concat_summary(max_chars),
        VaultKind::Filesystem(v) => v.use_concat_summary(max_chars),
        VaultKind::S3(v) => v.use_concat_summary(max_chars),
    }
}

#[cfg(feature = "semantic-search")]
fn vault_set_embed_on_store(inner: &VaultKind, enabled: bool) {
    match inner {
        VaultKind::Memory(v) => v.set_embed_on_store(enabled),
        VaultKind::Filesystem(v) => v.set_embed_on_store(enabled),
        VaultKind::S3(v) => v.set_embed_on_store(enabled),
    }
}

#[cfg(feature = "semantic-search")]
async fn vault_embed_all(
    inner: &VaultKind,
) -> Result<sk::types::EmbedStats, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.embed_all().await,
        VaultKind::Filesystem(v) => v.embed_all().await,
        VaultKind::S3(v) => v.embed_all().await,
    }
}

#[cfg(feature = "semantic-search")]
async fn vault_semantic_search(
    inner: &VaultKind,
    q: sk::types::SemanticSearchQuery,
) -> Result<Vec<sk::types::SemanticSearchHit>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.semantic_search(q).await,
        VaultKind::Filesystem(v) => v.semantic_search(q).await,
        VaultKind::S3(v) => v.semantic_search(q).await,
    }
}

#[cfg(feature = "semantic-search")]
async fn vault_typed_search(
    inner: &VaultKind,
    kind: &str,
    text: &str,
    limit: usize,
) -> Result<Vec<sk::types::SemanticSearchHit>, sk::vault::StowkenError> {
    match (inner, kind) {
        (VaultKind::Memory(v), "prompts") => v.search_prompts(text, limit).await,
        (VaultKind::Memory(v), "responses") => v.search_responses(text, limit).await,
        (VaultKind::Memory(v), "user_questions") => v.search_user_questions(text, limit).await,
        (VaultKind::Memory(v), "tool_uses") => v.search_tool_uses(text, limit).await,
        (VaultKind::Memory(v), "conversations") => v.search_conversations(text, limit).await,
        (VaultKind::Filesystem(v), "prompts") => v.search_prompts(text, limit).await,
        (VaultKind::Filesystem(v), "responses") => v.search_responses(text, limit).await,
        (VaultKind::Filesystem(v), "user_questions") => v.search_user_questions(text, limit).await,
        (VaultKind::Filesystem(v), "tool_uses") => v.search_tool_uses(text, limit).await,
        (VaultKind::Filesystem(v), "conversations") => v.search_conversations(text, limit).await,
        (VaultKind::S3(v), "prompts") => v.search_prompts(text, limit).await,
        (VaultKind::S3(v), "responses") => v.search_responses(text, limit).await,
        (VaultKind::S3(v), "user_questions") => v.search_user_questions(text, limit).await,
        (VaultKind::S3(v), "tool_uses") => v.search_tool_uses(text, limit).await,
        (VaultKind::S3(v), "conversations") => v.search_conversations(text, limit).await,
        _ => Err(sk::vault::StowkenError::Internal(format!(
            "unknown typed search kind '{kind}'"
        ))),
    }
}

#[cfg(feature = "semantic-search")]
async fn vault_cluster(
    inner: &VaultKind,
    k: usize,
    representatives: usize,
) -> Result<Vec<sk::types::ConversationCluster>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.cluster_conversations(k, representatives).await,
        VaultKind::Filesystem(v) => v.cluster_conversations(k, representatives).await,
        VaultKind::S3(v) => v.cluster_conversations(k, representatives).await,
    }
}

#[cfg(feature = "semantic-search")]
async fn vault_outliers(
    inner: &VaultKind,
    k: usize,
    threshold: f32,
    limit: usize,
) -> Result<Vec<sk::types::OutlierConversation>, sk::vault::StowkenError> {
    match inner {
        VaultKind::Memory(v) => v.find_outliers(k, threshold, limit).await,
        VaultKind::Filesystem(v) => v.find_outliers(k, threshold, limit).await,
        VaultKind::S3(v) => v.find_outliers(k, threshold, limit).await,
    }
}

#[cfg(feature = "semantic-search")]
fn shape_hit(py: Python<'_>, h: &sk::types::SemanticSearchHit) -> PyResult<PyObject> {
    use sk::types::MatchedVia;
    let d = PyDict::new_bound(py);
    d.set_item("conversation_id", &h.conversation_id)?;
    d.set_item("score", h.score)?;
    d.set_item("application", &h.application)?;
    d.set_item("model", &h.model)?;
    d.set_item("created_at", h.created_at.to_rfc3339())?;
    let m = PyDict::new_bound(py);
    match &h.matched_via {
        MatchedVia::Segment { hash, segment_type } => {
            m.set_item("kind", "segment")?;
            m.set_item("hash", &hash.0)?;
            m.set_item("segment_type", segment_type.to_string())?;
        }
        MatchedVia::Conversation => {
            m.set_item("kind", "conversation")?;
        }
    }
    d.set_item("matched_via", m)?;
    Ok(d.into())
}

#[cfg(feature = "semantic-search")]
fn shape_hits(py: Python<'_>, hits: &[sk::types::SemanticSearchHit]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for h in hits {
        list.append(shape_hit(py, h)?)?;
    }
    Ok(list.into())
}

#[cfg(feature = "semantic-search")]
fn shape_embed_stats(py: Python<'_>, s: &sk::types::EmbedStats) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("segments_embedded", s.segments_embedded)?;
    d.set_item("segments_skipped", s.segments_skipped)?;
    d.set_item("segments_already_done", s.segments_already_done)?;
    d.set_item("conversations_embedded", s.conversations_embedded)?;
    d.set_item("conversations_already_done", s.conversations_already_done)?;
    d.set_item("embedding_model", &s.embedding_model)?;
    d.set_item("summary_strategy", &s.summary_strategy)?;
    Ok(d.into())
}

#[cfg(feature = "semantic-search")]
fn shape_clusters(
    py: Python<'_>,
    cs: &[sk::types::ConversationCluster],
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for c in cs {
        let d = PyDict::new_bound(py);
        d.set_item("cluster_id", c.cluster_id)?;
        d.set_item("size", c.size)?;
        d.set_item("representative_ids", c.representative_ids.clone())?;
        d.set_item("members", c.members.clone())?;
        list.append(d)?;
    }
    Ok(list.into())
}

#[cfg(feature = "semantic-search")]
fn shape_outliers(
    py: Python<'_>,
    os: &[sk::types::OutlierConversation],
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for o in os {
        let d = PyDict::new_bound(py);
        d.set_item("conversation_id", &o.conversation_id)?;
        d.set_item("isolation_score", o.isolation_score)?;
        d.set_item("application", &o.application)?;
        d.set_item("model", &o.model)?;
        d.set_item("created_at", o.created_at.to_rfc3339())?;
        list.append(d)?;
    }
    Ok(list.into())
}

#[cfg(feature = "semantic-search")]
fn parse_semantic_query(
    text: String,
    granularity: String,
    limit: usize,
    min_score: f32,
    model: Option<String>,
    application: Option<String>,
    segment_type: Option<String>,
) -> PyResult<sk::types::SemanticSearchQuery> {
    use sk::types::{SearchGranularity, SegmentType};
    let granularity: SearchGranularity = granularity
        .parse()
        .map_err(|e: String| PyRuntimeError::new_err(e))?;
    let segment_type = match segment_type {
        Some(s) => Some(s.parse::<SegmentType>().map_err(PyRuntimeError::new_err)?),
        None => None,
    };
    Ok(sk::types::SemanticSearchQuery {
        text,
        granularity,
        model,
        application,
        segment_type,
        date_from: None,
        date_to: None,
        limit,
        min_score,
    })
}

// ── Main Python class ─────────────────────────────────────────────────────────

/// Compressed, deduplicated storage for LLM token sequences.
///
/// Each I/O method is exposed in two flavours:
///
///   ``vault.foo(...)``        sync; safe outside any event loop.
///   ``vault.foo_async(...)``  returns an awaitable; safe inside asyncio
///                             event loops (FastAPI, async workers).
///
/// The two share a single Tokio runtime under the hood.
///
/// Example::
///
///     # Sync
///     vault = stowken.Stowken.open("/var/lib/stowken")
///     vault.store({...})
///
///     # Async
///     async def ingest():
///         vault = stowken.Stowken.open("/var/lib/stowken")
///         await vault.store_async({...})
#[pyclass(name = "Stowken")]
pub struct PyStowken {
    /// Arc so that async futures can hold a reference to the vault across
    /// await points without lifetime gymnastics.
    inner: Arc<VaultKind>,
}

#[pymethods]
impl PyStowken {
    /// Create a new in-memory vault.
    ///
    /// Args:
    ///     enable_compression: zstd on/off. Default ``True``.
    ///     near_dedup: Jaccard threshold for storing variants as deltas
    ///         against near-duplicates. ``None`` (default) disables; a
    ///         typical conservative value is ``0.85``.
    #[new]
    #[pyo3(signature = (enable_compression=true, *, near_dedup=None))]
    fn new(_py: Python<'_>, enable_compression: bool, near_dedup: Option<f64>) -> PyResult<Self> {
        let config = StowkenConfig {
            enable_compression,
            near_dedup_threshold: near_dedup,
        };
        let backend = MemoryBackend::new();
        let inner = runtime()
            .block_on(Stowken::new(backend, config))
            .map_err(py_err)?;
        Ok(Self { inner: Arc::new(VaultKind::Memory(inner)) })
    }

    /// Open an S3-backed vault.
    ///
    /// Multi-writer safe via ETag-based optimistic concurrency on
    /// ref-count writes (see :mod:`stowken.storage.s3`). Two processes
    /// pointing at the same `(bucket, prefix)` can ingest concurrently
    /// without losing increments.
    ///
    /// The metadata index (SQLite) stays *local* at `local_db_path` —
    /// it's a derived cache that any vault can rebuild from the on-S3
    /// manifests via :meth:`reindex`. Multiple readers can each have
    /// their own local index; they'll converge on the same answer.
    ///
    /// Args:
    ///     bucket: S3 bucket name.
    ///     local_db_path: Path to a local SQLite file for the metadata
    ///         index. Created if absent. Per-process; do NOT share between
    ///         concurrent writers.
    ///     prefix: Key prefix inside the bucket. Default empty (root).
    ///     endpoint: Custom S3 endpoint URL (for MinIO, localstack, R2).
    ///     region: AWS region. Falls back to AWS_REGION env var if unset.
    ///     enable_compression: zstd on/off. Default ``True``.
    ///     near_dedup: Jaccard threshold for near-dedup. Default off.
    #[classmethod]
    #[pyo3(signature = (
        bucket,
        local_db_path,
        *,
        prefix = String::new(),
        endpoint = None,
        region = None,
        enable_compression = true,
        near_dedup = None,
    ))]
    #[allow(clippy::too_many_arguments)] // each arg is a Python kwarg by design
    fn open_s3(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        _py: Python<'_>,
        bucket: String,
        local_db_path: String,
        prefix: String,
        endpoint: Option<String>,
        region: Option<String>,
        enable_compression: bool,
        near_dedup: Option<f64>,
    ) -> PyResult<Self> {
        let config = StowkenConfig {
            enable_compression,
            near_dedup_threshold: near_dedup,
        };
        let inner = runtime()
            .block_on(async move {
                let backend = S3Backend::with_options(bucket, prefix, endpoint, region)
                    .await
                    .map_err(|e| e.to_string())?;
                Stowken::open(backend, config, &local_db_path)
                    .await
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner: Arc::new(VaultKind::S3(inner)) })
    }

    /// Open a filesystem-backed vault rooted at `path`.
    #[classmethod]
    #[pyo3(signature = (path, enable_compression=true, *, near_dedup=None))]
    fn open(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        _py: Python<'_>,
        path: String,
        enable_compression: bool,
        near_dedup: Option<f64>,
    ) -> PyResult<Self> {
        let config = StowkenConfig {
            enable_compression,
            near_dedup_threshold: near_dedup,
        };
        let db_path = format!("{}/metadata.db", path.trim_end_matches('/'));
        let inner = runtime()
            .block_on(async move {
                let backend = FilesystemBackend::new(&path).await.map_err(|e| e.to_string())?;
                Stowken::open(backend, config, &db_path)
                    .await
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner: Arc::new(VaultKind::Filesystem(inner)) })
    }

    // ── store ────────────────────────────────────────────────────────────────

    fn store(
        &self,
        _py: Python<'_>,
        conversation: &pyo3::Bound<'_, PyDict>,
    ) -> PyResult<StoreResult> {
        let conv = parse_conversation(conversation)?;
        let result = runtime()
            .block_on(vault_store(&self.inner, conv))
            .map_err(py_err)?;
        Ok(shape_store_result(result))
    }

    fn store_async<'py>(
        &self,
        py: Python<'py>,
        conversation: &pyo3::Bound<'_, PyDict>,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        // Eagerly convert the PyDict into a Conversation while we still hold
        // the GIL — the dict reference can't cross await points.
        let conv = parse_conversation(conversation)?;
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = vault_store(&inner, conv).await.map_err(py_err)?;
            Ok(shape_store_result(result))
        })
    }

    // ── retrieve ─────────────────────────────────────────────────────────────

    fn retrieve(&self, py: Python<'_>, id: &str) -> PyResult<PyObject> {
        let retrieved = runtime()
            .block_on(vault_retrieve(&self.inner, id))
            .map_err(py_err)?;
        build_retrieve_dict(py, &retrieved)
    }

    fn retrieve_async<'py>(
        &self,
        py: Python<'py>,
        id: String,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let retrieved = vault_retrieve(&inner, &id).await.map_err(py_err)?;
            Python::with_gil(|py| build_retrieve_dict(py, &retrieved))
        })
    }

    /// Retrieve multiple conversations by ID and return them as flat dicts with
    /// `conversation_id`, `model`, `application`, `text`, `turns`, `created_at`.
    /// `turns` is a list of `{"role": str, "text": str}` dicts in order.
    /// Missing IDs are silently skipped.
    fn retrieve_batch(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<PyObject> {
        let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let results = runtime()
            .block_on(vault_retrieve_batch(&self.inner, &id_refs))
            .map_err(py_err)?;
        let list = PyList::empty_bound(py);
        for ct in &results {
            list.append(build_conversation_text(py, ct)?)?;
        }
        Ok(list.into())
    }

    fn retrieve_batch_async<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            let results = vault_retrieve_batch(&inner, &id_refs).await.map_err(py_err)?;
            Python::with_gil(|py| {
                let list = PyList::empty_bound(py);
                for ct in &results {
                    list.append(build_conversation_text(py, ct)?)?;
                }
                Ok::<PyObject, PyErr>(list.into())
            })
        })
    }

    // ── delete ───────────────────────────────────────────────────────────────

    fn delete(&self, _py: Python<'_>, id: &str) -> PyResult<()> {
        runtime().block_on(vault_delete(&self.inner, id)).map_err(py_err)
    }

    fn delete_async<'py>(
        &self,
        py: Python<'py>,
        id: String,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            vault_delete(&inner, &id).await.map_err(py_err)?;
            Ok(())
        })
    }

    // ── stats ────────────────────────────────────────────────────────────────

    fn stats(&self, _py: Python<'_>) -> PyResult<TokenUsageStats> {
        let s = runtime().block_on(vault_stats(&self.inner)).map_err(py_err)?;
        Ok(shape_token_stats(s))
    }

    fn stats_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let s = vault_stats(&inner).await.map_err(py_err)?;
            Ok(shape_token_stats(s))
        })
    }

    // ── gc / gc_candidates ───────────────────────────────────────────────────

    fn gc(&self, _py: Python<'_>) -> PyResult<u64> {
        runtime().block_on(vault_gc(&self.inner)).map_err(py_err)
    }

    fn gc_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            vault_gc(&inner).await.map_err(py_err)
        })
    }

    fn gc_candidates(&self, py: Python<'_>) -> PyResult<PyObject> {
        let hashes = runtime()
            .block_on(vault_gc_candidates(&self.inner))
            .map_err(py_err)?;
        build_hash_list(py, &hashes)
    }

    fn gc_candidates_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let hashes = vault_gc_candidates(&inner).await.map_err(py_err)?;
            Python::with_gil(|py| build_hash_list(py, &hashes))
        })
    }

    // ── segment_stats ────────────────────────────────────────────────────────

    fn segment_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = runtime()
            .block_on(vault_segment_stats(&self.inner))
            .map_err(py_err)?;
        build_segment_stats_list(py, &stats)
    }

    fn segment_stats_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stats = vault_segment_stats(&inner).await.map_err(py_err)?;
            Python::with_gil(|py| build_segment_stats_list(py, &stats))
        })
    }

    // ── reindex ──────────────────────────────────────────────────────────────

    fn reindex(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = runtime().block_on(vault_reindex(&self.inner)).map_err(py_err)?;
        build_reindex_dict(py, &report)
    }

    fn reindex_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let report = vault_reindex(&inner).await.map_err(py_err)?;
            Python::with_gil(|py| build_reindex_dict(py, &report))
        })
    }

    // ── substrings (v0.5) ────────────────────────────────────────────────────

    /// Discover and register high-frequency token-level substrings from
    /// existing segments. Returns a list of dicts describing each
    /// promoted substring (`{id, length, created_at, source_occurrences}`).
    /// Future ingests use frame format `0x05` (substring references)
    /// when it beats the baseline; old segments are unaffected.
    #[pyo3(signature = (samples=500, min_occurrences=5))]
    fn train_substrings(
        &self,
        py: Python<'_>,
        samples: usize,
        min_occurrences: u32,
    ) -> PyResult<PyObject> {
        let promoted = runtime()
            .block_on(vault_train_substrings(&self.inner, samples, min_occurrences))
            .map_err(py_err)?;
        build_substring_info_list(py, &promoted)
    }

    #[pyo3(signature = (samples=500, min_occurrences=5))]
    fn train_substrings_async<'py>(
        &self,
        py: Python<'py>,
        samples: usize,
        min_occurrences: u32,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let promoted =
                vault_train_substrings(&inner, samples, min_occurrences)
                    .await
                    .map_err(py_err)?;
            Python::with_gil(|py| build_substring_info_list(py, &promoted))
        })
    }

    /// List all registered substrings.
    fn list_substrings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let infos = vault_list_substrings(&self.inner);
        build_substring_info_list(py, &infos)
    }

    fn list_substrings_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let infos = vault_list_substrings(&inner);
            Python::with_gil(|py| build_substring_info_list(py, &infos))
        })
    }

    /// Retroactively rewrite pre-train segments to use the substring
    /// registry. Returns a dict with ``segments_examined``,
    /// ``segments_rewritten``, ``bytes_saved``, and ``segments_skipped``.
    fn compact_substrings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = runtime()
            .block_on(vault_compact_substrings(&self.inner))
            .map_err(py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("segments_examined", report.segments_examined)?;
        dict.set_item("segments_rewritten", report.segments_rewritten)?;
        dict.set_item("bytes_saved", report.bytes_saved)?;
        dict.set_item("segments_skipped", report.segments_skipped)?;
        Ok(dict.into())
    }

    fn compact_substrings_async<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let report = vault_compact_substrings(&inner).await.map_err(py_err)?;
            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                dict.set_item("segments_examined", report.segments_examined)?;
                dict.set_item("segments_rewritten", report.segments_rewritten)?;
                dict.set_item("bytes_saved", report.bytes_saved)?;
                dict.set_item("segments_skipped", report.segments_skipped)?;
                Ok::<PyObject, PyErr>(dict.into())
            })
        })
    }

    /// Drop substrings from the registry that no `0x05` frame
    /// references anymore. Returns a dict with ``registry_size_before``,
    /// ``registry_size_after``, ``substrings_dropped``.
    fn gc_substrings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = runtime()
            .block_on(vault_gc_substrings(&self.inner))
            .map_err(py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("registry_size_before", report.registry_size_before)?;
        dict.set_item("registry_size_after", report.registry_size_after)?;
        dict.set_item("substrings_dropped", report.substrings_dropped)?;
        Ok(dict.into())
    }

    fn gc_substrings_async<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let report = vault_gc_substrings(&inner).await.map_err(py_err)?;
            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                dict.set_item("registry_size_before", report.registry_size_before)?;
                dict.set_item("registry_size_after", report.registry_size_after)?;
                dict.set_item("substrings_dropped", report.substrings_dropped)?;
                Ok::<PyObject, PyErr>(dict.into())
            })
        })
    }

    // ── near-duplicates ──────────────────────────────────────────────────────

    /// Find clusters of segments that are *almost* identical. Returns a
    /// list of dicts; each cluster has ``canonical`` (hash),
    /// ``canonical_token_count``, ``canonical_ref_count``, and a
    /// ``variants`` list of ``{hash, similarity, token_count, ref_count}``.
    #[pyo3(signature = (threshold=0.85, min_refs=5, limit=25))]
    fn find_near_duplicates(
        &self,
        py: Python<'_>,
        threshold: f64,
        min_refs: u64,
        limit: u64,
    ) -> PyResult<PyObject> {
        let clusters = runtime()
            .block_on(vault_find_near_duplicates(&self.inner, threshold, min_refs, limit))
            .map_err(py_err)?;
        build_near_dup_clusters(py, &clusters)
    }

    #[pyo3(signature = (threshold=0.85, min_refs=5, limit=25))]
    fn find_near_duplicates_async<'py>(
        &self,
        py: Python<'py>,
        threshold: f64,
        min_refs: u64,
        limit: u64,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let clusters =
                vault_find_near_duplicates(&inner, threshold, min_refs, limit)
                    .await
                    .map_err(py_err)?;
            Python::with_gil(|py| build_near_dup_clusters(py, &clusters))
        })
    }

    /// Number of segments tracked in the LSH (near-dedup) index. Useful
    /// for sanity-checking after `reindex` or before `find_near_duplicates`.
    fn near_dedup_index_size(&self, _py: Python<'_>) -> PyResult<u64> {
        runtime()
            .block_on(vault_near_dedup_index_size(&self.inner))
            .map_err(py_err)
    }

    fn near_dedup_index_size_async<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            vault_near_dedup_index_size(&inner).await.map_err(py_err)
        })
    }

    // ── dictionaries ─────────────────────────────────────────────────────────

    /// Train a new zstd dictionary from `samples` segments. Returns a dict
    /// describing the new dictionary. Does NOT activate — pass
    /// ``activate=True`` or call ``activate_dictionary`` separately.
    #[pyo3(signature = (samples=200, *, activate=false))]
    fn train_dictionary(
        &self,
        py: Python<'_>,
        samples: usize,
        activate: bool,
    ) -> PyResult<PyObject> {
        let info = if activate {
            runtime()
                .block_on(vault_train_and_activate_dict(&self.inner, samples))
                .map_err(py_err)?
        } else {
            runtime()
                .block_on(vault_train_dict(&self.inner, samples))
                .map_err(py_err)?
        };
        build_dict_info_dict(py, &info)
    }

    #[pyo3(signature = (samples=200, *, activate=false))]
    fn train_dictionary_async<'py>(
        &self,
        py: Python<'py>,
        samples: usize,
        activate: bool,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let info = if activate {
                vault_train_and_activate_dict(&inner, samples).await.map_err(py_err)?
            } else {
                vault_train_dict(&inner, samples).await.map_err(py_err)?
            };
            Python::with_gil(|py| build_dict_info_dict(py, &info))
        })
    }

    /// Mark `dict_id` as the dictionary new compressions will use. Old
    /// frames keep their original dict — rotation is non-destructive.
    fn activate_dictionary(&self, _py: Python<'_>, dict_id: u32) -> PyResult<()> {
        runtime()
            .block_on(vault_activate_dict(&self.inner, dict_id))
            .map_err(py_err)
    }

    fn activate_dictionary_async<'py>(
        &self,
        py: Python<'py>,
        dict_id: u32,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            vault_activate_dict(&inner, dict_id).await.map_err(py_err)?;
            Ok(())
        })
    }

    /// List all known dictionaries, sorted by creation time. Each entry is
    /// a dict with ``id``, ``created_at``, ``sample_count``, ``size_bytes``,
    /// and ``is_active``.
    fn list_dictionaries(&self, py: Python<'_>) -> PyResult<PyObject> {
        let infos = vault_list_dicts(&self.inner);
        build_dict_info_list(py, &infos)
    }

    fn list_dictionaries_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let infos = vault_list_dicts(&inner);
            Python::with_gil(|py| build_dict_info_list(py, &infos))
        })
    }

    // ── Semantic search (feature: semantic-search) ──────────────────────────

    /// Configure the embedding adapter.
    ///
    /// Args:
    ///     embed_fn: ``Callable[[list[str]], list[list[float]]]``. Each
    ///         returned vector must have length ``dimension``.
    ///     dimension: Embedding vector size.
    ///     model_name: Identifier persisted alongside each embedding.
    ///         Changing this means previously embedded segments won't match.
    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (embed_fn, *, dimension, model_name))]
    fn set_embedding_adapter(
        &self,
        _py: Python<'_>,
        embed_fn: PyObject,
        dimension: usize,
        model_name: String,
    ) -> PyResult<()> {
        let adapter = semantic_bridge::PyEmbeddingAdapter {
            callback: embed_fn,
            dim: dimension,
            model: model_name,
        };
        vault_set_embedding_adapter(&self.inner, adapter);
        Ok(())
    }

    /// Configure an LLM-based summarizer for conversation embeddings.
    ///
    /// Args:
    ///     summarize_fn: ``Callable[[str], str]``.
    ///     model_name: Persisted as part of the summary strategy ID.
    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (summarize_fn, *, model_name))]
    fn set_summarizer(
        &self,
        _py: Python<'_>,
        summarize_fn: PyObject,
        model_name: String,
    ) -> PyResult<()> {
        let adapter = semantic_bridge::PySummarizerAdapter {
            callback: summarize_fn,
            model: model_name,
        };
        vault_set_summarizer(&self.inner, adapter);
        Ok(())
    }

    /// Use the cheap "concat-and-truncate" summary strategy. Default after
    /// `set_embedding_adapter`. ``max_chars`` defaults to 24_000 (~6K tokens).
    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (max_chars=24_000))]
    fn use_concat_summary(&self, _py: Python<'_>, max_chars: usize) {
        vault_use_concat_summary(&self.inner, max_chars);
    }

    /// Toggle whether ``store()`` automatically embeds new segments and a
    /// conversation summary. Default ``True``. Disable for bulk loads, then
    /// run ``embed_all`` once afterwards.
    #[cfg(feature = "semantic-search")]
    fn set_embed_on_store(&self, _py: Python<'_>, enabled: bool) {
        vault_set_embed_on_store(&self.inner, enabled);
    }

    /// Embed every stored segment + conversation summary that lacks an
    /// embedding for the configured model.
    #[cfg(feature = "semantic-search")]
    fn embed_all(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = runtime()
            .block_on(vault_embed_all(&self.inner))
            .map_err(py_err)?;
        shape_embed_stats(py, &stats)
    }

    #[cfg(feature = "semantic-search")]
    fn embed_all_async<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stats = vault_embed_all(&inner).await.map_err(py_err)?;
            Python::with_gil(|py| shape_embed_stats(py, &stats))
        })
    }

    /// Search conversations by semantic meaning.
    ///
    /// Returns a list of dicts: ``{conversation_id, score, matched_via,
    /// application, model, created_at}``, sorted by score descending.
    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (
        text,
        *,
        granularity = "both".to_owned(),
        limit = 10,
        min_score = 0.0,
        model = None,
        application = None,
        segment_type = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn semantic_search(
        &self,
        py: Python<'_>,
        text: String,
        granularity: String,
        limit: usize,
        min_score: f32,
        model: Option<String>,
        application: Option<String>,
        segment_type: Option<String>,
    ) -> PyResult<PyObject> {
        let q = parse_semantic_query(
            text,
            granularity,
            limit,
            min_score,
            model,
            application,
            segment_type,
        )?;
        let hits = runtime()
            .block_on(vault_semantic_search(&self.inner, q))
            .map_err(py_err)?;
        shape_hits(py, &hits)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (
        text,
        *,
        granularity = "both".to_owned(),
        limit = 10,
        min_score = 0.0,
        model = None,
        application = None,
        segment_type = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn semantic_search_async<'py>(
        &self,
        py: Python<'py>,
        text: String,
        granularity: String,
        limit: usize,
        min_score: f32,
        model: Option<String>,
        application: Option<String>,
        segment_type: Option<String>,
    ) -> PyResult<pyo3::Bound<'py, PyAny>> {
        let q = parse_semantic_query(
            text,
            granularity,
            limit,
            min_score,
            model,
            application,
            segment_type,
        )?;
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let hits = vault_semantic_search(&inner, q).await.map_err(py_err)?;
            Python::with_gil(|py| shape_hits(py, &hits))
        })
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (text, *, limit = 10))]
    fn search_prompts(&self, py: Python<'_>, text: String, limit: usize) -> PyResult<PyObject> {
        let hits = runtime()
            .block_on(vault_typed_search(&self.inner, "prompts", &text, limit))
            .map_err(py_err)?;
        shape_hits(py, &hits)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (text, *, limit = 10))]
    fn search_responses(&self, py: Python<'_>, text: String, limit: usize) -> PyResult<PyObject> {
        let hits = runtime()
            .block_on(vault_typed_search(&self.inner, "responses", &text, limit))
            .map_err(py_err)?;
        shape_hits(py, &hits)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (text, *, limit = 10))]
    fn search_user_questions(
        &self,
        py: Python<'_>,
        text: String,
        limit: usize,
    ) -> PyResult<PyObject> {
        let hits = runtime()
            .block_on(vault_typed_search(&self.inner, "user_questions", &text, limit))
            .map_err(py_err)?;
        shape_hits(py, &hits)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (text, *, limit = 10))]
    fn search_tool_uses(&self, py: Python<'_>, text: String, limit: usize) -> PyResult<PyObject> {
        let hits = runtime()
            .block_on(vault_typed_search(&self.inner, "tool_uses", &text, limit))
            .map_err(py_err)?;
        shape_hits(py, &hits)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (text, *, limit = 10))]
    fn search_conversations(
        &self,
        py: Python<'_>,
        text: String,
        limit: usize,
    ) -> PyResult<PyObject> {
        let hits = runtime()
            .block_on(vault_typed_search(&self.inner, "conversations", &text, limit))
            .map_err(py_err)?;
        shape_hits(py, &hits)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (k = 10, *, representative_count = 3))]
    fn cluster_conversations(
        &self,
        py: Python<'_>,
        k: usize,
        representative_count: usize,
    ) -> PyResult<PyObject> {
        let clusters = runtime()
            .block_on(vault_cluster(&self.inner, k, representative_count))
            .map_err(py_err)?;
        shape_clusters(py, &clusters)
    }

    #[cfg(feature = "semantic-search")]
    #[pyo3(signature = (k = 10, *, threshold = 0.5, limit = 20))]
    fn find_outliers(
        &self,
        py: Python<'_>,
        k: usize,
        threshold: f32,
        limit: usize,
    ) -> PyResult<PyObject> {
        let outliers = runtime()
            .block_on(vault_outliers(&self.inner, k, threshold, limit))
            .map_err(py_err)?;
        shape_outliers(py, &outliers)
    }

    // ── repr ─────────────────────────────────────────────────────────────────

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
