//! Node.js bindings for Stowken (napi-rs).
//!
//! Build:  npm run build   (release)
//!         npm run build -- --release=false   (debug)
//!
//! Usage:
//!   const { Stowken } = require('stowken');
//!   const vault = new Stowken();
//!   const result = await vault.store({ model: 'gpt-4', messages: [...] });
//!   const stats  = await vault.stats();

#![allow(clippy::new_without_default)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

// ── Result types ──────────────────────────────────────────────────────────────

/// Result of a `store()` call.
#[napi(object)]
pub struct StoreResult {
    /// The assigned conversation ID.
    pub id: String,
    /// Total segment count (new + deduped).
    pub total_segments: i64,
    /// Segments written for the first time.
    pub new_segments: i64,
    /// Segments that already existed and were reused.
    pub deduped_segments: i64,
    /// Bytes saved by deduplication.
    pub bytes_saved: i64,
    /// Compression ratio (raw / compressed).
    pub compression_ratio: f64,
}

/// Aggregate storage statistics from `stats()`.
#[napi(object)]
pub struct TokenUsageStats {
    pub total_tokens: i64,
    pub total_conversations: i64,
    pub unique_segments: i64,
    pub total_segments: i64,
    pub dedup_ratio: f64,
    pub compression_ratio: f64,
    pub storage_bytes: i64,
    pub savings_percentage: f64,
}

/// Per-segment-type statistics from `segmentStats()`.
#[napi(object)]
pub struct SegmentTypeStats {
    pub segment_type: String,
    pub unique_count: i64,
    pub total_references: i64,
    pub dedup_ratio: f64,
    pub total_tokens: i64,
    pub compressed_bytes: i64,
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

fn parse_message(obj: &serde_json::Value) -> napi::Result<Message> {
    let role = obj["role"]
        .as_str()
        .ok_or_else(|| Error::from_reason("message missing 'role'"))?
        .to_owned();

    // Prefer explicit token arrays (pre-tokenized); otherwise pass the text
    // through as-is and let the vault's built-in tiktoken tokenizer handle it.
    let content = if let Some(tokens_arr) = obj.get("tokens").and_then(|v| v.as_array()) {
        let tokens: Vec<u32> = tokens_arr
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();
        MessageContent::Tokens(tokens)
    } else {
        let text = match obj.get("content") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        MessageContent::Text(text)
    };

    let name = obj.get("name").and_then(|v| v.as_str()).map(str::to_owned);
    let tool_call_id = obj.get("tool_call_id").and_then(|v| v.as_str()).map(str::to_owned);

    Ok(Message { role, content, name, tool_call_id })
}

fn parse_conversation(val: serde_json::Value) -> napi::Result<Conversation> {
    let model = val["model"]
        .as_str()
        .ok_or_else(|| Error::from_reason("conversation missing 'model'"))?
        .to_owned();

    let tokenizer = val["tokenizer"]
        .as_str()
        .unwrap_or("cl100k_base")
        .to_owned();

    let application = val["application"].as_str().map(str::to_owned);
    let id = val["id"].as_str().map(str::to_owned);

    let messages_raw = val["messages"]
        .as_array()
        .ok_or_else(|| Error::from_reason("conversation missing 'messages'"))?;

    let mut messages = Vec::with_capacity(messages_raw.len());
    for msg in messages_raw {
        messages.push(parse_message(msg)?);
    }

    Ok(Conversation { id, model, tokenizer, application, metadata: None, messages })
}

fn vault_err(e: impl std::fmt::Display) -> napi::Error {
    Error::from_reason(e.to_string())
}

// ── Main binding class ────────────────────────────────────────────────────────

/// In-memory Stowken instance.
///
/// @example
/// ```js
/// const { Stowken } = require('stowken');
///
/// const vault = new Stowken();
///
/// const result = await vault.store({
///   model: 'gpt-4',
///   messages: [
///     { role: 'system',    content: 'You are helpful.' },
///     { role: 'user',      content: 'Hello!' },
///     { role: 'assistant', content: 'Hi there!' },
///   ]
/// });
/// console.log(result.id, result.dedupedSegments);
///
/// const stats = await vault.stats();
/// console.log(`Saved ${stats.savingsPercentage.toFixed(1)}%`);
/// ```
#[napi(js_name = "Stowken")]
pub struct JsStowken {
    inner: Stowken<MemoryBackend>,
}

#[napi]
impl JsStowken {
    /// Create a new in-memory vault.
    ///
    /// @param enableCompression - Enable zstd compression. Default `true`.
    #[napi(constructor)]
    pub fn new(enable_compression: Option<bool>) -> napi::Result<Self> {
        let config = StowkenConfig {
            enable_compression: enable_compression.unwrap_or(true),
            dictionary_path: None,
            tenant_id: None,
            ..StowkenConfig::default()
        };
        let backend = MemoryBackend::new();
        let rt = tokio::runtime::Runtime::new().map_err(vault_err)?;
        let inner = rt.block_on(Stowken::new(backend, config)).map_err(vault_err)?;
        Ok(Self { inner })
    }

    /// Store a conversation object (OpenAI-compatible format).
    ///
    /// Accepts `{ model, messages, tokenizer?, application?, id? }`.
    /// Each message may include a `tokens: number[]` field with pre-tokenized
    /// content; if omitted, the `content` string is tokenized automatically
    /// using the built-in tiktoken encoder matching the conversation's
    /// `tokenizer` field (default: `cl100k_base`).
    ///
    /// @returns StoreResult with the assigned ID and deduplication metrics.
    #[napi]
    pub async fn store(&self, conversation: serde_json::Value) -> napi::Result<StoreResult> {
        let conv = parse_conversation(conversation)?;
        let result = self.inner.store(conv).await.map_err(vault_err)?;
        Ok(StoreResult {
            id: result.id,
            total_segments: result.total_segments as i64,
            new_segments: result.new_segments as i64,
            deduped_segments: result.deduped_segments as i64,
            bytes_saved: result.bytes_saved as i64,
            compression_ratio: result.compression_ratio,
        })
    }

    /// Retrieve a conversation by ID.
    ///
    /// @returns Object with `id`, `model`, `tokenizer`, `totalTokens`, and `segments`.
    #[napi]
    pub async fn retrieve(&self, id: String) -> napi::Result<serde_json::Value> {
        let retrieved = self.inner.retrieve(&id).await.map_err(vault_err)?;

        let segments: Vec<serde_json::Value> = retrieved
            .segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "segmentType": s.segment_type.to_string(),
                    "tokenCount":  s.token_count,
                    "hash":        s.hash.0,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "id":          retrieved.manifest.id,
            "model":       retrieved.manifest.model,
            "tokenizer":   retrieved.manifest.tokenizer,
            "application": retrieved.manifest.application,
            "totalTokens": retrieved.manifest.total_tokens,
            "segments":    segments,
        }))
    }

    /// Delete a conversation by ID.
    #[napi]
    pub async fn delete(&self, id: String) -> napi::Result<()> {
        self.inner.delete(&id).await.map_err(vault_err)
    }

    /// Return aggregate storage statistics.
    #[napi]
    pub async fn stats(&self) -> napi::Result<TokenUsageStats> {
        let s = self.inner.stats().await.map_err(vault_err)?;
        Ok(TokenUsageStats {
            total_tokens: s.total_tokens as i64,
            total_conversations: s.total_conversations as i64,
            unique_segments: s.unique_segments as i64,
            total_segments: s.total_segments as i64,
            dedup_ratio: s.dedup_ratio,
            compression_ratio: s.compression_ratio,
            storage_bytes: s.storage_bytes as i64,
            savings_percentage: s.savings_percentage,
        })
    }

    /// Run garbage collection.
    ///
    /// @returns Number of zero-reference segments deleted.
    #[napi]
    pub async fn gc(&self) -> napi::Result<i64> {
        self.inner.gc().await.map(|n| n as i64).map_err(vault_err)
    }

    /// Return per-segment-type statistics.
    #[napi]
    pub async fn segment_stats(&self) -> napi::Result<Vec<SegmentTypeStats>> {
        let stats = self.inner.segment_stats().await.map_err(vault_err)?;
        Ok(stats
            .into_iter()
            .map(|s| SegmentTypeStats {
                segment_type: s.segment_type.to_string(),
                unique_count: s.unique_count as i64,
                total_references: s.total_references as i64,
                dedup_ratio: s.dedup_ratio,
                total_tokens: s.total_tokens as i64,
                compressed_bytes: s.compressed_bytes as i64,
            })
            .collect())
    }
}
