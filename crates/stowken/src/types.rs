//! Core data types for Stowken.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A token is an integer ID from a tokenizer vocabulary.
pub type Token = u32;

/// Segment types matching LLM conversation structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentType {
    SystemPrompt,
    Context,
    UserTurn,
    AssistantTurn,
    ToolCall,
    ToolResult,
}

impl std::fmt::Display for SegmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::SystemPrompt => "system_prompt",
            Self::Context => "context",
            Self::UserTurn => "user_turn",
            Self::AssistantTurn => "assistant_turn",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for SegmentType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "system_prompt" => Ok(Self::SystemPrompt),
            "context" => Ok(Self::Context),
            "user_turn" => Ok(Self::UserTurn),
            "assistant_turn" => Ok(Self::AssistantTurn),
            "tool_call" => Ok(Self::ToolCall),
            "tool_result" => Ok(Self::ToolResult),
            other => Err(format!("unknown segment type: {other}")),
        }
    }
}

/// A single segment of a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub segment_type: SegmentType,
    pub tokens: Vec<Token>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// SHA-256 hex-encoded hash of a segment's raw token content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegmentHash(pub String);

impl std::fmt::Display for SegmentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A stored segment reference within a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    pub segment_type: SegmentType,
    pub hash: SegmentHash,
    pub token_count: u32,
    pub position: u32,
}

/// Current manifest schema version. Bump when fields are added or semantics change.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Conversation manifest — the lightweight pointer structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationManifest {
    /// Schema version for forward compatibility. Defaults to the current
    /// constant when missing on read (older manifests predate this field).
    #[serde(default = "default_manifest_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub application: Option<String>,
    pub model: String,
    pub tokenizer: String,
    pub total_tokens: u64,
    pub segments: Vec<SegmentRef>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

fn default_manifest_schema_version() -> u32 {
    MANIFEST_SCHEMA_VERSION
}

/// Input format: a conversation as the user provides it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Auto-generated if `None`.
    pub id: Option<String>,
    pub application: Option<String>,
    pub model: String,
    pub tokenizer: String,
    pub messages: Vec<Message>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// A single message in a conversation (OpenAI/Anthropic compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// "system", "user", "assistant", or "tool"
    pub role: String,
    pub content: MessageContent,
    /// For tool calls/results.
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

/// Message content: either raw text (requires a tokenizer adapter) or pre-tokenized.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Tokens(Vec<Token>),
}

impl MessageContent {
    /// Returns the token slice if content is pre-tokenized.
    pub fn as_tokens(&self) -> Option<&[Token]> {
        match self {
            Self::Tokens(t) => Some(t),
            Self::Text(_) => None,
        }
    }
}

/// A segment stored in the backend.
#[derive(Debug, Clone)]
pub struct StoredSegment {
    pub hash: SegmentHash,
    pub segment_type: SegmentType,
    pub tokenizer: String,
    pub token_count: u32,
    /// varint-encoded + zstd-compressed token data.
    pub compressed_data: Vec<u8>,
    /// Uncompressed size in bytes (token_count * 4).
    pub raw_size: u32,
    pub compressed_size: u32,
    pub ref_count: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Configuration for a `Stowken` instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StowkenConfig {
    /// Enable varint + zstd compression. Disable for benchmarking or backends
    /// that apply their own compression.
    pub enable_compression: bool,
}

impl Default for StowkenConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
        }
    }
}

/// Results from a successful `store` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreResult {
    pub id: String,
    pub total_segments: u64,
    pub new_segments: u64,
    pub deduped_segments: u64,
    pub bytes_saved: u64,
    pub compression_ratio: f64,
}

/// A retrieved conversation with all segments reassembled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedConversation {
    pub manifest: ConversationManifest,
    pub segments: Vec<RetrievedSegment>,
}

/// A single retrieved and decompressed segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedSegment {
    pub segment_type: SegmentType,
    pub hash: SegmentHash,
    pub tokens: Vec<Token>,
    pub token_count: u32,
    pub position: u32,
}

/// Aggregate storage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageStats {
    pub total_tokens: u64,
    pub total_conversations: u64,
    pub unique_segments: u64,
    /// Total segment references including duplicates.
    pub total_segments: u64,
    /// `1.0 - (unique / total)`
    pub dedup_ratio: f64,
    /// `compressed_size / raw_size`
    pub compression_ratio: f64,
    /// Actual bytes stored on disk / backend.
    pub storage_bytes: u64,
    /// Bytes without Stowken (token_count * 4 per reference).
    pub naive_bytes: u64,
    pub savings_percentage: f64,
}

/// Per-segment-type statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentTypeStats {
    pub segment_type: SegmentType,
    pub unique_count: u64,
    pub total_references: u64,
    pub dedup_ratio: f64,
    pub avg_token_count: f64,
    pub total_tokens: u64,
    pub compressed_bytes: u64,
}

/// Query filters for analytics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyticsQuery {
    pub model: Option<String>,
    pub application: Option<String>,
    pub segment_type: Option<SegmentType>,
    pub date_from: Option<chrono::DateTime<chrono::Utc>>,
    pub date_to: Option<chrono::DateTime<chrono::Utc>>,
    /// Valid values: "model", "application", "segment_type", "day"
    pub group_by: Option<Vec<String>>,
}

/// Training data export configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub include_system_prompts: bool,
    pub include_context: bool,
    /// Skip duplicate user/assistant turn pairs.
    pub deduplicate_pairs: bool,
    pub tokenizer: Option<String>,
    pub model: Option<String>,
    pub application: Option<String>,
    pub max_conversations: Option<u64>,
}

/// Supported export formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Jsonl,
    HuggingFace,
    Parquet,
}

/// Export operation summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStats {
    pub total_pairs: u64,
    pub unique_pairs: u64,
    pub tokens_exported: u64,
}

/// Info about a unique system prompt segment (for audit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptInfo {
    pub hash: SegmentHash,
    pub token_count: u32,
    pub ref_count: u64,
}

/// A heavily-referenced segment surfaced by the duplicate finder.
///
/// The `wasted_bytes` field reports how many bytes the duplicates would have
/// cost if stored naively — i.e. what dedup is saving on this segment alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateSegment {
    pub hash: SegmentHash,
    pub segment_type: SegmentType,
    pub token_count: u32,
    pub ref_count: u64,
    /// `(ref_count - 1) * raw_size` — bytes saved vs. naive storage.
    pub wasted_bytes: u64,
}

/// Trait for supplying a tokenizer to Stowken.
///
/// Stowken does not ship with a tokenizer. Implement this trait
/// wrapping tiktoken-rs, HuggingFace tokenizers, etc.
pub trait TokenizerAdapter: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<Token>;
    fn detokenize(&self, tokens: &[Token]) -> String;
    fn vocab_size(&self) -> u32;
    fn name(&self) -> &str;
}
