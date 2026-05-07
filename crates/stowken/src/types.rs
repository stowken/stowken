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
    /// A sub-segment continuation of the immediately preceding segment.
    /// Produced when `max_segment_tokens` splits a long message across
    /// multiple chunks. Callers that need the original message should
    /// concatenate all contiguous `Continuation` tokens with the preceding
    /// non-`Continuation` segment.
    Continuation,
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
            Self::Continuation => "continuation",
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
            "continuation" => Ok(Self::Continuation),
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
    /// Jaccard similarity threshold for storing a segment as a delta against
    /// a near-duplicate canonical. `None` disables near-dedup entirely
    /// (default). `Some(0.85)` is a conservative starting value.
    ///
    /// Lower values catch more variants but raise false-positive risk; very
    /// low values can produce inflated delta sizes and worse storage. Stick
    /// to the 0.80–0.95 range.
    pub near_dedup_threshold: Option<f64>,
}

impl Default for StowkenConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            near_dedup_threshold: None,
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

/// A flat, human-readable view of a retrieved conversation — tokens detokenized
/// to text and joined with role markers. Returned by `retrieve_batch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationText {
    pub conversation_id: String,
    pub model: String,
    pub application: Option<String>,
    /// Full conversation text: `"[system] ...\n[user] ...\n[assistant] ..."`.
    pub text: String,
    /// Per-turn breakdown in order.
    pub turns: Vec<ConversationTurn>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// One turn within a `ConversationText`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub text: String,
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

/// Result of a reindex pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexStats {
    pub conversations_indexed: u64,
    /// Distinct segment hashes whose metadata was reindexed.
    pub unique_segments_indexed: u64,
    /// Segment references found in manifests whose backend bytes were missing.
    /// Indicates corruption — these refs are skipped, not surfaced as errors.
    pub segments_missing: u64,
}

/// Result of a substring compaction pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstringCompactStats {
    /// Distinct segments examined.
    pub segments_examined: u64,
    /// Segments rewritten to use the substring registry.
    pub segments_rewritten: u64,
    /// Total bytes saved across all rewrites (`old - new`, summed).
    pub bytes_saved: u64,
    /// Segments skipped because they're already 0x04 (delta) or 0x05
    /// (substring) frames and shouldn't be re-encoded.
    pub segments_skipped: u64,
}

/// Result of a substring GC pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstringGcStats {
    /// Substrings examined in the registry.
    pub registry_size_before: u64,
    /// Substrings still referenced after the scan.
    pub registry_size_after: u64,
    /// Substrings dropped from the registry.
    pub substrings_dropped: u64,
}

/// One canonical-and-its-near-duplicates cluster surfaced by
/// `find_near_duplicates`. The canonical has the highest reference count
/// in the cluster; variants are ordered by similarity descending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearDuplicateCluster {
    pub canonical: SegmentHash,
    pub canonical_token_count: u32,
    pub canonical_ref_count: u64,
    pub variants: Vec<NearDuplicateVariant>,
}

/// One near-duplicate of a canonical segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearDuplicateVariant {
    pub hash: SegmentHash,
    pub similarity: f64,
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

// ─────────────────────────────────────────────────────────────────────────────
// Semantic search types (feature: `semantic-search`)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "semantic-search")]
mod semantic {
    use super::*;
    use std::sync::Arc;

    /// Trait for supplying an embedding backend to Stowken.
    ///
    /// Implement this to plug in any embedding model — local (e.g. fastembed,
    /// candle) or remote (OpenAI, Cohere, Voyage). The trait is sync because
    /// callers bridge to async via `tokio::task::spawn_blocking`.
    ///
    /// Every `Vec<f32>` returned by `embed_batch` must have length == `dimension()`.
    pub trait EmbeddingAdapter: Send + Sync {
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String>;
        fn dimension(&self) -> usize;
        fn model_name(&self) -> &str;
    }

    /// Trait for supplying an LLM-based summarizer used to produce a short
    /// natural-language summary of a conversation prior to embedding.
    pub trait SummarizerAdapter: Send + Sync {
        fn summarize(&self, conversation_text: &str) -> Result<String, String>;
        fn model_name(&self) -> &str;
    }

    /// How a conversation is reduced to text before being embedded.
    #[derive(Clone)]
    pub enum SummaryStrategy {
        /// Detokenize all segments, concatenate with role markers, truncate to
        /// `max_chars`. Default: 24_000 chars (~6K tokens).
        ConcatTruncate { max_chars: usize },
        /// Use an LLM to generate a short paragraph, then embed it.
        LlmGenerated(Arc<dyn SummarizerAdapter>),
    }

    impl Default for SummaryStrategy {
        fn default() -> Self {
            Self::ConcatTruncate { max_chars: 24_000 }
        }
    }

    impl SummaryStrategy {
        /// String identifier persisted alongside each conversation embedding.
        /// Allows multiple strategies to coexist for the same `(conv, model)`.
        pub fn id(&self) -> String {
            match self {
                Self::ConcatTruncate { .. } => "concat".to_owned(),
                Self::LlmGenerated(s) => format!("llm:{}", s.model_name()),
            }
        }
    }

    impl std::fmt::Debug for SummaryStrategy {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::ConcatTruncate { max_chars } => f
                    .debug_struct("ConcatTruncate")
                    .field("max_chars", max_chars)
                    .finish(),
                Self::LlmGenerated(s) => f
                    .debug_struct("LlmGenerated")
                    .field("model", &s.model_name())
                    .finish(),
            }
        }
    }

    /// Which embeddings to scan during a semantic search.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SearchGranularity {
        /// Per-unique-segment embeddings only.
        Segment,
        /// Per-conversation summary embeddings only.
        Conversation,
        /// Both — best score per conversation across granularities.
        Both,
    }

    impl Default for SearchGranularity {
        fn default() -> Self {
            Self::Both
        }
    }

    impl std::str::FromStr for SearchGranularity {
        type Err = String;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "segment" => Ok(Self::Segment),
                "conversation" => Ok(Self::Conversation),
                "both" => Ok(Self::Both),
                other => Err(format!("unknown granularity: {other}")),
            }
        }
    }

    /// Query for `Stowken::semantic_search`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SemanticSearchQuery {
        pub text: String,
        #[serde(default)]
        pub granularity: SearchGranularity,
        pub model: Option<String>,
        pub application: Option<String>,
        pub segment_type: Option<SegmentType>,
        pub date_from: Option<chrono::DateTime<chrono::Utc>>,
        pub date_to: Option<chrono::DateTime<chrono::Utc>>,
        pub limit: usize,
        pub min_score: f32,
    }

    impl SemanticSearchQuery {
        pub fn new(text: impl Into<String>) -> Self {
            Self {
                text: text.into(),
                granularity: SearchGranularity::Both,
                model: None,
                application: None,
                segment_type: None,
                date_from: None,
                date_to: None,
                limit: 10,
                min_score: 0.0,
            }
        }
    }

    impl Serialize for SearchGranularity {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let v = match self {
                Self::Segment => "segment",
                Self::Conversation => "conversation",
                Self::Both => "both",
            };
            s.serialize_str(v)
        }
    }

    impl<'de> Deserialize<'de> for SearchGranularity {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let s = String::deserialize(d)?;
            s.parse().map_err(serde::de::Error::custom)
        }
    }

    /// One result row from a semantic search.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SemanticSearchHit {
        pub conversation_id: String,
        pub score: f32,
        pub matched_via: MatchedVia,
        pub application: Option<String>,
        pub model: String,
        pub created_at: chrono::DateTime<chrono::Utc>,
    }

    /// How a conversation matched a semantic query.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    pub enum MatchedVia {
        Segment {
            hash: SegmentHash,
            segment_type: SegmentType,
        },
        Conversation,
    }

    /// Summary stats from an embed pass.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbedStats {
        pub segments_embedded: u64,
        pub segments_skipped: u64,
        pub segments_already_done: u64,
        pub conversations_embedded: u64,
        pub conversations_already_done: u64,
        pub embedding_model: String,
        pub summary_strategy: String,
    }

    /// One cluster of conversations from `cluster_conversations`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConversationCluster {
        pub cluster_id: u32,
        pub size: usize,
        /// Conversation IDs closest to the cluster centroid (descending).
        pub representative_ids: Vec<String>,
        /// All member conversation IDs.
        pub members: Vec<String>,
    }

    /// One outlier conversation from `find_outliers`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct OutlierConversation {
        pub conversation_id: String,
        /// Distance (1 - cosine) to the nearest centroid. Higher = more isolated.
        pub isolation_score: f32,
        pub application: Option<String>,
        pub model: String,
        pub created_at: chrono::DateTime<chrono::Utc>,
    }
}

#[cfg(feature = "semantic-search")]
pub use semantic::*;
