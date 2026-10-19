# TokenVault — Complete Implementation Specification

## For: Claude Code Implementation
## Language: Rust (with Python + Node.js bindings via PyO3 and napi-rs)
## Storage: Pluggable backend via trait abstraction (filesystem, S3, GCS, Azure Blob, memory)

---

## 1. Project Overview

TokenVault is an embeddable library for compressed storage and retrieval of LLM token sequences. It treats token arrays (the integer sequences LLMs consume and produce) as a first-class data type, applying segment-level deduplication, variable-width integer encoding, and tokenizer-specific dictionary compression to achieve 70–85% storage reduction over naive approaches.

It is distributed as:
- A Rust crate (`tokenvault`)
- A Python package (`tokenvault-py`) via PyO3 + maturin
- A Node.js package (`@tokenvault/core`) via napi-rs

### Core Value Proposition
- **Segment-level dedup**: Conversations are decomposed into typed segments (system_prompt, context, user_turn, assistant_turn, tool_call, tool_result). Identical segments are stored once regardless of how many conversations reference them.
- **Token-aware compression**: Variable-width integer encoding + zstd with token-frequency-trained dictionaries. Tokens follow Zipf's law; the top 256 tokens (~45% of usage) encode in 1 byte instead of 4.
- **Pluggable storage**: A `StorageBackend` trait allows any backing store — local filesystem, S3, GCS, Azure, or in-memory for testing.
- **Structural queryability**: Manifests enable queries by segment type, token counts, models, and applications without scanning raw data.

---

## 2. Project Structure

```
tokenvault/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── tokenvault-core/          # Core library — all logic lives here
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # Public API surface
│   │       ├── types.rs          # Core data types
│   │       ├── vault.rs          # TokenVault main struct (orchestrator)
│   │       ├── segmenter.rs      # Conversation → typed segments
│   │       ├── compression/
│   │       │   ├── mod.rs
│   │       │   ├── varint.rs     # Variable-width token encoding
│   │       │   └── dictionary.rs # zstd dictionary compression
│   │       ├── dedup/
│   │       │   ├── mod.rs
│   │       │   ├── exact.rs      # SHA-256 exact dedup
│   │       │   └── near.rs       # MinHash LSH near-dedup (optional)
│   │       ├── storage/
│   │       │   ├── mod.rs
│   │       │   ├── backend.rs    # StorageBackend trait definition
│   │       │   ├── filesystem.rs # Local filesystem backend
│   │       │   ├── memory.rs     # In-memory backend (testing)
│   │       │   └── s3.rs         # S3-compatible backend
│   │       ├── index/
│   │       │   ├── mod.rs
│   │       │   ├── metadata.rs   # SQLite metadata index
│   │       │   └── analytics.rs  # Token usage analytics queries
│   │       └── export/
│   │           ├── mod.rs
│   │           ├── training.rs   # Training data export (JSONL, HF format)
│   │           └── parquet.rs    # Parquet export for analytics
│   ├── tokenvault-python/        # Python bindings
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── lib.rs
│   └── tokenvault-node/          # Node.js bindings
│       ├── Cargo.toml
│       ├── package.json
│       └── src/
│           └── lib.rs
├── tests/
│   ├── integration/
│   │   ├── roundtrip.rs          # Store → retrieve → verify
│   │   ├── dedup.rs              # Dedup ratio validation
│   │   ├── compression.rs        # Compression ratio benchmarks
│   │   └── analytics.rs          # Query correctness
│   └── fixtures/
│       ├── conversations/        # Sample conversation JSON files
│       └── dictionaries/         # Test compression dictionaries
├── benches/
│   ├── compression.rs            # Compression throughput benchmarks
│   ├── ingest.rs                 # Ingest throughput benchmarks
│   └── retrieval.rs              # Retrieval latency benchmarks
├── examples/
│   ├── basic_usage.rs            # Minimal example
│   ├── with_s3.rs                # S3 backend example
│   ├── analytics.rs              # Running analytics queries
│   └── training_export.rs        # Exporting training data
├── dictionaries/
│   └── README.md                 # How to train custom dictionaries
└── docs/
    ├── ARCHITECTURE.md
    ├── STORAGE_BACKENDS.md
    └── COMPRESSION.md
```

---

## 3. Core Data Types (`types.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A token is an integer ID from a tokenizer vocabulary
pub type Token = u32;

/// Segment types matching LLM conversation structure
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

/// A single segment of a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub segment_type: SegmentType,
    pub tokens: Vec<Token>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Hash of a segment's raw token content (SHA-256, hex-encoded)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegmentHash(pub String);

/// A stored segment reference within a manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    pub segment_type: SegmentType,
    pub hash: SegmentHash,
    pub token_count: u32,
    pub position: u32,
}

/// Conversation manifest — the lightweight pointer structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationManifest {
    pub id: String,
    pub application: Option<String>,
    pub model: String,
    pub tokenizer: String,
    pub total_tokens: u64,
    pub segments: Vec<SegmentRef>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Input format: a conversation as the user provides it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: Option<String>,           // auto-generated if None
    pub application: Option<String>,
    pub model: String,
    pub tokenizer: String,
    pub messages: Vec<Message>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// A single message in a conversation (OpenAI/Anthropic compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,                 // "system", "user", "assistant", "tool"
    pub content: MessageContent,
    pub name: Option<String>,         // for tool calls
    pub tool_call_id: Option<String>,
}

/// Message content can be text (to be tokenized) or pre-tokenized
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Tokens(Vec<Token>),
}

/// Stored segment data (what lives in the storage backend)
#[derive(Debug, Clone)]
pub struct StoredSegment {
    pub hash: SegmentHash,
    pub segment_type: SegmentType,
    pub tokenizer: String,
    pub token_count: u32,
    pub compressed_data: Vec<u8>,     // variable-width encoded + zstd compressed
    pub raw_size: u32,                // uncompressed size in bytes (token_count * 4)
    pub compressed_size: u32,
    pub ref_count: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Configuration for a TokenVault instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultConfig {
    /// Enable/disable near-duplicate detection (MinHash)
    pub enable_near_dedup: bool,
    /// MinHash similarity threshold (0.0–1.0) for near-dedup
    pub near_dedup_threshold: f64,
    /// Enable/disable compression (useful to disable for benchmarking)
    pub enable_compression: bool,
    /// Path to tokenizer-specific zstd dictionaries
    pub dictionary_path: Option<String>,
    /// Tenant ID for multi-tenant isolation
    pub tenant_id: Option<String>,
    /// Enable encryption at rest
    pub enable_encryption: bool,
    /// Encryption key (AES-256-GCM). In production, use a KMS reference instead.
    pub encryption_key: Option<Vec<u8>>,
}

/// Results from analytics queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageStats {
    pub total_tokens: u64,
    pub total_conversations: u64,
    pub unique_segments: u64,
    pub total_segments: u64,           // including duplicates
    pub dedup_ratio: f64,              // 1.0 - (unique / total)
    pub compression_ratio: f64,        // compressed_size / raw_size
    pub storage_bytes: u64,            // actual bytes on disk
    pub naive_bytes: u64,              // what it would be without TokenVault
    pub savings_percentage: f64,
}

/// Per-segment-type breakdown
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

/// Query filters for analytics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyticsQuery {
    pub model: Option<String>,
    pub application: Option<String>,
    pub segment_type: Option<SegmentType>,
    pub date_from: Option<chrono::DateTime<chrono::Utc>>,
    pub date_to: Option<chrono::DateTime<chrono::Utc>>,
    pub group_by: Option<Vec<String>>,  // "model", "application", "segment_type", "day"
}

/// Training data export options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub include_system_prompts: bool,
    pub include_context: bool,
    pub deduplicate_pairs: bool,       // dedup identical user/assistant turn pairs
    pub tokenizer: Option<String>,     // filter by tokenizer
    pub model: Option<String>,         // filter by model
    pub application: Option<String>,
    pub max_conversations: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Jsonl,
    HuggingFace,
    Parquet,
}
```

---

## 4. Storage Backend Trait (`storage/backend.rs`)

This is the core abstraction that enables platform-agnostic storage.

```rust
use async_trait::async_trait;
use crate::types::*;

/// Errors that storage backends can produce
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Segment not found: {0}")]
    SegmentNotFound(String),
    #[error("Conversation not found: {0}")]
    ConversationNotFound(String),
    #[error("Backend error: {0}")]
    BackendError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type StorageResult<T> = Result<T, StorageError>;

/// The pluggable storage backend trait
///
/// Implementors provide the actual persistence layer.
/// TokenVault handles compression, dedup, and segmentation;
/// the backend just stores and retrieves bytes.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    // ── Segment operations ──

    /// Store a compressed segment blob. Idempotent — if hash exists, increment ref_count.
    async fn put_segment(&self, segment: &StoredSegment) -> StorageResult<()>;

    /// Retrieve a compressed segment by hash
    async fn get_segment(&self, hash: &SegmentHash) -> StorageResult<StoredSegment>;

    /// Check if a segment exists (for dedup check without fetching data)
    async fn has_segment(&self, hash: &SegmentHash) -> StorageResult<bool>;

    /// Batch existence check (returns set of hashes that already exist)
    async fn has_segments(&self, hashes: &[SegmentHash]) -> StorageResult<Vec<SegmentHash>>;

    /// Increment reference count for an existing segment
    async fn increment_ref(&self, hash: &SegmentHash) -> StorageResult<()>;

    /// Decrement reference count. Returns true if ref_count reached 0 (eligible for GC).
    async fn decrement_ref(&self, hash: &SegmentHash) -> StorageResult<bool>;

    /// Delete a segment (only called during GC when ref_count == 0)
    async fn delete_segment(&self, hash: &SegmentHash) -> StorageResult<()>;

    // ── Manifest operations ──

    /// Store a conversation manifest
    async fn put_manifest(&self, manifest: &ConversationManifest) -> StorageResult<()>;

    /// Retrieve a conversation manifest by ID
    async fn get_manifest(&self, id: &str) -> StorageResult<ConversationManifest>;

    /// Delete a conversation manifest (does NOT delete segments; caller handles ref counting)
    async fn delete_manifest(&self, id: &str) -> StorageResult<()>;

    /// List conversation IDs with optional filters
    async fn list_conversations(&self, query: &AnalyticsQuery, limit: u64, offset: u64)
        -> StorageResult<Vec<String>>;

    // ── Analytics operations ──

    /// Get aggregate statistics
    async fn get_stats(&self) -> StorageResult<TokenUsageStats>;

    /// Get per-segment-type statistics
    async fn get_segment_type_stats(&self) -> StorageResult<Vec<SegmentTypeStats>>;

    /// Run an analytics query (token usage grouped/filtered)
    async fn query_analytics(&self, query: &AnalyticsQuery)
        -> StorageResult<Vec<HashMap<String, serde_json::Value>>>;

    /// List all unique system prompts (returns hashes + token counts)
    async fn list_unique_system_prompts(&self) -> StorageResult<Vec<(SegmentHash, u32, u64)>>;
    // Returns: (hash, token_count, ref_count)

    /// Find all conversations referencing a specific segment
    async fn find_conversations_by_segment(&self, hash: &SegmentHash)
        -> StorageResult<Vec<String>>;

    // ── Maintenance ──

    /// Garbage collect segments with ref_count == 0
    async fn garbage_collect(&self) -> StorageResult<u64>; // returns count of deleted segments

    /// Get storage size on disk/in backend
    async fn storage_size_bytes(&self) -> StorageResult<u64>;
}
```

### Backend Implementations to Build

1. **`FilesystemBackend`**: Stores segments as files in a content-addressed directory structure (`{root}/segments/aa/bb/aabb1234...`). Manifests in `{root}/manifests/{id}.json`. Metadata index in `{root}/metadata.db` (SQLite). This is the default backend.

2. **`MemoryBackend`**: Everything in `HashMap`s. For testing and short-lived use cases. No persistence.

3. **`S3Backend`**: Segments stored as S3 objects. Manifests stored as S3 objects. Metadata index in a local SQLite file or a separate metadata backend. Uses `aws-sdk-s3` crate. Should work with any S3-compatible store (MinIO, GCS with S3 interop, R2, etc.).

4. **`GcsBackend`**: Google Cloud Storage. Uses `google-cloud-storage` crate.

5. **`AzureBlobBackend`**: Azure Blob Storage. Uses `azure_storage_blobs` crate.

Each cloud backend should:
- Use the same content-addressed key scheme (`segments/{hash_prefix}/{full_hash}`)
- Store manifests at `manifests/{conversation_id}.json`
- Support a local SQLite metadata cache that syncs with the remote store
- Handle retries and exponential backoff for transient failures
- Support configurable bucket/container names and authentication

---

## 5. Segmenter (`segmenter.rs`)

The segmenter decomposes a `Conversation` (list of messages) into typed `Segment`s.

### Logic

```
for each message in conversation.messages:
    match message.role:
        "system"    → SegmentType::SystemPrompt
        "user"      → SegmentType::UserTurn
        "assistant" → SegmentType::AssistantTurn
        "tool"      → if message.tool_call_id.is_some():
                           SegmentType::ToolResult
                       else:
                           SegmentType::ToolCall

    if message.content is Text:
        // Tokenization is NOT handled by TokenVault.
        // The caller must provide pre-tokenized content OR
        // TokenVault stores text as-is (UTF-8 bytes treated as token IDs).
        // For the MVP, require pre-tokenized input (Vec<Token>).
        // Provide a helper that accepts text + tokenizer name and calls
        // an external tokenizer (tiktoken, sentencepiece) if available.

    segment = Segment {
        segment_type,
        tokens: message.content.as_tokens(),
        metadata: extract_metadata(message),
    }
    segments.push(segment)
```

### Context Detection Heuristic

Some applications inject RAG context as part of the user message rather than as a separate message. The segmenter should support an optional `context_delimiter` configuration — a token sequence or text pattern that marks the boundary between injected context and the actual user query within a single message. When detected, the message is split into a `Context` segment and a `UserTurn` segment.

### Tokenization Strategy

TokenVault does NOT include a tokenizer. Token sequences should arrive pre-tokenized. However, provide a `TokenizerAdapter` trait:

```rust
pub trait TokenizerAdapter: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<Token>;
    fn detokenize(&self, tokens: &[Token]) -> String;
    fn vocab_size(&self) -> u32;
    fn name(&self) -> &str;
}
```

Users can implement this trait wrapping `tiktoken-rs`, `tokenizers` (HuggingFace), or any other tokenizer. TokenVault ships with no tokenizer dependency — it's the caller's responsibility.

When `MessageContent::Text` is provided without a tokenizer adapter configured, return an error with a clear message explaining that either pre-tokenized content or a tokenizer adapter is required.

---

## 6. Compression (`compression/`)

### 6.1 Variable-Width Integer Encoding (`varint.rs`)

Encode token IDs using a variable-width scheme based on token magnitude:

```
Token value 0–127:         1 byte   (0xxxxxxx)
Token value 128–16383:     2 bytes  (10xxxxxx xxxxxxxx)
Token value 16384–2097151: 3 bytes  (110xxxxx xxxxxxxx xxxxxxxx)
Token value 2097152+:      4 bytes  (111xxxxx xxxxxxxx xxxxxxxx xxxxxxxx)
```

This is similar to UTF-8 / protobuf varint encoding but optimized for the token value distribution.

Implement:
- `fn encode_tokens(tokens: &[Token]) -> Vec<u8>`
- `fn decode_tokens(data: &[u8]) -> Result<Vec<Token>, DecodeError>`
- These must be perfectly lossless and roundtrip-safe.

### 6.2 Dictionary Compression (`dictionary.rs`)

Apply zstd compression on top of the varint-encoded byte stream, optionally using a pre-trained dictionary.

```rust
pub struct DictionaryCompressor {
    /// Pre-trained zstd dictionary for this tokenizer, if available
    dictionary: Option<zstd::dict::EncoderDictionary<'static>>,
    decompression_dict: Option<zstd::dict::DecoderDictionary<'static>>,
    compression_level: i32,  // default: 3
}
```

Implement:
- `fn compress(&self, varint_data: &[u8]) -> Vec<u8>`
- `fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>, DecompressError>`
- `fn train_dictionary(samples: &[Vec<u8>], dict_size: usize) -> Vec<u8>` — static method that trains a zstd dictionary from sample varint-encoded token sequences

Dictionary training workflow:
1. Collect 10K+ segments of varint-encoded token data (from a representative corpus)
2. Call `zstd::dict::from_samples()` with target dictionary size (default 110KB)
3. Save the dictionary to a file, keyed by tokenizer name
4. Load at runtime based on the `tokenizer` field of incoming conversations

Ship with NO pre-trained dictionaries. Provide a CLI command / API method to train dictionaries from stored data:
```
tokenvault train-dictionary --tokenizer gpt-4 --samples 50000 --output ./dictionaries/gpt4.dict
```

### 6.3 Full Compression Pipeline

```
compress(tokens: &[Token]) -> Vec<u8>:
    1. varint_bytes = varint::encode_tokens(tokens)
    2. compressed = dictionary_compressor.compress(varint_bytes)
    3. return compressed

decompress(data: &[u8]) -> Vec<Token>:
    1. varint_bytes = dictionary_compressor.decompress(data)
    2. tokens = varint::decode_tokens(varint_bytes)
    3. return tokens
```

---

## 7. Deduplication (`dedup/`)

### 7.1 Exact Dedup (`exact.rs`)

```rust
use sha2::{Sha256, Digest};

pub fn hash_segment(tokens: &[Token]) -> SegmentHash {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    SegmentHash(hex::encode(hasher.finalize()))
}
```

The dedup check is simple: hash the segment, check if the hash exists in the backend. If yes, increment ref_count and skip storage. If no, compress and store.

### 7.2 Near-Dedup (`near.rs`) — Optional Feature

Behind a cargo feature flag `near-dedup`.

Use MinHash LSH to detect near-duplicate segments. This catches cases like:
- System prompts with interpolated user names: "You are helping {name}..."
- RAG chunks with minor formatting differences
- Tool calls with similar but not identical arguments

Implementation:
1. Convert token sequence to a set of n-grams (n=3 or n=5 at the token level)
2. Compute MinHash signature (128 hash functions)
3. LSH band hashing to find candidate pairs
4. For candidates above the similarity threshold, store as: reference to canonical segment + delta patch

Delta patch format:
```rust
pub struct SegmentDelta {
    pub canonical_hash: SegmentHash,
    pub operations: Vec<DeltaOp>,
}

pub enum DeltaOp {
    Keep { offset: u32, length: u32 },
    Insert { tokens: Vec<Token> },
    Delete { offset: u32, length: u32 },
}
```

This is a Phase 2 feature. The MVP should implement the trait and hash computation but can skip delta storage initially.

---

## 8. Metadata Index (`index/`)

### 8.1 SQLite Metadata Store (`metadata.rs`)

Use `rusqlite` for the embedded metadata index. This index lives alongside the storage backend and provides queryability.

Schema:

```sql
CREATE TABLE conversations (
    id              TEXT PRIMARY KEY,
    application     TEXT,
    model           TEXT NOT NULL,
    tokenizer       TEXT NOT NULL,
    total_tokens    INTEGER NOT NULL,
    segment_count   INTEGER NOT NULL,
    created_at      TEXT NOT NULL,      -- ISO 8601
    metadata_json   TEXT                -- arbitrary JSON
);

CREATE TABLE segment_refs (
    conversation_id TEXT NOT NULL,
    segment_hash    TEXT NOT NULL,
    segment_type    TEXT NOT NULL,
    position        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX idx_segref_conv ON segment_refs(conversation_id);
CREATE INDEX idx_segref_hash ON segment_refs(segment_hash);
CREATE INDEX idx_segref_type ON segment_refs(segment_type);

CREATE TABLE segments_meta (
    hash            TEXT PRIMARY KEY,
    segment_type    TEXT NOT NULL,
    tokenizer       TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    compressed_size INTEGER NOT NULL,
    raw_size        INTEGER NOT NULL,
    ref_count       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_segmeta_type ON segments_meta(segment_type);
CREATE INDEX idx_segmeta_refs ON segments_meta(ref_count);
```

### 8.2 Analytics Queries (`analytics.rs`)

Pre-built queries exposed through the API:

1. **Total stats**: `SELECT SUM(total_tokens), COUNT(*), ... FROM conversations`
2. **Dedup ratio**: `SELECT segment_type, COUNT(*) as unique, SUM(ref_count) as total FROM segments_meta GROUP BY segment_type`
3. **Tokens by model by day**: `SELECT model, DATE(created_at), SUM(token_count) FROM segment_refs JOIN conversations ... GROUP BY model, DATE(created_at)`
4. **Unique system prompts**: `SELECT hash, token_count, ref_count FROM segments_meta WHERE segment_type = 'system_prompt' ORDER BY ref_count DESC`
5. **Conversations using segment X**: `SELECT conversation_id FROM segment_refs WHERE segment_hash = ?`
6. **Cost attribution**: `SELECT application, segment_type, SUM(token_count) FROM segment_refs JOIN conversations ... GROUP BY application, segment_type`
7. **Compression stats**: `SELECT SUM(compressed_size) as actual, SUM(raw_size) as naive FROM segments_meta`

---

## 9. Main Vault Orchestrator (`vault.rs`)

```rust
pub struct TokenVault<B: StorageBackend> {
    backend: B,
    config: VaultConfig,
    compressor: CompressionPipeline,
    index: MetadataIndex,           // SQLite-based
    tokenizer: Option<Box<dyn TokenizerAdapter>>,
}
```

### Public API Methods

```rust
impl<B: StorageBackend> TokenVault<B> {
    // ── Construction ──
    pub async fn new(backend: B, config: VaultConfig) -> Result<Self, VaultError>;
    pub async fn open(backend: B, config: VaultConfig) -> Result<Self, VaultError>;

    // ── Core Operations ──

    /// Store a conversation. Returns the conversation ID and storage metrics.
    pub async fn store(&self, conversation: Conversation) -> Result<StoreResult, VaultError>;

    /// Retrieve a full conversation (all segments reassembled)
    pub async fn retrieve(&self, id: &str) -> Result<RetrievedConversation, VaultError>;

    /// Retrieve specific segments of a conversation
    pub async fn retrieve_segments(
        &self,
        id: &str,
        segment_type: Option<SegmentType>,
    ) -> Result<Vec<RetrievedSegment>, VaultError>;

    /// Retrieve a single segment by its content hash
    pub async fn retrieve_segment_by_hash(
        &self,
        hash: &SegmentHash,
    ) -> Result<RetrievedSegment, VaultError>;

    /// Delete a conversation (decrements refs, does not immediately GC segments)
    pub async fn delete(&self, id: &str) -> Result<(), VaultError>;

    // ── Analytics ──

    /// Get overall storage statistics
    pub async fn stats(&self) -> Result<TokenUsageStats, VaultError>;

    /// Get per-segment-type statistics
    pub async fn segment_stats(&self) -> Result<Vec<SegmentTypeStats>, VaultError>;

    /// Run a filtered analytics query
    pub async fn query(&self, query: AnalyticsQuery)
        -> Result<Vec<HashMap<String, serde_json::Value>>, VaultError>;

    /// List all unique system prompts
    pub async fn list_system_prompts(&self) -> Result<Vec<SystemPromptInfo>, VaultError>;

    /// Find conversations that contain a specific segment
    pub async fn find_by_segment(&self, hash: &SegmentHash) -> Result<Vec<String>, VaultError>;

    // ── Export ──

    /// Export training data pairs
    pub async fn export_training_data(
        &self,
        config: ExportConfig,
        writer: &mut dyn std::io::Write,
    ) -> Result<ExportStats, VaultError>;

    // ── Maintenance ──

    /// Run garbage collection on orphaned segments
    pub async fn gc(&self) -> Result<u64, VaultError>;

    /// Train a compression dictionary from stored data
    pub async fn train_dictionary(
        &self,
        tokenizer: &str,
        sample_count: usize,
    ) -> Result<Vec<u8>, VaultError>;

    // ── Configuration ──

    /// Set a tokenizer adapter for text-to-token conversion
    pub fn set_tokenizer(&mut self, tokenizer: Box<dyn TokenizerAdapter>);
}
```

### Store Pipeline (internal)

```rust
async fn store_internal(&self, conversation: Conversation) -> Result<StoreResult, VaultError> {
    // 1. Generate conversation ID if not provided
    let id = conversation.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // 2. Segment the conversation
    let segments = self.segmenter.segment(&conversation)?;

    // 3. For each segment:
    let mut segment_refs = Vec::new();
    let mut new_segments = 0u64;
    let mut deduped_segments = 0u64;
    let mut bytes_saved = 0u64;

    for (position, segment) in segments.iter().enumerate() {
        // 3a. Hash the raw tokens
        let hash = dedup::exact::hash_segment(&segment.tokens);

        // 3b. Check if segment already exists
        if self.backend.has_segment(&hash).await? {
            // Dedup hit — just increment ref count
            self.backend.increment_ref(&hash).await?;
            deduped_segments += 1;
            bytes_saved += (segment.tokens.len() as u64) * 4;
        } else {
            // 3c. Compress: varint encode → zstd compress
            let compressed = self.compressor.compress(&segment.tokens)?;

            // 3d. Store the compressed segment
            let stored = StoredSegment {
                hash: hash.clone(),
                segment_type: segment.segment_type.clone(),
                tokenizer: conversation.tokenizer.clone(),
                token_count: segment.tokens.len() as u32,
                compressed_data: compressed.clone(),
                raw_size: (segment.tokens.len() as u32) * 4,
                compressed_size: compressed.len() as u32,
                ref_count: 1,
                created_at: chrono::Utc::now(),
            };
            self.backend.put_segment(&stored).await?;
            new_segments += 1;
        }

        // 3e. Build segment reference
        segment_refs.push(SegmentRef {
            segment_type: segment.segment_type.clone(),
            hash,
            token_count: segment.tokens.len() as u32,
            position: position as u32,
        });
    }

    // 4. Create and store manifest
    let manifest = ConversationManifest {
        id: id.clone(),
        application: conversation.application,
        model: conversation.model,
        tokenizer: conversation.tokenizer,
        total_tokens: segment_refs.iter().map(|s| s.token_count as u64).sum(),
        segments: segment_refs,
        created_at: chrono::Utc::now(),
        metadata: conversation.metadata,
    };
    self.backend.put_manifest(&manifest).await?;

    // 5. Update metadata index
    self.index.index_conversation(&manifest).await?;

    // 6. Return results
    Ok(StoreResult {
        id,
        total_segments: new_segments + deduped_segments,
        new_segments,
        deduped_segments,
        bytes_saved,
        compression_ratio: /* computed from stored data */,
    })
}
```

---

## 10. Export Module (`export/`)

### Training Data Export (`training.rs`)

Exports deduplicated user/assistant turn pairs suitable for fine-tuning.

```rust
pub async fn export_jsonl(
    vault: &TokenVault<impl StorageBackend>,
    config: &ExportConfig,
    writer: &mut dyn Write,
) -> Result<ExportStats, ExportError> {
    // 1. Query conversations matching filters
    // 2. For each conversation, load manifest
    // 3. Extract user_turn + assistant_turn pairs
    // 4. If deduplicate_pairs: track seen (user_hash, assistant_hash) pairs, skip dupes
    // 5. If !include_system_prompts: omit system prompt segments
    // 6. If !include_context: omit context segments
    // 7. Write each pair as a JSONL line:
    //    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    // 8. Return stats: total_pairs, unique_pairs, tokens_exported
}
```

### Parquet Export (`parquet.rs`)

For analytics use cases. Uses the `parquet` crate to write columnar data:

Columns: `conversation_id`, `model`, `application`, `segment_type`, `token_count`, `segment_hash`, `created_at`

This enables downstream analytics in tools like DuckDB, Pandas, or Spark.

---

## 11. Python Bindings (`tokenvault-python/`)

Use PyO3 + maturin for building the Python wheel.

### Python API Surface

```python
import tokenvault

# Create a vault with filesystem backend
vault = tokenvault.Vault(
    backend="filesystem",
    path="/path/to/vault",
    config=tokenvault.Config(
        enable_compression=True,
        enable_near_dedup=False,
    )
)

# Or with S3
vault = tokenvault.Vault(
    backend="s3",
    bucket="my-tokenvault-bucket",
    prefix="conversations/",
    region="us-east-1",
)

# Store a conversation (OpenAI format)
result = vault.store({
    "model": "gpt-4",
    "tokenizer": "cl100k_base",
    "messages": [
        {"role": "system", "content_tokens": [1234, 5678, ...]},
        {"role": "user", "content_tokens": [9012, 3456, ...]},
        {"role": "assistant", "content_tokens": [7890, 1234, ...]},
    ]
})
print(result.id, result.dedup_ratio, result.compression_ratio)

# Retrieve
conversation = vault.retrieve(result.id)
for segment in conversation.segments:
    print(segment.type, segment.token_count, segment.tokens[:10])

# Retrieve only assistant responses
responses = vault.retrieve_segments(result.id, segment_type="assistant_turn")

# Analytics
stats = vault.stats()
print(f"Storage savings: {stats.savings_percentage:.1f}%")
print(f"Dedup ratio: {stats.dedup_ratio:.2f}")

# System prompt audit
prompts = vault.list_system_prompts()
for p in prompts:
    print(f"Hash: {p.hash}, used in {p.ref_count} conversations")

# Export training data
vault.export_training_data(
    path="training_pairs.jsonl",
    format="jsonl",
    deduplicate_pairs=True,
    include_system_prompts=False,
)

# Garbage collection
deleted = vault.gc()
print(f"Cleaned up {deleted} orphaned segments")
```

### PyO3 Implementation Notes

- Use `#[pyclass]` for `Vault`, `StoreResult`, `RetrievedConversation`, `TokenUsageStats`
- Use `#[pymethods]` for all public API methods
- Accept Python dicts for conversation input (serde_json round-trip)
- Return Python dicts for analytics results
- Use `pyo3-asyncio` or `tokio::runtime::Runtime::block_on` for async→sync bridging
- The `maturin` build tool handles wheel packaging

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "tokenvault"
description = "Compressed storage for LLM token sequences"
requires-python = ">=3.8"

[tool.maturin]
features = ["python-bindings"]
```

---

## 12. Node.js Bindings (`tokenvault-node/`)

Use napi-rs for building the npm package.

### Node.js API Surface

```typescript
import { Vault, Config } from '@tokenvault/core';

const vault = new Vault({
  backend: 'filesystem',
  path: '/path/to/vault',
  config: {
    enableCompression: true,
    enableNearDedup: false,
  },
});

// Store
const result = await vault.store({
  model: 'claude-sonnet-4-20250514',
  tokenizer: 'claude-v3',
  messages: [
    { role: 'system', contentTokens: [1234, 5678] },
    { role: 'user', contentTokens: [9012, 3456] },
    { role: 'assistant', contentTokens: [7890, 1234] },
  ],
});

// Retrieve
const conversation = await vault.retrieve(result.id);

// Stats
const stats = await vault.stats();
console.log(`Savings: ${stats.savingsPercentage.toFixed(1)}%`);
```

### napi-rs Implementation Notes

- Use `#[napi]` attribute macros
- Async methods return `Promise<T>` via napi's async support
- Use `serde_json::Value` ↔ `napi::JsObject` conversion
- Package via napi-rs's built-in npm packaging

---

## 13. Cargo Dependencies

### `tokenvault-core/Cargo.toml`

```toml
[package]
name = "tokenvault-core"
version = "0.1.0"
edition = "2021"

[dependencies]
# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Hashing
sha2 = "0.10"
hex = "0.4"

# Compression
zstd = "0.13"

# Database
rusqlite = { version = "0.31", features = ["bundled"] }

# Async
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"

# Time
chrono = { version = "0.4", features = ["serde"] }

# IDs
uuid = { version = "1", features = ["v4"] }

# Error handling
thiserror = "1"

# Logging
tracing = "0.1"

# Cloud storage (optional)
aws-sdk-s3 = { version = "1", optional = true }
google-cloud-storage = { version = "0.20", optional = true }
azure_storage_blobs = { version = "0.20", optional = true }

# Near-dedup (optional)
# MinHash implementation — can use the `gaoya` crate or implement from scratch
gaoya = { version = "0.2", optional = true }

# Export
parquet = { version = "53", optional = true }

[features]
default = ["filesystem"]
filesystem = []
s3 = ["aws-sdk-s3"]
gcs = ["google-cloud-storage"]
azure = ["azure_storage_blobs"]
near-dedup = ["gaoya"]
export-parquet = ["parquet"]
python-bindings = []
node-bindings = []
all-backends = ["filesystem", "s3", "gcs", "azure"]
full = ["all-backends", "near-dedup", "export-parquet"]
```

---

## 14. Testing Strategy

### Unit Tests (per module)
- `varint.rs`: Roundtrip encoding for all token ranges (0, 127, 128, 16383, 16384, max_u32)
- `dictionary.rs`: Compress/decompress roundtrip with and without dictionary
- `exact.rs`: Hash stability (same tokens → same hash, different tokens → different hash)
- `segmenter.rs`: Correct segmentation of multi-turn conversations with all role types
- `metadata.rs`: SQL query correctness for all analytics queries

### Integration Tests (`tests/integration/`)
- **Roundtrip**: Store conversation → retrieve → verify tokens match exactly
- **Dedup**: Store same conversation twice → verify segment ref_counts incremented, storage size unchanged
- **Compression benchmarks**: Measure compression ratios on fixture data, assert within expected ranges
- **Analytics**: Store multiple conversations → run analytics queries → verify aggregated results
- **Delete + GC**: Store → delete → GC → verify segments removed when ref_count reaches 0
- **Multi-backend**: Run the same test suite against Filesystem and Memory backends

### Benchmark Tests (`benches/`)
Use `criterion` crate:
- Compression throughput (tokens/sec)
- Dedup check throughput (hashes/sec)
- Store pipeline throughput (conversations/sec)
- Retrieve latency (p50, p95, p99)

### Fixture Data (`tests/fixtures/`)
Create JSON files representing realistic conversations:
- `simple_chat.json`: 2-turn conversation, 500 tokens
- `long_conversation.json`: 20-turn conversation, 10K tokens
- `agent_loop.json`: 8-step tool-calling agent, 15K tokens with repeated context
- `shared_system_prompt.json`: 100 conversations sharing the same system prompt
- `rag_conversation.json`: conversation with RAG context chunks

---

## 15. CLI Tool

Include a simple CLI binary in the workspace for testing and dictionary training:

```
tokenvault store --file conversation.json --backend fs --path ./vault
tokenvault retrieve --id conv_abc123 --backend fs --path ./vault
tokenvault stats --backend fs --path ./vault
tokenvault gc --backend fs --path ./vault
tokenvault train-dictionary --tokenizer gpt-4 --samples 50000 --backend fs --path ./vault --output ./dictionaries/gpt4.dict
tokenvault export --format jsonl --output training.jsonl --backend fs --path ./vault
```

---

## 16. Performance Targets

| Operation | Target (p95) |
|-----------|-------------|
| Store conversation (5K tokens, high dedup) | < 5ms |
| Store conversation (5K tokens, no dedup) | < 20ms |
| Retrieve full conversation (5K tokens) | < 10ms |
| Retrieve single segment by hash | < 2ms |
| Dedup check (batch of 10 hashes) | < 1ms |
| Analytics query (1 day, 100K conversations) | < 100ms |
| Bulk ingest throughput | 10K conversations/sec |
| Compression ratio (varint + zstd, no dict) | 30–40% of int32 |
| Compression ratio (varint + zstd + dict) | 20–30% of int32 |
| Dedup ratio (system prompts) | 95–99% |
| Combined storage savings | 70–85% |

---

## 17. Implementation Order

Build in this order, testing each layer before moving to the next:

1. **`types.rs`** — All data types
2. **`compression/varint.rs`** — Variable-width encoding with exhaustive roundtrip tests
3. **`compression/dictionary.rs`** — zstd compression wrapper
4. **`dedup/exact.rs`** — SHA-256 hashing
5. **`storage/backend.rs`** — Trait definition
6. **`storage/memory.rs`** — In-memory backend (enables testing everything else)
7. **`segmenter.rs`** — Conversation decomposition
8. **`index/metadata.rs`** — SQLite metadata store
9. **`vault.rs`** — Main orchestrator (store + retrieve pipeline)
10. **Integration tests** — Roundtrip, dedup, compression
11. **`storage/filesystem.rs`** — Filesystem backend
12. **`index/analytics.rs`** — Analytics queries
13. **`export/training.rs`** — JSONL export
14. **`dedup/near.rs`** — MinHash near-dedup (behind feature flag)
15. **`export/parquet.rs`** — Parquet export (behind feature flag)
16. **`storage/s3.rs`** — S3 backend (behind feature flag)
17. **Python bindings** — PyO3 wrapper
18. **Node.js bindings** — napi-rs wrapper
19. **CLI tool** — Binary for testing and ops
20. **Benchmarks** — criterion benchmarks
21. **Cloud backends** (GCS, Azure) — behind feature flags

---

## 18. Key Design Decisions & Constraints

1. **No tokenizer dependency**: TokenVault stores and retrieves token integers. It does not tokenize or detokenize text. This keeps the library lean and avoids pulling in massive tokenizer models. Users provide pre-tokenized data or supply their own `TokenizerAdapter`.

2. **Sync metadata, async storage**: The SQLite metadata index is synchronous (rusqlite). The storage backend trait is async (for cloud backends). The vault orchestrator bridges these with `spawn_blocking` for SQLite operations.

3. **Tenant-scoped dedup**: In multi-tenant mode, segment hashes are prefixed with the tenant ID before storage. Two tenants with identical system prompts do NOT share storage. This prevents information leakage through hash probing.

4. **Compression is optional**: The `enable_compression` config flag allows disabling compression for benchmarking or for backends that apply their own compression (e.g., S3 with server-side encryption).

5. **Segment immutability**: Once stored, a segment's content never changes. If a conversation is "edited," new segments are created and the manifest is updated. Old segments are decremented and eventually GC'd.

6. **Manifest-first architecture**: The manifest is the source of truth for conversation structure. Segments are content-addressed blobs. This separation enables cheap conversation-level operations (list, delete, query metadata) without touching segment data.

7. **No streaming ingest in MVP**: Conversations must be complete before storage. Streaming ingest (storing segments as they're generated token-by-token) is a future enhancement.

8. **Dictionary training is offline**: Dictionary training reads existing stored data and produces a dictionary file. It does not run during normal store/retrieve operations. Dictionaries are loaded at vault initialization.
