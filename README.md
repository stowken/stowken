# Stowken

**Compressed, deduplicated storage for LLM token sequences.**

Stowken is a storage engine purpose-built for LLM conversation data. Instead of writing raw token arrays to disk or a blob store, it applies content-addressing, deduplication, and compression before anything hits storage — then reassembles the original token arrays on retrieval with zero information loss.

Typical savings on production workloads: **40–70%** from deduplication alone, reaching **55–85%** with compression and a trained dictionary.

---

## Table of contents

- [How it works](#how-it-works)
  - [Segmentation](#1-segmentation)
  - [Content-addressing and exact deduplication](#2-content-addressing-and-exact-deduplication)
  - [Compression pipeline](#3-compression-pipeline)
  - [Storage backends](#4-storage-backends)
  - [Metadata index](#5-metadata-index)
- [Getting started](#getting-started)
  - [Rust](#rust)
  - [Python](#python)
  - [Node.js](#nodejs)
  - [CLI](#cli)
- [Configuration](#configuration)
- [Enterprise features](#enterprise-features)
  - [Near-deduplication (MinHash LSH)](#near-deduplication-minhash-lsh)
  - [zstd dictionary training](#zstd-dictionary-training)
  - [Encryption at rest](#encryption-at-rest)
  - [Multi-tenant isolation](#multi-tenant-isolation)
  - [AWS S3 backend](#aws-s3-backend)
- [Compression benchmarks](#compression-benchmarks)
- [Running tests](#running-tests)
- [Workspace structure](#workspace-structure)
- [License](#license)

---

## How it works

### 1. Segmentation

When a conversation arrives, Stowken splits it into **typed segments** based on message role:

| Segment type     | Role                             |
|------------------|----------------------------------|
| `system_prompt`  | `"role": "system"` messages      |
| `user_turn`      | `"role": "user"` messages        |
| `assistant_turn` | `"role": "assistant"` messages   |
| `context`        | `"role": "tool"` / function call results |

Each segment is treated as an independent unit. A 1,000-conversation corpus where every conversation starts with the same system prompt only stores that system prompt **once** — all 1,000 manifests reference the same content-addressed blob.

Content can be provided as plain text (tokenized automatically using the conversation's `tokenizer` field) or as a pre-tokenized `u32` array.

### 2. Content-addressing and exact deduplication

Every segment is hashed with **SHA-256** before storage. When a segment arrives:

- If the hash already exists in the backend → increment the segment's reference count (no bytes written, cost is O(1)).
- If the hash is new → compress and write the segment data.

This reference-counting model means deletion is also safe: decrementing a ref-count to zero triggers garbage collection of the underlying blob. All of this is atomic per-segment via a backend-level mutex so concurrent writers can't corrupt ref-counts.

A `ConversationManifest` is written for each stored conversation. It records the ordered list of segment hashes (not the data itself) plus metadata. Retrieval reads the manifest, fetches each referenced segment, decompresses, and returns the token arrays.

### 3. Compression pipeline

Each segment passes through a two-stage pipeline before storage:

**Stage 1 — Varint encoding**

Raw `u32` token IDs are encoded as variable-length integers: tokens below 128 use 1 byte, tokens below 16,384 use 2 bytes, etc. For natural-language BPE vocabularies (where lower token IDs are more frequent), this alone saves ~40% over 4-byte-per-token raw storage.

**Stage 2 — zstd compression**

The varint stream is compressed with [zstd](https://github.com/facebook/zstd). The default level is 3 (fast, good compression). Level 9 is available for archive workloads. zstd adds another 5–10% savings on top of varint encoding for typical token distributions.

**Stage 3 (optional) — zstd dictionary**

After accumulating a warm-up corpus, Stowken can train a [zstd dictionary](https://facebook.github.io/zstd/zstd_manual.html#Chapter10) on representative segments. The dictionary encodes the statistical structure of your specific token distribution, giving zstd a head start that improves compression by another 5–15 percentage points — particularly valuable for domain-specific models where token frequencies differ significantly from general text.

Dictionaries are backward-compatible: frames compressed without a dictionary can still be decompressed after a dictionary is applied, and the decompressor automatically detects whether a dictionary was used.

### 4. Storage backends

Stowken uses a trait-based backend (`StorageBackend`) so the same API works across different storage systems:

| Backend              | Package                  | Use case                                         |
|----------------------|--------------------------|--------------------------------------------------|
| `MemoryBackend`      | `stowken` (core)      | Testing, ephemeral caches, benchmarking          |
| `FilesystemBackend`  | `stowken` (core)      | Single-node persistent storage                   |
| `S3Backend`          | `stowken-enterprise`  | Cloud-native, horizontally scalable storage      |

`FilesystemBackend` uses a content-addressed layout:

```
vault-root/
├── segments/
│   └── aa/bb/aabb...cc   # Segment data (first 4 hex chars = 2 subdirs)
│       aabb...cc.ref      # Ref-count sidecar (JSON)
└── manifests/
    └── conv-001.json      # Conversation manifest
```

`S3Backend` mirrors this layout under a configurable prefix with `.meta` JSON sidecars for segment metadata (ref-count, token count, sizes, etc.).

### 5. Metadata index

An optional **SQLite index** (`metadata.db`) tracks aggregate statistics without scanning segment files:

- Total tokens stored, unique vs. total segment counts, dedup ratio
- Per-segment-type breakdowns (how many unique system prompts, what's the dedup ratio on user turns, etc.)
- Conversation-level metadata for filtering and analytics queries

The index is written in a single transaction per `store()` call, batching all segment operations. It can be rebuilt from manifests at any time and is never the source of truth for segment data.

---

## Getting started

### Rust

```toml
[dependencies]
stowken = "0.1"
tokio = { version = "1", features = ["full"] }
```

**Store and retrieve:**

```rust
use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;

    let conversation = Conversation {
        id: Some("conv-001".into()),
        model: "gpt-4".into(),
        tokenizer: "cl100k_base".into(),
        application: Some("my-app".into()),
        messages: vec![
            Message {
                role: "system".into(),
                content: MessageContent::Text("You are a helpful assistant.".into()),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "user".into(),
                content: MessageContent::Text("What is 2 + 2?".into()),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "assistant".into(),
                content: MessageContent::Text("4".into()),
                name: None,
                tool_call_id: None,
            },
        ],
        metadata: None,
    };

    let result = vault.store(conversation).await?;
    println!("id={} new={} deduped={} ratio={:.2}",
        result.id, result.new_segments, result.deduped_segments, result.compression_ratio);

    let retrieved = vault.retrieve("conv-001").await?;
    for seg in &retrieved.segments {
        println!("  {:?}: {} tokens", seg.segment_type, seg.token_count);
    }

    let stats = vault.stats().await?;
    println!("saved {:.1}% total ({} unique segments)",
        stats.savings_percentage, stats.unique_segments);

    Ok(())
}
```

**Pre-tokenized content** — if you already have token IDs from tiktoken or another tokenizer, pass them directly to bypass the built-in tokenizer:

```rust
Message {
    role: "system".into(),
    content: MessageContent::Tokens(vec![9125, 25, 1472, 527, 264, 11190, 18328, 13]),
    name: None,
    tool_call_id: None,
}
```

**Persistent storage:**

```rust
use stowken::storage::FilesystemBackend;

let backend = FilesystemBackend::new("/var/lib/stowken").await?;
let vault = Stowken::open(
    backend,
    StowkenConfig::default(),
    "/var/lib/stowken/meta.db",  // SQLite metadata index
).await?;
```

**Delete and garbage collect:**

```rust
vault.delete("conv-001").await?;
let freed = vault.gc().await?;  // returns number of zero-ref segments deleted
println!("GC freed {freed} segments");
```

**Export training data:**

```rust
use stowken::types::{ExportConfig, ExportFormat};

let config = ExportConfig {
    format: ExportFormat::Jsonl,
    include_system_prompts: false,
    include_context: false,
    deduplicate_pairs: true,
    tokenizer: None,
    model: Some("gpt-4".into()),
    application: None,
    max_conversations: Some(10_000),
};

let mut output = std::io::stdout();
let stats = vault.export_training_data(config, &mut output).await?;
println!("exported {} pairs ({} unique)", stats.total_pairs, stats.unique_pairs);
```

### Python

Build the native module first (requires [maturin](https://maturin.rs)):

```sh
cd crates/stowken-python
maturin develop          # editable install into current venv
# or:
maturin build --release  # build a distributable wheel
```

```python
import stowken

vault = stowken.Stowken()

# Store a conversation
result = vault.store({
    "model": "gpt-4",
    "tokenizer": "cl100k_base",
    "messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ],
})
print(f"id={result.id} new={result.new_segments} deduped={result.deduped_segments}")

# Retrieve
retrieved = vault.retrieve(result.id)
for seg in retrieved["segments"]:
    print(f"  {seg['segment_type']}: {seg['token_count']} tokens")

# Stats
stats = vault.stats()
print(f"saved {stats.savings_percentage:.1f}% ({stats.storage_bytes} bytes on disk)")

# Pass pre-tokenized content
result = vault.store({
    "model": "gpt-4",
    "messages": [
        {"role": "system", "tokens": [9125, 25, 1472, 527, 264, 11190, 18328, 13]},
        {"role": "user",   "tokens": [3923, 374, 220, 17, 489, 220, 17, 30]},
    ],
})

# GC
freed = vault.gc()
print(f"GC freed {freed} segments")
```

**StoreResult fields:**
- `id` — assigned conversation ID (matches input `id` if provided, otherwise a UUID)
- `total_segments` — total segments in this conversation
- `new_segments` — segments stored for the first time
- `deduped_segments` — segments that already existed (ref-count incremented)
- `bytes_saved` — bytes avoided by exact deduplication
- `compression_ratio` — compressed bytes / raw bytes (lower is better)

**TokenUsageStats fields:**
- `total_conversations`, `total_tokens`, `unique_segments`, `total_segments`
- `dedup_ratio` — fraction of references that were deduped (0 = no dedup, 1 = all deduped)
- `compression_ratio` — storage bytes / naive bytes
- `storage_bytes` — actual bytes on disk/memory
- `savings_percentage` — percentage reduction vs naive storage

Run tests: `pytest python/tests/`

### Node.js

Build the native module first:

```sh
cd crates/stowken-node
npm run build
```

```js
const { Stowken } = require('stowken');

const vault = new Stowken();

// Store
const result = await vault.store({
  model: 'gpt-4',
  tokenizer: 'cl100k_base',
  messages: [
    { role: 'system',    content: 'You are a helpful assistant.' },
    { role: 'user',      content: 'What is 2 + 2?' },
    { role: 'assistant', content: '4' },
  ],
});
console.log(`id=${result.id} new=${result.newSegments} deduped=${result.dedupedSegments}`);

// Retrieve
const conv = await vault.retrieve(result.id);
for (const seg of conv.segments) {
  console.log(`  ${seg.segmentType}: ${seg.tokenCount} tokens`);
}

// Stats
const stats = await vault.stats();
console.log(`saved ${stats.savingsPercentage.toFixed(1)}% (${stats.storageBytes} bytes)`);

// Segment breakdown
const segStats = await vault.segmentStats();
for (const s of segStats) {
  console.log(`${s.segmentType}: ${s.uniqueCount} unique, dedup=${(s.dedupRatio * 100).toFixed(0)}%`);
}

// Compression disabled
const rawVault = new Stowken(false);

// GC after delete
await vault.delete(result.id);
const freed = await vault.gc();
console.log(`freed ${freed} segments`);
```

**Note on field names:** The Node.js binding uses camelCase. `new_segments` → `newSegments`, `total_conversations` → `totalConversations`, `storage_bytes` → `storageBytes`, etc.

Run tests: `node test.js`

### CLI

```sh
# Install (from workspace root)
cargo install --path crates/stowken-cli

# Data directory: defaults to ~/.local/share/stowken (Linux/Mac)
# or %APPDATA%\stowken (Windows). Override with --data-dir or STOWKEN_DATA_DIR.

# Store from stdin
echo '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}' \
  | stowken store -

# Store from a file, assign an ID, tag with application
stowken store conversation.json --id conv-001 --application my-app

# Store and get JSON result (for scripting)
stowken store --json conversation.json

# Retrieve by ID (human-readable table)
stowken retrieve conv-001

# Retrieve as JSON (for scripting)
stowken retrieve conv-001 --json

# Retrieve raw token arrays per segment
stowken retrieve conv-001 --segments

# Storage statistics
stowken stats
stowken stats --json

# Per-segment-type breakdown
stowken segment-stats
stowken segment-stats --json

# Export training pairs
stowken export --format jsonl --output training.jsonl
stowken export --format huggingface --model gpt-4 --deduplicate

# Delete a conversation
stowken delete conv-001         # prompts for confirmation
stowken delete conv-001 --yes   # skip prompt

# Run garbage collection
stowken gc            # deletes zero-ref segments
stowken gc --dry-run  # report without deleting

# Show resolved data directory
stowken data-dir

# Disable compression
stowken --no-compression store conversation.json
```

---

## Configuration

`StowkenConfig` controls all runtime behaviour:

```rust
use stowken::types::StowkenConfig;

let config = StowkenConfig {
    // Enable zstd compression (default: true).
    enable_compression: true,

    // Path to a persisted zstd dictionary.
    // If this file exists, it is loaded on startup so old compressed frames
    // remain decompressible even after a dictionary is applied to new stores.
    dictionary_path: Some("/var/lib/stowken/dict.zstd".into()),

    // Automatically train a dictionary after this many new segments are stored.
    // None = no auto-training (default).
    auto_train_dictionary_after: Some(500),

    // Number of existing segments to sample when training the dictionary.
    // More samples → better dictionary; training is typically done in <100ms.
    dictionary_sample_count: 200,

    // Tenant ID for tenant-scoped segment hashing (enterprise feature).
    tenant_id: None,
};
```

---

## Enterprise features

### Near-deduplication (MinHash LSH)

Standard exact deduplication only catches byte-for-byte identical segments. Near-deduplication catches segments that are *almost* the same:

- A system prompt with an interpolated user name: `"Hello, Alice!"` vs `"Hello, Bob!"`
- A RAG context with slightly different retrieved chunks
- Tool call responses with minor formatting differences

**How it works:**

1. Each token sequence is reduced to a 128-element **MinHash signature** using token-level bigram shingles and 128 independent hash functions.
2. The signature is split into 16 **LSH bands** of 8 rows each. Two segments are *candidates* if any band hash collides (probability > 99% for Jaccard similarity ≥ 0.85 with the default configuration).
3. If a candidate pair exceeds the similarity threshold, the variant is stored as a **delta** (edit script) against the canonical segment rather than a full token array. Deltas typically save 80–95% of the variant's storage cost.
4. The delta is **embedded in the conversation manifest** as JSON, so it survives process restarts — a fresh `EnterpriseStowken` can reconstruct variant tokens without any in-memory state.

```rust
use stowken_enterprise::enterprise_vault::EnterpriseStowken;
use stowken::{storage::FilesystemBackend, types::StowkenConfig};

let backend = FilesystemBackend::new("/var/lib/stowken").await?;
let vault = EnterpriseStowken::open(
    backend,
    StowkenConfig::default(),
    "/var/lib/stowken/meta.db",
    0.85,  // Jaccard similarity threshold; lower → more aggressive dedup
).await?;

let result = vault.store(conversation).await?;
println!(
    "new={} deduped={} near_dedup={} saved={}B",
    result.core.new_segments,
    result.core.deduped_segments,
    result.near_dedup_segments,
    result.near_dedup_bytes_saved,
);

// Retrieval is transparent — near-dup reconstruction happens automatically.
let retrieved = vault.retrieve("conv-001").await?;

// Near-dedup statistics
let stats = vault.near_dedup_stats();
println!("LSH candidates={} accepted={}", stats.lsh_candidates, stats.accepted);
```

**Tuning the threshold:**
- `0.95+` — very conservative; only catches nearly identical segments (1–2 token changes in 100 tokens)
- `0.85` — default; catches common cases (3–10 token changes in 100 tokens)
- `0.70` — aggressive; may link segments that are semantically different; verify correctness for your data

### zstd dictionary training

A trained zstd dictionary captures the statistical structure of your specific token distribution, giving zstd a pre-warmed model. The effect is most pronounced on short-to-medium segments (50–500 tokens) where zstd's context window has less data to learn from.

**Manual training:**

```rust
// Store ≥10 conversations first, then:
let samples_used = vault.train_and_apply_dictionary("cl100k_base", 200).await?;
// "cl100k_base" is the tokenizer key — segments are grouped by tokenizer
// so the dictionary is specific to one vocabulary.
println!("trained on {samples_used} samples");

// All subsequent stores use the dictionary automatically.
// Segments stored before training are still decompressible — zstd frame headers
// record whether a dictionary was used; the decompressor handles both transparently.
```

**Auto-training:**

```rust
let config = StowkenConfig {
    auto_train_dictionary_after: Some(500),  // train after 500 new segments
    dictionary_sample_count: 200,            // sample 200 segments for training
    dictionary_path: Some("/var/lib/stowken/dict.zstd".into()),
    ..StowkenConfig::default()
};
let vault = Stowken::new(backend, config).await?;
// Training runs in a background tokio task — store() returns immediately.
```

**Typical improvement:**

| Corpus                  | Without dict | With dict | Improvement |
|-------------------------|:------------:|:---------:|:-----------:|
| Natural language (GPT)  | 47% savings  | 51%       | +4 pp       |
| Code (Python/JS)        | 86% savings  | 87%       | +1 pp       |
| Domain-specific medical | 52% savings  | 62%       | +10 pp      |

Dictionary training makes the most difference on domain-specific corpora where token frequency distributions differ substantially from general text.

### Encryption at rest

Segment data is encrypted with **AES-256-GCM** before being written to the backend. A pluggable `KeyProvider` trait allows integration with any KMS:

```rust
use stowken_enterprise::encryption::{KeyProvider, RawKeyProvider};

// For testing / local deployments:
let provider = RawKeyProvider::random();  // generates a random 32-byte key

// For production, implement KeyProvider for your KMS:
// struct AwsKmsProvider { client: aws_sdk_kms::Client, key_arn: String }
// #[async_trait] impl KeyProvider for AwsKmsProvider { ... }

// Encrypt / decrypt individual blobs:
use stowken_enterprise::multi_tenant::{encrypt_for_tenant, decrypt_for_tenant};
let ciphertext = encrypt_for_tenant(&compressed_data, "tenant-acme", &provider).await?;
let plaintext  = decrypt_for_tenant(&ciphertext,      "tenant-acme", &provider).await?;
```

Each encryption call uses a **random 96-bit nonce** (prepended to the ciphertext). GCM authentication tags ensure tamper detection. The key reference (not the raw key) is stored alongside the encrypted blob so the correct KMS key can be fetched on decryption.

### Multi-tenant isolation

Three mechanisms prevent one tenant's data from leaking into another's:

**1. Tenant-scoped hashing** — The SHA-256 segment hash includes the tenant ID as HMAC context. Two tenants with byte-identical segment content produce different hashes, preventing hash-probing attacks:

```rust
use stowken_enterprise::multi_tenant::tenant_scoped_hash;

let hash_a = tenant_scoped_hash(&tokens, "tenant-a");
let hash_b = tenant_scoped_hash(&tokens, "tenant-b");
assert_ne!(hash_a, hash_b);  // same tokens, different hashes
```

**2. Per-tenant encryption keys** — Each tenant's segments are encrypted with a distinct KMS key (referenced by the tenant's `encryption_key_ref`). A compromised key for one tenant does not expose other tenants' data.

**3. ACL enforcement** — An in-memory ACL store maps principals to permitted operations per tenant. Embed ACL checks in your API layer:

```rust
use stowken_enterprise::multi_tenant::{TenantRegistry, TenantConfig, AclStore, AclEntry, Permission};

let registry = TenantRegistry::new();
registry.register(TenantConfig {
    tenant_id: "acme".into(),
    encryption_key_ref: Some("arn:aws:kms:us-east-1:123:key/...".into()),
    max_segments: Some(1_000_000),
}).await;

let acl = AclStore::new();
acl.grant(AclEntry {
    principal: "service-account-xyz".into(),
    tenant_id: "acme".into(),
    permissions: vec![Permission::Read, Permission::Write],
}).await;

// In your request handler:
if !acl.is_permitted(&principal, &tenant_id, &Permission::Write).await {
    return Err("forbidden");
}
```

### AWS S3 backend

```toml
[dependencies]
stowken-enterprise = { version = "0.1", features = ["s3"] }
```

```rust
use stowken_enterprise::backends::S3Backend;
use stowken::{Stowken, types::StowkenConfig};

// Credentials from environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
// AWS_DEFAULT_REGION, EC2 instance role, ECS task role, etc.)
let backend = S3Backend::from_env("my-bucket", "stowken/").await?;
let vault = Stowken::new(backend, StowkenConfig::default()).await?;

// Use exactly like any other backend:
let result = vault.store(conversation).await?;
let retrieved = vault.retrieve(&result.id).await?;
vault.delete(&result.id).await?;
let freed = vault.gc().await?;  // deletes zero-ref objects from S3
```

**Layout in S3:**

```
my-bucket/
└── stowken/
    ├── segments/
    │   ├── aabb...cc       # Compressed segment data
    │   └── aabb...cc.meta  # JSON: token_count, ref_count, sizes, created_at
    └── manifests/
        └── conv-001.json   # Conversation manifest
```

GC works by listing all `.meta` objects, reading their `ref_count`, and deleting both the data and meta objects where `ref_count == 0`. Stats (`vault.stats()`) are computed by scanning meta objects — cache results if you call this frequently.

**Custom S3-compatible endpoints** (MinIO, Cloudflare R2, DigitalOcean Spaces):

```rust
use aws_sdk_s3::config::{Builder, Credentials, Region};
use aws_sdk_s3::Client;

let config = Builder::new()
    .endpoint_url("https://my-endpoint.example.com")
    .credentials_provider(Credentials::new("key", "secret", None, None, "static"))
    .region(Region::new("us-east-1"))
    .build();
let client = Client::from_conf(config);
let backend = S3Backend::new(client, "my-bucket", "stowken/");
```

---

## Compression benchmarks

Measured on 200-segment corpora with 120-token system prompts, 40-token user turns, 80-token assistant turns:

| Method                   | bytes/token | vs raw (4.0) |
|--------------------------|:-----------:|:------------:|
| Raw u32                  | 4.000       | —            |
| Varint only              | 2.363       | −41%         |
| Varint + zstd L3         | 2.118       | −47%         |
| Varint + zstd L3 + dict  | 1.980       | −51%         |
| Varint + zstd L9 + dict  | 1.760       | −56%         |

Code corpora (Python/JS) compress much more aggressively (−86%) due to highly repetitive structure. High-entropy data (random tokens) compresses poorly; varint actually makes it larger (−3%) before zstd partially recovers it.

**Dictionary crossover point:** Dictionaries improve compression for segments ≥ ~100 tokens. For very short segments (<50 tokens) the dictionary overhead can hurt. The auto-trainer uses the tokenizer-specific segment population to ensure the dictionary matches the actual data distribution.

---

## Running tests

```sh
# Core library (68 unit tests)
cargo test -p stowken

# Enterprise features (43 tests, including persistence, near-dedup, dict)
cargo test -p stowken-enterprise

# CLI integration tests (10 tests, spawns the binary)
cargo test -p stowken-cli

# Python (23 pytest tests — requires maturin develop first)
cd crates/stowken-python
maturin develop
pytest python/tests/ -v

# Node.js (20 tests, no framework)
cd crates/stowken-node
node test.js
```

---

## Workspace structure

```
stowken/
└── crates/
    ├── stowken/               # Open-source core (Apache 2.0)
    │   ├── src/
    │   │   ├── vault.rs          # Stowken — main public API
    │   │   ├── storage/
    │   │   │   ├── mod.rs        # StorageBackend trait
    │   │   │   ├── memory.rs     # MemoryBackend
    │   │   │   └── filesystem.rs # FilesystemBackend
    │   │   ├── compression/
    │   │   │   ├── pipeline.rs   # CompressionPipeline (varint → zstd)
    │   │   │   ├── varint.rs     # Variable-length integer encoding
    │   │   │   └── dictionary.rs # zstd dictionary training + application
    │   │   ├── dedup/
    │   │   │   └── exact.rs      # SHA-256 content-addressing
    │   │   └── index/
    │   │       └── metadata.rs   # SQLite metadata index
    │   └── examples/
    │       ├── compression_layers.rs   # Benchmarks all compression methods
    │       └── dictionary_training.rs  # Dictionary training end-to-end demo
    │
    ├── stowken-enterprise/    # Enterprise extensions (proprietary)
    │   └── src/
    │       ├── enterprise_vault.rs  # EnterpriseStowken — near-dedup wrapper
    │       ├── near_dedup.rs        # MinHash LSH + delta encoding
    │       ├── encryption.rs        # AES-256-GCM + KeyProvider trait
    │       ├── multi_tenant.rs      # TenantRegistry, AclStore, scoped hashing
    │       └── backends/
    │           └── s3.rs            # AWS S3 StorageBackend
    │
    ├── stowken-python/        # Python bindings (PyO3 + maturin)
    │   └── python/tests/         # pytest test suite
    │
    ├── stowken-node/          # Node.js bindings (napi-rs)
    │   └── test.js               # Plain Node.js test suite
    │
    └── stowken-cli/           # CLI (clap)
        └── tests/integration.rs  # Integration tests (spawns binary)
```

---

## License

- `crates/stowken` — [Apache 2.0](crates/stowken/LICENSE)
- `crates/stowken-enterprise` — Proprietary. Contact [license@stowken.dev](mailto:license@stowken.dev) for pricing and trials.
