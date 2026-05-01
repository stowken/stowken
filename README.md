# Stowken

**Compressed, deduplicated, queryable storage for LLM conversations.**

Stowken stores LLM conversations as content-addressed token segments. Identical system prompts and contexts are stored once across the corpus; everything else is varint+zstd compressed. A SQLite index sits on top so you can audit what's actually flowing through your LLM in production.

## How much does it save?

It depends on how much your traffic repeats. To make this concrete, here are two real measurements from `stowken bench` — one on crowd-sourced data with no repeating prompts (the worst case), one on a simulated production chatbot with a shared system prompt (the best case):

| Workload | Conversations | Tokens | Incoming | Stowken | Saved |
|---|---:|---:|---:|---:|---:|
| **OpenAssistant** (English, full — unique content) | 3,482 | 1.30M | 5.0 MB | 2.3 MB | **54.2%** (zstd only) |
| **Production chatbot** (5 shared prompts × 500 convs) | 500 | 101K | 395 KB | 2.6 KB | **99.3%** (dedup-dominated) |
| **Drift workload, baseline** (1000 convs, 1K-token system prompts with interpolated names + RAG) | 1,000 | 3.12M | 11.9 MB | 4.9 MB | 58.8% |
| **Drift workload, `--near-dedup 0.85`** | 1,000 | 3.12M | 11.9 MB | 3.5 MB | **71.0%** (+12 pp) |
| **Drift workload, `substrings train`** | 1,000 | 3.12M | 11.9 MB | 2.5 MB | **78.8%** (+20 pp) |
| **Drift workload, train + `substrings compact`** (retroactive rewrite) | 1,000 | 3.12M | 11.9 MB | 1.9 MB | **83.8%** (+25 pp) |

Real production usage lands somewhere along this spectrum depending on:
- How much your system prompts and RAG contexts are reused **byte-for-byte** (exact dedup wins here)
- How much they drift across calls — interpolated names, slight RAG context variations (`--near-dedup` wins here, but only if your segments are large enough that the 33-byte delta header is small relative to the per-variant savings)
- The token distribution of your unique content (zstd wins ~50% on natural language)
- Whether you can tolerate the ingest cost of near-dedup (50–60× slower than the baseline path on synthetic drift corpora)

`--near-dedup` is off by default for a reason: on workloads with no actual drift (like crowd-sourced data), it adds cost without finding anything. Reproduce both drift bench numbers above with:

```sh
python scripts/synth_drift_corpus.py -o drift.jsonl -n 1000 --system-tokens 1000 --rag-tokens 2000
stowken bench drift.jsonl                    # baseline
stowken --near-dedup 0.85 bench drift.jsonl  # near-dedup on
```

You can run the same benchmark on your own data:

```sh
stowken bench /path/to/your-conversations.jsonl
```

It outputs the same three-stage breakdown (`naive → dedup → zstd`) plus a per-segment-type table so you can see where the savings come from. To reproduce the OpenAssistant number above:

```sh
pip install datasets
python scripts/fetch_openassistant.py --output oasst.jsonl
stowken bench oasst.jsonl
```

---

## What it gives you

- **Storage savings on shared content.** System prompts, RAG contexts, and tool definitions repeat across thousands of conversations. They're stored once.
- **Lossless retrieval.** Segments roundtrip exactly — same `u32` IDs in, same out.
- **Structural analytics.** SQLite-indexed metadata lets you ask cost-attribution and prompt-drift questions in milliseconds. No log scanning.
- **Three data planes:** Rust library (core), CLI, Python package.

## What it doesn't try to be (yet)

- A cloud database. Local filesystem only in v1.
- A drop-in OpenAI proxy. You hand it conversations; it doesn't capture them.
- A near-duplicate detector. Exact dedup only.
- A multi-tenant service. One vault, one process.

---

## Install

**Rust:**
```toml
[dependencies]
stowken = "0.1"
tokio = { version = "1", features = ["full"] }
```

**Python:**
```sh
pip install stowken     # once published; until then: build a wheel locally
```

**CLI:**
```sh
cargo install --path crates/stowken-cli
```

---

## Quick start

### Rust

```rust
use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;

    let conv = Conversation {
        id: Some("conv-001".into()),
        model: "gpt-4".into(),
        tokenizer: "cl100k_base".into(),
        application: Some("my-app".into()),
        messages: vec![
            Message { role: "system".into(),    content: MessageContent::Text("You are a helpful assistant.".into()), name: None, tool_call_id: None },
            Message { role: "user".into(),      content: MessageContent::Text("What is 2 + 2?".into()),               name: None, tool_call_id: None },
            Message { role: "assistant".into(), content: MessageContent::Text("4".into()),                            name: None, tool_call_id: None },
        ],
        metadata: None,
    };

    let result = vault.store(conv).await?;
    println!("stored {}: {} new, {} deduped", result.id, result.new_segments, result.deduped_segments);

    let retrieved = vault.retrieve("conv-001").await?;
    for seg in &retrieved.segments {
        println!("  {:?}: {} tokens", seg.segment_type, seg.token_count);
    }

    Ok(())
}
```

For persistence, swap `MemoryBackend::new()` for `FilesystemBackend::new("/var/lib/stowken").await?` and use `Stowken::open(backend, config, "/var/lib/stowken/metadata.db").await?`.

### Python

```python
import stowken

# Ephemeral
vault = stowken.Stowken()

# Persistent (creates the directory + metadata.db if needed)
vault = stowken.Stowken.open("/var/lib/stowken")

# S3-backed (multi-writer safe; v0.6+)
vault = stowken.Stowken.open_s3(
    bucket="my-vault",
    local_db_path="/var/lib/stowken-s3.db",   # SQLite metadata index, local
    prefix="prod/",                            # optional key prefix
    region="us-east-1",                        # optional; AWS_REGION env also works
    # endpoint="http://localhost:9000",        # optional MinIO/R2/etc. endpoint
)

result = vault.store({
    "id": "conv-001",
    "model": "gpt-4",
    "tokenizer": "cl100k_base",
    "messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ],
})
print(f"new={result.new_segments} deduped={result.deduped_segments}")

# retrieve() returns the actual u32 token IDs in seg["tokens"]
retrieved = vault.retrieve("conv-001")
for seg in retrieved["segments"]:
    print(seg["segment_type"], seg["tokens"][:10])
```

Pre-tokenized input is supported — pass `{"role": "user", "tokens": [...]}` to skip the built-in tiktoken pass.

**Async API.** Every I/O method has an `_async` variant that returns an awaitable, safe to use inside an asyncio event loop (FastAPI handlers, async ingestion pipelines):

```python
import asyncio
import stowken

async def main():
    vault = stowken.Stowken.open("/var/lib/stowken")
    result = await vault.store_async({"model": "gpt-4", "messages": [...]})
    conv = await vault.retrieve_async(result.id)
    stats = await vault.stats_async()

asyncio.run(main())
```

Sync and async share the same vault instance and the same Tokio runtime under the hood — you can mix them freely. Available pairs: `store/store_async`, `retrieve/retrieve_async`, `delete/delete_async`, `stats/stats_async`, `segment_stats/segment_stats_async`, `gc/gc_async`, `gc_candidates/gc_candidates_async`, `reindex/reindex_async`.

### CLI

```sh
# Default data dir: ~/.local/share/stowken (override with --data-dir or $STOWKEN_DATA_DIR)

# Store from stdin or file
echo '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}' \
  | stowken store - --id conv-001 --application my-app

# Retrieve
stowken retrieve conv-001
stowken retrieve conv-001 --segments       # raw token arrays

# Aggregate stats
stowken stats
stowken segment-stats

# GC
stowken gc --dry-run                        # lists candidates, doesn't delete
stowken gc

# Export training pairs
stowken export --format jsonl --output training.jsonl

# Benchmark on your own data (read-only — does not touch your vault)
stowken bench /path/to/conversations.jsonl
stowken --json bench ./conv-dir/ --max 1000

# Rebuild the SQLite index from on-disk manifests (after backup restore, etc.)
stowken reindex

# Dictionary management (zstd with a trained dictionary)
stowken dict train --samples 200 --activate     # train + use for new frames
stowken dict list                                # show all registered dicts
stowken dict activate 1234567890                 # rotate to a specific dict

# Near-duplicate detection (off by default)
stowken --near-dedup 0.85 store conv.json        # ingest with near-dedup ON
stowken find-near-duplicates --threshold 0.85    # surface drift clusters

# Token-level shared-substring dedup (v0.5; off until trained)
stowken substrings train --samples 500           # discover + register substrings
stowken substrings list                          # inspect the registry
stowken substrings compact                       # retroactively rewrite pre-train segments
stowken substrings gc                            # drop unreferenced substrings
```

---

## The interesting commands

These are the commands that justify having a tool instead of just zstd-ing your blob store:

### `stowken audit-system-prompts`

Lists every distinct system prompt by reference count. Prompt drift — interpolated names, formatting tweaks, copy-paste mutations — shows up immediately as separate hashes that should be one.

```
+------+--------+-------------------+
| Refs | Tokens | Hash              |
+------+--------+-------------------+
| 8    | 6      | 487e8068a1851be3… |
+------+--------+-------------------+
1 unique system prompt(s).
```

### `stowken cost --by application,model`

Token attribution across whatever dimensions you want — `application`, `model`, `segment_type`, `day`, comma-combined.

```
application	model	    total_tokens
----------------------------------------
agent	    claude-opus	42
chatbot	    gpt-4	    70
```

### `stowken find-duplicates --threshold 10`

The high-traffic segments. If a 4,000-token RAG block is referenced 50,000 times, you're either getting massive value from caching or massive value from refactoring it into something explicit.

```
+------+---------------+--------+--------+-------------------+
| Refs | Type          | Tokens | Saved  | Hash              |
+------+---------------+--------+--------+-------------------+
| 50000| context       | 4000   | 781 MB | a3f29c1...        |
+------+---------------+--------+--------+-------------------+
```

---

## How it works

**1. Segmentation.** Each message is classified by role:

| Role               | Segment type      |
|--------------------|-------------------|
| `system`           | `system_prompt`   |
| `user`             | `user_turn`       |
| `assistant`        | `assistant_turn`  |
| `tool` (with id)   | `tool_result`     |
| `tool` (no id)     | `tool_call`       |

Optionally, a configurable token-sequence delimiter inside a `user_turn` splits it into a `Context` + `UserTurn` pair so RAG context dedups separately from the actual question.

**2. Content addressing.** Each segment is hashed with SHA-256 over the little-endian token bytes. Two segments with identical tokens always produce the same hash.

**3. Deduplication.** If the hash already exists, increment the segment's reference count. If not, compress and store. The `ConversationManifest` records the ordered list of hashes plus metadata; it never embeds the segment data.

**4. Compression.** Every stored segment starts with a 1-byte format version, followed by varint-encoded token IDs, followed by zstd. Varint alone saves ~40% over 4-byte-per-token (lower IDs are more frequent in BPE vocabs); zstd takes another 5–15 percentage points on top.

**5. Metadata index.** A SQLite database (`metadata.db`) tracks conversation manifests and segment metadata. All the analytics commands run against this index. The index can be rebuilt from manifests; it's never the source of truth for segment data.

```
vault-root/
├── segments/
│   └── aa/bb/aabb...cc       # versioned compressed bytes
│       aabb...cc.meta        # ref-count + sizes JSON sidecar
├── manifests/
│   └── conv-001.json         # ordered list of segment hashes
└── metadata.db               # SQLite index
```

### Frame format

The first byte of every compressed segment identifies the codec:

| Byte | Codec                          | Body                                                |
|------|--------------------------------|-----------------------------------------------------|
| 0x01 | varint only                    | varint(tokens)                                      |
| 0x02 | varint + zstd                  | zstd(varint(tokens))                                |
| 0x03 | varint + zstd with dict        | dict_id(u32 LE) ‖ zstd(varint(tokens), dict)        |
| 0x04 | delta against canonical        | canonical_hash(32 bytes) ‖ streaming delta ops      |
| 0x05 | substring-referenced segment   | sequence of `Inline(tokens)` and `Ref(substring_id)` ops |

For 0x03 frames, the 4-byte little-endian zstd dictionary ID directly follows the version byte. The decompressor looks the ID up in the per-vault dictionary registry; multiple dictionaries coexist so rotation is non-destructive — old frames stay readable as long as their dict file is in the registry.

For 0x04 frames, the variant is stored as a streaming delta against another segment (the *canonical*). Decompression fetches the canonical from the backend and applies the delta. Used by near-duplicate detection — variants can be 80–95% smaller than the full token stream.

For 0x05 frames, the segment is encoded as a sequence of inline runs and references into a per-vault substring registry. Long shared boilerplate (e.g. "as a large language model, I cannot...") is stored once as a substring blob; every segment that contains it gets a small reference instead of the full tokens. Used by token-level shared-substring dedup; train substrings with `stowken substrings train` and the next ingest pass uses them automatically when they win.

The manifest carries a `schema_version` field with the same forward-compatibility guarantee. Future codecs get new bytes; existing data stays readable indefinitely.

### Dictionary registry

Filesystem-backed vaults persist trained dictionaries under `<vault-root>/dictionaries/`:

```
dictionaries/
├── 1234567890.zstd     # raw bytes (filename matches dict_id)
├── 1234567890.json     # info: created_at, sample_count, size
└── active              # one line: "1234567890"
```

In-memory vaults (`stowken.Stowken()` with no path) hold the registry in RAM only.

---

## Configuration

`StowkenConfig` is small on purpose:

```rust
let config = StowkenConfig {
    enable_compression: true,   // varint+zstd; set false for varint only
};
```

Tokenizer support is opt-in via `set_tokenizer()` or a built-in tiktoken adapter (when the `tiktoken` feature is enabled, which it is by default). Pre-tokenized input via `MessageContent::Tokens(Vec<u32>)` always works with no tokenizer configured.

---

## Workspace

```
stowken/
└── crates/
    ├── stowken/         # core library (Apache 2.0)
    ├── stowken-cli/     # command-line tool
    └── stowken-python/  # Python bindings (PyO3 + maturin)
```

## Running tests

```sh
cargo test --workspace            # 94 Rust tests across all crates

cd crates/stowken-python
PYO3_PYTHON=python3.13 maturin build --release
pip install --force-reinstall ../../target/wheels/stowken-*.whl
pytest python/tests/              # 28 Python tests
```

## License

Apache 2.0. See `LICENSE`.

## Storage backends

| Backend | Persistence | Multi-writer safe | Where |
|---|---|---|---|
| `MemoryBackend` | none | n/a | testing, ephemeral caches |
| `FilesystemBackend` | local disk | no (single-process) | local dev, single-host servers |
| `S3Backend` *(v0.6, behind `s3` Cargo feature)* | S3 | **yes** — ETag-based optimistic ref-count CAS | cloud workloads, multi-writer ingest |

The S3 backend uses `If-None-Match: *` for first-time `.meta` writes and `If-Match: <etag>` for ref-count updates, with capped exponential backoff on `412 Precondition Failed`. Two writers concurrently incrementing the same segment's ref-count produce the correct final value (see the `s3_concurrent_increments_no_lost_writes` test). Compatible with any S3-compatible store: AWS S3, MinIO, localstack, Cloudflare R2, DigitalOcean Spaces.

## Status

Early. Frame format and manifest schema are versioned so on-disk data should survive future changes, but the API is not yet stable. Scale: tested against synthetic corpora up to ~500 conversations / 100K tokens. Not yet validated on production-scale data (millions of conversations, >10 GB on disk).

See [ROADMAP.md](ROADMAP.md) for the planned v0.2 → v0.6 sequence — including the three deferred features (versioned dictionaries, near-dedup, token-level shared-substring dedup) that v0.1 deliberately leaves out.

If you're using it on real workloads, pin the version, version-control your data, and tell me what broke.
