# Stowken

[![Crates.io](https://img.shields.io/crates/v/stowken.svg)](https://crates.io/crates/stowken)
[![PyPI](https://img.shields.io/pypi/v/stowken.svg)](https://pypi.org/project/stowken/)
[![CI](https://github.com/stowken/stowken/actions/workflows/ci.yml/badge.svg)](https://github.com/stowken/stowken/actions/workflows/ci.yml)
[![docs.rs](https://docs.rs/stowken/badge.svg)](https://docs.rs/stowken)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Most teams running LLM APIs in production are storing every conversation as raw JSON — which means the same system prompt written to disk thousands of times, no visibility into what's actually flowing through the API, and no way to answer "how many tokens did we spend on model X last month?" without a full table scan.

Stowken is a storage library that fixes this. It stores LLM conversations as content-addressed token segments: identical system prompts are stored once regardless of how many users trigger them, everything is compressed, and a SQLite index sits on top so you can query your traffic without rebuilding it from logs.

---

## Documentation

| Guide | What it covers |
|---|---|
| [Getting Started](docs/getting-started.md) | Install, store, retrieve, first semantic search |
| [Semantic Search](docs/semantic-search.md) | Granularity, typed filters, summary strategies, custom adapters |
| [Analytics](docs/analytics.md) | Clustering, outlier detection, retrieve_batch workflows |
| [Python Cookbook](docs/python-cookbook.md) | DataFrame integration, async usage, fine-tuning export |
| [Architecture](docs/architecture.md) | How dedup, compression, and embedding interact end-to-end |

---

## What you get

**Storage efficiency.** Repeated system prompts collapse to a single copy. RAG context that overlaps across users is stored once. Everything else is varint+zstd compressed. On corpora with shared prompts across users, savings are dominated by dedup (99%+ on fully-shared prompts). On diverse corpora, you still get 54–65% from compression alone — better than uncompressed JSON without any configuration.

**Visibility into your traffic.** The SQLite index lets you answer questions that would otherwise require log archaeology:

```sh
stowken audit-system-prompts          # which prompts are in use, how many times each
stowken cost --by application,model   # token spend broken down any way you want
stowken find-duplicates               # what segments are most heavily shared
stowken segment-stats                 # per-role token distribution across your corpus
```

**Retrieval, not just archival.** Stored conversations can be retrieved and decompressed in microseconds — 15–30µs p99 on a hot vault. This isn't a write-only log; it's a queryable store.

**Semantic search across compressed data** *(feature: `semantic-search`)*. Find conversations by meaning, not keywords. "Find me sessions where the agent struggled with authentication errors" works even though those words may never appear verbatim. Two-level embeddings cover both individual segments (system prompts, user turns, assistant responses, tool calls) and whole-conversation summaries. Plug in any embedding model; OpenAI adapters are built in.

**Behavioral analytics.** `cluster_conversations(k=10)` groups your archive by topic using in-process k-means on conversation embeddings. `find_outliers()` surfaces conversations that don't fit any cluster — usually the most interesting ones. Typed search wrappers let you ask "which system prompts mention payment processing?" against only the system-prompt embedding index, not the full table.

---

## When it pays off most

The more structural repetition your traffic has, the more Stowken helps:

| Your workload | What Stowken does | Typical savings |
|---|---|---:|
| Multi-tenant chatbot (shared system prompt per product) | One copy of each system prompt, regardless of user count | Up to 99% on prompt tokens |
| RAG application (document chunks reused across queries) | Overlapping context segments deduplicated automatically | 20–60% depending on overlap rate |
| RLHF / fine-tuning pipeline (chosen+rejected pairs) | Shared conversation prefixes stored once | ~65% on the full corpus |
| Diverse real-world traffic (creative writing, support, etc.) | Compression only | 54–65% |
| Genuinely unique content, no shared structure | Same as zstd | 50–60% |

If you have a system prompt that goes to 10,000 users per day, Stowken stores it once. If you're doing RLHF with chosen/rejected pairs that share a conversation prefix, those prefixes are stored once. If your traffic is completely unique with no repeated structure, you're getting good compression and the analytics layer — compression alone won't be the headline.

---

## Install

```sh
git clone https://github.com/stowken/stowken
cd stowken

# CLI
cargo install --path crates/stowken-cli

# Python wheel
cd crates/stowken-python
maturin build --release
pip install ../../target/wheels/stowken-*.whl
```

```toml
# Cargo.toml
[dependencies]
stowken = "0.6"
stowken = { version = "0.6", features = ["s3"] }             # S3 backend
stowken = { version = "0.6", features = ["semantic-search"] } # semantic search
```

```sh
pip install stowken
```

```sh
# CLI with semantic search + built-in OpenAI adapters
cargo install --path crates/stowken-cli --features semantic-search
```

---

## Quick start

### Python

```python
import stowken

# In-memory (testing / ephemeral)
vault = stowken.Stowken()

# Persistent local vault
vault = stowken.Stowken.open("/var/lib/stowken")

# S3-backed, multi-writer safe
vault = stowken.Stowken.open_s3(
    bucket="my-vault",
    local_db_path="/var/lib/stowken-s3.db",
    prefix="prod/",
    region="us-east-1",
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

print(result["deduped_segments"])   # how many segments were already stored
print(result["bytes_saved"])        # bytes not written this call

retrieved = vault.retrieve("conv-001")
for seg in retrieved["segments"]:
    print(seg["segment_type"], seg["tokens"][:10])
```

Pass `{"role": "user", "tokens": [9125, 25, ...]}` to skip the built-in tokenization pass if you're already tokenizing upstream.

**Async API.** Every I/O method has an `_async` variant safe inside an asyncio event loop. Sync and async share the same vault and Tokio runtime — mix freely:

```python
async def handle_conversation(conv):
    result = await vault.store_async(conv)
    return result
```

Available pairs: `store`, `retrieve`, `delete`, `stats`, `segment_stats`, `gc`, `gc_candidates`, `reindex`, `find_near_duplicates`, `near_dedup_index_size`, `train_dictionary`, `activate_dictionary`, `list_dictionaries`, `train_substrings`, `list_substrings`, `compact_substrings`, `gc_substrings`.

### CLI

```sh
# Data dir defaults to ~/.local/share/stowken (override: --data-dir or $STOWKEN_DATA_DIR)

# Ingest — single conversation
stowken store conversation.json --id conv-001 --application my-app
echo '{"model":"gpt-4","messages":[...]}' | stowken store -

# Ingest — bulk (JSONL file or directory of JSON/JSONL files)
stowken ingest conversations.jsonl --application my-app
stowken ingest ./export-dir/ --application my-app --max 10000

# Retrieve
stowken retrieve conv-001
stowken retrieve conv-001 --segments       # raw token arrays

# Retrieve multiple conversations at once (pairs naturally with search results)
stowken retrieve-batch conv-001 conv-002 conv-003
echo -e "conv-001\nconv-002" | stowken retrieve-batch --stdin

# Analytics (the useful part)
stowken audit-system-prompts               # prompt inventory by ref count
stowken cost --by application,model        # token attribution
stowken find-duplicates --threshold 10     # heavily-shared segments
stowken find-near-duplicates               # prompt drift clusters
stowken stats
stowken segment-stats

# Maintenance
stowken gc --dry-run
stowken reindex                            # rebuild SQLite from manifests

# Advanced compression (see "Tuning" below)
stowken dict train --samples 200 --activate
stowken substrings train --samples 500
stowken substrings compact
```

S3 CLI integration lands in v0.6.1. For now the CLI uses the filesystem backend; Python has full S3 support.

### Rust

```rust
use stowken::{storage::FilesystemBackend, types::*, Stowken};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = FilesystemBackend::new("/var/lib/stowken").await?;
    let vault = Stowken::open(backend, StowkenConfig::default(), "/var/lib/stowken/metadata.db").await?;

    let result = vault.store(Conversation {
        id: Some("conv-001".into()),
        model: "gpt-4".into(),
        tokenizer: "cl100k_base".into(),
        application: Some("prod".into()),
        messages: vec![
            Message { role: "system".into(), content: MessageContent::Text("You are a helpful assistant.".into()), name: None, tool_call_id: None },
            Message { role: "user".into(),   content: MessageContent::Text("Hello".into()), name: None, tool_call_id: None },
        ],
        metadata: None,
    }).await?;

    println!("deduped {} segments, saved {} bytes", result.deduped_segments, result.bytes_saved);
    Ok(())
}
```

---

## Storage backends

| Backend | Persistence | Multi-writer | Use case |
|---|---|---|---|
| `MemoryBackend` | none | n/a | tests, ephemeral |
| `FilesystemBackend` | local disk | single-process | local dev, single-host |
| `S3Backend` *(feature `s3`)* | S3-compatible | **yes** — ETag-based CAS | cloud, multi-writer ingest |

The S3 backend uses `If-None-Match: *` for first-time segment writes and `If-Match: <etag>` for every ref-count update, retrying on `412 Precondition Failed` up to 8 times with capped exponential backoff. Two writers concurrently deduplicating the same segment produce the correct final ref count — pinned by `s3_concurrent_increments_no_lost_writes`.

Compatible with AWS S3, MinIO, localstack, Cloudflare R2, and DigitalOcean Spaces via `S3Backend::with_options(bucket, prefix, endpoint, region)`.

---

## How it works

**Segmentation.** Each message is classified by role:

| Role | Segment type |
|---|---|
| `system` | `system_prompt` |
| `user` | `user_turn` |
| `assistant` | `assistant_turn` |
| `tool` (with id) | `tool_result` |
| `tool` (no id) | `tool_call` |

Long messages can optionally be split at paragraph boundaries into `Continuation` chunks, each deduplicated independently (see `--split-segments` below).

**Content addressing.** Each segment is SHA-256 hashed over its little-endian token bytes. Same tokens → same hash, always.

**Deduplication.** If the hash exists, increment the ref count and return. The `ConversationManifest` records ordered hashes; it never embeds token data. This is why a system prompt used by 10,000 users takes up storage proportional to 1 system prompt, not 10,000.

**Compression.** Four paths, arranged so the smallest representation wins:

| Frame | Codec | When |
|---|---|---|
| `0x01` | varint only | compression disabled |
| `0x02` | varint + zstd | default |
| `0x03` | varint + zstd + dictionary | after `dict train --activate` |
| `0x04` | delta against a canonical | `--near-dedup` enabled, near-duplicate found |
| `0x05` | substring references | after `substrings train` + `compact` |

Each path is gated by a strict cost-model check — if a smaller frame can't be produced, the previous best is kept. **There is never a net regression in stored bytes.**

**Metadata index.** SQLite tracks conversations and segment metadata; the analytics commands run against it. The near-dedup index (MinHash signatures, LSH band buckets, and embedding aliases) lives separately in append-only logs under `near_dedup/`, decoupled from SQLite so it scales independently. Both are derived caches — `stowken reindex` rebuilds them from on-disk manifests.

```
vault-root/
├── segments/          # content-addressed compressed frames
├── manifests/         # per-conversation ordered segment lists
├── dictionaries/      # zstd dicts, one per training run (v0.3+)
├── substrings/        # packed substring registry (v0.5+)
├── near_dedup/        # MinHash/LSH index + embedding aliases (append-only logs)
└── metadata.db        # SQLite analytics index
```

---

## Semantic search *(feature: `semantic-search`)*

Stowken stores conversations as deduplicated token segments. The semantic search layer sits on top of that structure — it embeds the same unique blobs your dedup logic already identified, so you never embed the same system prompt twice no matter how many conversations reference it.

### How it works

**Two-level embeddings.** Every unique segment gets one embedding (stored once, regardless of how many conversations share it). Every conversation also gets an embedding of its full summary (using a concat-and-truncate strategy by default, or an LLM-generated summary if you opt in). Searches can scan either level or both.

**Pluggable adapters.** Provide any embedding model via the `EmbeddingAdapter` trait. The built-in `OpenAiEmbeddingAdapter` calls `POST /v1/embeddings`; swap it for a local model with zero code changes.

**Embed-on-store.** By default, embeddings are written immediately when you call `store()`. Pass `set_embed_on_store(false)` for bulk ingest and run `embed_all()` afterward.

**Cosine similarity.** Embeddings are L2-normalized at write time, so retrieval is a dot-product scan — no ANN index needed for realistic vault sizes.

### Python quickstart

```python
import stowken

vault = stowken.Stowken.open("/var/lib/stowken")

# One-line setup with built-in OpenAI adapters
vault.use_openai_embeddings(api_key="sk-...", model="text-embedding-3-small")

# Optional: richer conversation-level summaries via LLM
vault.use_openai_summarizer(api_key="sk-...", model="gpt-4o-mini")
# Or keep the default (fast, free): vault.use_concat_summary(max_chars=24_000)

# Retroactively embed an existing vault
stats = vault.embed_all()
print(f"embedded {stats['conversations_embedded']} conversations, "
      f"{stats['segments_embedded']} unique segments")

# Semantic search across both levels (default)
results = vault.semantic_search("production failures", limit=5)
# [{"conversation_id": "...", "score": 0.91, "matched_via": {...}, ...}, ...]

# Read the matching conversations (closes the search → read loop)
convs = vault.retrieve_batch([r["conversation_id"] for r in results])
for conv in convs:
    print(conv["text"][:500])   # full conversation with [role] markers

# Typed search against specific segment roles only
system_hits  = vault.search_prompts("billing system instructions", limit=10)
reply_hits   = vault.search_responses("I cannot help with that", limit=10)
user_hits    = vault.search_user_questions("this isn't working", limit=10)
tool_hits    = vault.search_tool_uses("database query", limit=10)
conv_hits    = vault.search_conversations("debugging session with tool failures", limit=10)
```

**Async equivalents** are available for all methods: `embed_all_async`, `semantic_search_async`, etc.

### Clustering and outlier detection

```python
# Group conversations into k topic clusters
clusters = vault.cluster_conversations(k=12, representative_count=3)
# [{"cluster_id": 0, "size": 87, "representative_ids": [...], "members": [...]}, ...]

# Conversations that don't fit any cluster — often the most revealing
outliers = vault.find_outliers(k=12, threshold=0.4, limit=20)
# [{"conversation_id": "...", "isolation_score": 0.78, "model": "gpt-4", ...}, ...]
```

Clustering uses in-process k-means++ on conversation-level embeddings. No external service required. Deterministic output via a configurable seed.

### CLI commands

```sh
export OPENAI_API_KEY=sk-...

# Bulk ingest without per-conversation embedding (faster for large files)
stowken ingest conversations.jsonl --no-embed

# Embed all stored conversations (run once on an existing vault)
# Embed all stored conversations
# The default (concat) is free — detokenizes and truncates the conversation text.
# Use --summary-strategy llm only for behavioral queries; it costs one GPT call per conversation.
stowken embed-all --openai-key $OPENAI_API_KEY
# With LLM-generated summaries (opt-in, richer for behavioral queries):
stowken embed-all --openai-key $OPENAI_API_KEY --summary-strategy llm --summary-model gpt-4o-mini

# Semantic search
stowken semantic-search "production failures" --openai-key $OPENAI_API_KEY --limit 10
stowken semantic-search "auth errors" --granularity segment --segment-type assistant_turn

# Typed search wrappers
stowken search-prompts      "billing system instructions"
stowken search-responses    "I cannot help with that"
stowken search-user-questions "this isn't working"
stowken search-tool-uses    "database query"
stowken search-conversations "debugging session"

# Close the loop: search → read the matching conversations
stowken semantic-search "payment errors" --limit 5 --json \
  | jq -r '.[].conversation_id' \
  | stowken retrieve-batch --stdin

# Analytics
stowken cluster --k 10 --representatives 3
stowken outliers --k 10 --threshold 0.5 --limit 20
```

All CLI commands accept `--application`, `--model`, and date-range filters. Pass `--openai-key` or set `OPENAI_API_KEY`.

### Rust API

```rust
use stowken::{
    storage::FilesystemBackend,
    types::{SemanticSearchQuery, SearchGranularity},
    Stowken,
};

vault.set_embedding_adapter(my_adapter);        // anything implementing EmbeddingAdapter
vault.use_concat_summary(24_000);               // ConcatTruncate strategy (default)
// or:
vault.set_summarizer(my_llm_summarizer);        // LlmGenerated strategy

let stats = vault.embed_all().await?;

let hits = vault.semantic_search(SemanticSearchQuery {
    text: "production failures".into(),
    granularity: SearchGranularity::Both,
    limit: 10,
    min_score: 0.5,
    ..SemanticSearchQuery::new("production failures")
}).await?;
```

### Summary strategies

| Strategy | Cost | Quality | How to enable |
|---|---|---|---|
| `ConcatTruncate` (default) | Free — no API calls | Good for keyword-rich queries | `vault.use_concat_summary(max_chars)` |
| `LlmGenerated` | One LLM call per conversation | Best for intent / behavioral queries | `vault.set_summarizer(adapter)` or `vault.use_openai_summarizer(...)` |

`ConcatTruncate` detokenizes all segments with role markers (`[system] ... [user] ... [assistant] ...`) and truncates to `max_chars`. It works well for retrieving conversations that mention specific topics. `LlmGenerated` produces a one-paragraph summary capturing user intent, agent behavior, and tool usage — better for behavioral queries like "sessions where the agent gave up and apologized."

### Custom embedding adapters

Implement two traits to plug in any model:

```rust
use stowken::types::{EmbeddingAdapter, SummarizerAdapter};

struct MyEmbedder { /* ... */ }

impl EmbeddingAdapter for MyEmbedder {
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> { /* ... */ }
    fn dimension(&self) -> usize { 1536 }
    fn model_name(&self) -> &str { "my-model-v1" }
}

vault.set_embedding_adapter(MyEmbedder { /* ... */ });
```

```python
# Python: pass a callable
def my_embed(texts: list[str]) -> list[list[float]]:
    return my_model.encode(texts).tolist()

vault.set_embedding_adapter(my_embed, dimension=768, model_name="my-model-v1")

def my_summarize(text: str) -> str:
    return my_llm.summarize(text)

vault.set_summarizer(my_summarize, model_name="my-llm-v1")
```

---

## Tuning

The defaults (varint + zstd + exact dedup) are the right starting point. The options below are for specific workloads, measured on your data before enabling in production.

### `--dict-warmup N` (bench) / `dict train --activate` (production)

Trains a zstd dictionary on a sample of your corpus, then compresses subsequent segments with it. Works best on domain-specific token distributions (medical, code, legal, consistent prompt styles). Not worth enabling for highly diverse traffic.

**Tradeoff:** additional storage savings (2–34pp depending on corpus), but dictionary decompression adds latency per segment. On a large diverse corpus this can be 10× higher p99 retrieval latency. Measure on your retrieval SLA before enabling.

### `--split-segments N`

Splits messages longer than N tokens at paragraph boundaries before storage, enabling finer-grained dedup within long messages. Useful when your system prompts or RAG context blocks have shared sub-sections.

**Tradeoff:** improves dedup rate on diverse corpora (LMSys: 1.5% → 8.8% exact dedup at 256 tokens) but shorter segments compress less well — combined savings can decrease. Useful when retrieval granularity matters more than compression ratio.

### `--near-dedup <threshold>`

Stores near-duplicate segments (Jaccard similarity above threshold) as streaming deltas against a canonical. Useful when your prompts are systematically varied — interpolated user names, slight formatting changes, A/B prompt versions.

**Tradeoff:** 8–18× ingest cost per segment. In our benchmarks across three real-world datasets (HH-RLHF, LMSys, OpenAssistant), near-dedup added zero storage savings over baseline — the data simply wasn't near-duplicate at the token level. It showed +13pp on a synthetic corpus designed to have systematic drift. **Enable only after confirming near-duplicate structure in your actual data** with `stowken find-near-duplicates`.

### Substrings (`substrings train` + `substrings compact`)

Discovers recurring token sequences across your corpus and promotes them to shared blobs. Effective on corpora with genuine sub-segment boilerplate (boilerplate disclaimers, repeated RAG fragments inside otherwise-unique segments).

**Tradeoff:** a post-ingest batch step, not a real-time optimization. No retrieval overhead — substring frames decompress at the same speed as plain zstd. Run `compact` retroactively to rewrite segments stored before training.

---

## Benchmark numbers

`stowken bench` across four corpora (v0.6.1 — full results in [`bench_results/REPORT.md`](bench_results/REPORT.md)):

### Storage

| Corpus | Convs | Incoming | Stored | Saved | Notes |
|---|---:|---:|---:|---:|---|
| Production chatbot (5 shared prompts × 500 convs) | 500 | 395 KB | 2.6 KB | **99.3%** | dedup-dominated |
| HH-RLHF chosen+rejected pairs | 10 K | 5.2 MB | 1.8 MB | **65.5%** | 38.9% from dedup, 43.5% from compression |
| LMSys-Chat-1M sample | 10 K | 18.7 MB | 7.0 MB | **62.5%** | dict-warmup; 60.4% without |
| OpenAssistant unique Q&A | 3.5K | 5.0 MB | 2.2 MB | **56.1%** | dict-warmup; 54.2% without |
| Drift synthetic (near-dup) | 1 K | 1.6 MB | 497 KB | **69.7%** | dict-warmup; 35.9% without |

### Retrieval (hot in-memory vault, decompression only)

| Corpus | p50 | p99 | Tokens/sec |
|---|---:|---:|---:|
| OpenAssistant (baseline) | 16 µs | 33 µs | 20.9M |
| HH-RLHF (baseline) | 20 µs | 80 µs | 4.9M |
| LMSys (baseline) | 15 µs | 122 µs | 18.3M |
| Drift (substrings) | 13 µs | 18 µs | 30.4M |

Dict-warmup trades retrieval speed for density. LMSys p99 rises from 122 µs to 1,601 µs with `--dict-warmup 2000` — measure against your retrieval SLA before enabling. Filesystem and S3 backends add their respective I/O on top.

### Reproducing the benchmarks

```sh
# Crowd-sourced unique content:
pip install datasets
python scripts/fetch_openassistant.py --output oa.jsonl
stowken bench oa.jsonl --retrieve 1000

# Chosen+rejected pairs (no HF gating required):
python scripts/fetch_hh_rlhf.py --output hh_rlhf.jsonl --max 10000
stowken bench hh_rlhf.jsonl --retrieve 1000

# Real-world diverse (LMSys-Chat-1M; requires HF terms acceptance):
huggingface-cli login
python scripts/fetch_lmsys.py --output lmsys.jsonl --max 10000
stowken bench lmsys.jsonl --retrieve 1000

# Synthetic near-duplicate corpus:
python scripts/synth_drift_corpus.py --output drift.jsonl --conversations 1000

# Full 23-run sweep (all corpora × all flags):
bash scripts/bench_sweep.sh     # results → bench_results/
```

---

## Running the tests

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# Semantic search integration tests (no network — deterministic mock embedder):
cargo test -p stowken --features semantic-search

# S3 integration tests (requires MinIO or real S3):
docker run -d --rm -p 9000:9000 \
    -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
    minio/minio server /data
aws --endpoint-url http://localhost:9000 s3 mb s3://stowken-test
STOWKEN_S3_TEST_BUCKET=stowken-test \
    STOWKEN_S3_TEST_ENDPOINT=http://localhost:9000 \
    AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
    AWS_REGION=us-east-1 \
    cargo test -p stowken --features s3 --test integration_tests s3 -- --test-threads 1

# Python (63 pytest tests):
cd crates/stowken-python
PYO3_PYTHON=python3.13 maturin build --release
pip install --force-reinstall ../../target/wheels/stowken-*.whl
pytest python/tests/
```

CI runs Rust build + test + clippy and Python 3.11/3.12 maturin build + pytest on every push (see `.github/workflows/ci.yml`).

---

## Workspace

```
stowken/
├── crates/
│   ├── stowken/            # core library
│   │   └── src/
│   │       ├── semantic.rs     # semantic search methods (feature: semantic-search)
│   │       └── clustering.rs   # k-means++ for cluster_conversations / find_outliers
│   ├── stowken-cli/        # CLI tool
│   │   └── src/
│   │       └── openai_adapter.rs  # built-in OpenAI embedding + summarizer adapters
│   └── stowken-python/     # Python bindings (PyO3 + maturin)
├── scripts/
│   ├── fetch_openassistant.py
│   ├── fetch_hh_rlhf.py
│   ├── fetch_lmsys.py
│   ├── synth_drift_corpus.py
│   └── bench_sweep.sh
├── bench_results/
│   └── REPORT.md
├── ROADMAP.md
└── CHANGELOG.md
```

## License

MIT OR Apache-2.0 — your choice.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). To report a bug: open an issue with your Rust version, OS, and a minimal reproducible example. Security issues: email security@stowken.dev.
