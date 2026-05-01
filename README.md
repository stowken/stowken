# Stowken

**Compressed, deduplicated, queryable storage for LLM conversations.**

Stowken stores LLM conversations as content-addressed token segments. Identical system prompts are stored once across the corpus; near-duplicates are stored as deltas; long recurring boilerplate is promoted to shared substring blobs; everything else is varint+zstd compressed. A SQLite index sits on top so you can audit what's actually flowing through your LLM in production.

Runs locally on a filesystem or against S3 (and any S3-compatible store: MinIO, R2, Spaces). The S3 backend is multi-writer safe via ETag-based optimistic concurrency on ref-count writes.

> Status: pre-1.0. The on-disk format is forward-compatible (every frame carries a 1-byte version tag, every manifest carries a `schema_version`), but the public API may still break on minor version bumps until 1.0.

---

## Headline numbers

### Storage

Same `stowken bench` command across four corpora, four configurations:

| Workload | Tokens | Incoming | Stowken | Saved |
|---|---:|---:|---:|---:|
| **OpenAssistant** (3,482 convs, unique crowd-sourced content) | 1.30M | 5.0 MB | 2.3 MB | **54%** (zstd only) |
| **Production chatbot** (5 shared prompts × 500 convs) | 101K | 395 KB | 2.6 KB | **99.3%** (dedup-dominated) |
| **Drift workload, baseline** (1000 convs, 1K-token system prompts with interpolated names + RAG) | 3.12M | 11.9 MB | 4.9 MB | 58.8% |
| **Drift workload, `--near-dedup 0.85`** | 3.12M | 11.9 MB | 3.5 MB | **71.0%** *(+12 pp)* |
| **Drift workload, `substrings train`** | 3.12M | 11.9 MB | 2.5 MB | **78.8%** *(+20 pp)* |
| **Drift workload, `substrings train` + `compact`** *(retroactive rewrite)* | 3.12M | 11.9 MB | **1.9 MB** | **83.8%** *(+25 pp)* |

Real production usage lands somewhere along this spectrum. The shape that matters is **how much of your traffic is shared between calls**:

| Trait of your data | Best feature | Typical add'l savings |
|---|---|---:|
| Identical system prompts byte-for-byte | exact dedup *(always on)* | up to 99% |
| System prompts with interpolated values (`Hi, {name}`) | `--near-dedup` | +5–20pp |
| Long shared boilerplate inside otherwise-distinct messages | `substrings train` + `compact` | +5–25pp |
| Genuinely unique content (creative writing, crowdsourced) | varint + zstd *(always on)* | ~50% |

### Retrieval

Decompression-only on a hot in-memory vault, OpenAssistant 3,482 convs, 1.30M tokens:

| Metric | Value |
|---|---:|
| Throughput | **20.9M tokens/sec** |
| Conversations/sec | **56,684** |
| Latency p50 | **16 µs** |
| Latency p95 | 26 µs |
| Latency p99 | 31 µs |

Near-dedup adds canonical-resolution overhead — ~30% slower on a 0x04-heavy corpus (drift, p50 29 µs → 42 µs). Substring frames decompress at the same speed as plain zstd. Filesystem and S3 add their respective I/O on top of these numbers.

---

## Install

The crate hasn't been published yet (see ROADMAP and the licensing/availability discussion). Until then, build from source:

```sh
git clone https://github.com/stowken/stowken
cd stowken

# Rust library + CLI
cargo install --path crates/stowken-cli

# Python wheel
cd crates/stowken-python
maturin build --release
pip install ../../target/wheels/stowken-*.whl
```

When the first version is published you'll be able to:

```toml
# Cargo.toml
[dependencies]
stowken = "0.6"

# enable the S3 backend
stowken = { version = "0.6", features = ["s3"] }
```

```sh
pip install stowken
```

---

## Quick start

### Python

```python
import stowken

# In-memory (testing)
vault = stowken.Stowken()

# Persistent on local disk
vault = stowken.Stowken.open("/var/lib/stowken")

# S3-backed (multi-writer safe)
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

retrieved = vault.retrieve("conv-001")
for seg in retrieved["segments"]:
    print(seg["segment_type"], seg["tokens"][:10])
```

Pre-tokenized input is supported — pass `{"role": "user", "tokens": [9125, 25, ...]}` to skip the built-in tiktoken pass.

**Async API.** Every I/O method has an `_async` variant safe inside an asyncio event loop:

```python
async def main():
    vault = stowken.Stowken.open("/var/lib/stowken")
    await vault.store_async({"model": "gpt-4", "messages": [...]})
    conv = await vault.retrieve_async("conv-001")

asyncio.run(main())
```

Sync and async share the same vault and the same Tokio runtime — they can be mixed freely. Available pairs: `store`, `retrieve`, `delete`, `stats`, `segment_stats`, `gc`, `gc_candidates`, `reindex`, `find_near_duplicates`, `near_dedup_index_size`, `train_dictionary`, `activate_dictionary`, `list_dictionaries`, `train_substrings`, `list_substrings`, `compact_substrings`, `gc_substrings`.

### CLI

```sh
# Default data dir: ~/.local/share/stowken (override with --data-dir or $STOWKEN_DATA_DIR)

# Ingest from stdin or file
stowken store conversation.json --id conv-001 --application my-app
echo '{"model":"gpt-4","messages":[...]}' | stowken store -

# Retrieve
stowken retrieve conv-001
stowken retrieve conv-001 --segments       # raw token arrays

# Aggregate analytics
stowken stats
stowken segment-stats
stowken audit-system-prompts                # find prompt drift
stowken cost --by application,model         # token attribution
stowken find-duplicates --threshold 10      # high-traffic refactor candidates
stowken find-near-duplicates                # similarity clusters

# Maintenance
stowken gc --dry-run
stowken reindex                             # rebuild SQLite from manifests

# Compression knobs (each off-by-default)
stowken --near-dedup 0.85 store conv.json   # ingest with near-dedup ON
stowken dict train --samples 200 --activate # zstd dictionary
stowken substrings train --samples 500      # substring promotion
stowken substrings compact                  # retroactively rewrite old segments
stowken substrings gc                       # drop unreferenced substrings

# Benchmarking
stowken bench /path/to/conversations.jsonl
stowken bench /path/to/conversations.jsonl --retrieve 1000  # +retrieval pass

# Export training pairs
stowken export --format jsonl --output training.jsonl
```

S3 backing for the CLI lands in v0.6.1; for now the CLI uses the filesystem backend.

### Rust

```rust
use stowken::{
    storage::MemoryBackend,                      // or FilesystemBackend, S3Backend
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
            Message { role: "system".into(),    content: MessageContent::Text("You are a helpful assistant.".into()),
                      name: None, tool_call_id: None },
            Message { role: "user".into(),      content: MessageContent::Text("What is 2 + 2?".into()),
                      name: None, tool_call_id: None },
            Message { role: "assistant".into(), content: MessageContent::Text("4".into()),
                      name: None, tool_call_id: None },
        ],
        metadata: None,
    };

    let result = vault.store(conv).await?;
    let retrieved = vault.retrieve(&result.id).await?;
    Ok(())
}
```

For a persistent vault, swap `MemoryBackend::new()` for `FilesystemBackend::new("/var/lib/stowken").await?` and use `Stowken::open(backend, config, "/var/lib/stowken/metadata.db").await?`. For S3, enable the `s3` feature and use `S3Backend::with_options(bucket, prefix, endpoint, region).await?`.

---

## Storage backends

| Backend | Persistence | Multi-writer safe | Use case |
|---|---|---|---|
| `MemoryBackend` | none | n/a | testing, ephemeral caches |
| `FilesystemBackend` | local disk | no (single-process per data dir) | local dev, single-host servers |
| `S3Backend` *(behind `s3` Cargo feature)* | S3 / MinIO / R2 / Spaces | **yes** — ETag-based optimistic ref-count CAS | cloud, multi-writer ingest |

The S3 backend uses `If-None-Match: *` on first-time meta writes and `If-Match: <etag>` on every ref-count update; on `412 Precondition Failed` the loop re-reads and retries up to 8 times with capped exponential backoff. Two writers concurrently incrementing the same segment's ref-count produce the correct final value — pinned by `s3_concurrent_increments_no_lost_writes`.

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

A configurable token-sequence delimiter inside a `user_turn` optionally splits it into a `Context` + `UserTurn` pair so RAG context dedups separately from the actual question.

**2. Content addressing.** Each segment is hashed with SHA-256 over the little-endian token bytes. Two segments with identical tokens always produce the same hash.

**3. Deduplication.** If the hash already exists, increment the segment's reference count. If not, compress and store. The `ConversationManifest` records the ordered list of hashes plus metadata; it never embeds segment data.

**4. Compression.** Multiple paths, all arranged so the cheapest representation wins:

- **Always on:** varint + zstd. Varint alone saves ~40% over 4-byte-per-token; zstd takes another 5–15pp.
- **Opt-in via `dict train`:** trained zstd dictionary improves compression on domain-specific token distributions.
- **Opt-in via `--near-dedup`:** segments that are *almost* identical are stored as a streaming delta against the canonical.
- **Opt-in via `substrings train`:** long recurring substrings are promoted to shared blobs; segments reference them by ID.

Each path is gated by a strict cost-model check: if a smaller frame can't be produced, the previous best representation is kept. There is never a net-regression in stored bytes.

**5. Metadata index.** A SQLite database (`metadata.db`) tracks conversations, segment metadata, near-dedup signatures, and LSH band buckets. All the analytics commands run against this index. The index is a derived cache — `stowken reindex` rebuilds it from on-disk manifests.

```
vault-root/
├── segments/
│   └── aa/bb/aabb...cc            # versioned compressed bytes
│       aabb...cc.meta             # ref-count + sizes JSON sidecar
├── manifests/
│   └── conv-001.json              # ordered list of segment hashes
├── dictionaries/                  # zstd dicts, named by dict_id (v0.3+)
├── substrings/                    # packed substring registry (v0.5+)
│   ├── blobs.dat
│   └── index.json
└── metadata.db                    # SQLite index
```

### Frame format

Every compressed segment starts with a 1-byte version tag. Decompression dispatches per-frame so old data stays readable forever.

| Byte | Codec | Body |
|------|-------|------|
| `0x01` | varint only | `varint(tokens)` |
| `0x02` | varint + zstd | `zstd(varint(tokens))` |
| `0x03` | varint + zstd with dictionary | `dict_id(u32 LE) || zstd(varint(tokens), dict)` |
| `0x04` | delta against a canonical segment | `canonical_hash(32 bytes) || streaming delta ops` |
| `0x05` | substring-referenced segment | sequence of `Inline(tokens)` and `Ref(substring_id)` ops |

The manifest carries a `schema_version` field with the same forward-compatibility guarantee. Future codecs get new bytes; existing data stays readable indefinitely.

---

## Configuration

`StowkenConfig` is small on purpose:

```rust
let config = StowkenConfig {
    enable_compression: true,         // varint+zstd; set false for varint only
    near_dedup_threshold: None,       // Some(0.85) to enable near-dedup
};
```

Tokenizer support is opt-in via `set_tokenizer()` or a built-in tiktoken adapter (when the `tiktoken` feature is enabled, which it is by default). Pre-tokenized input via `MessageContent::Tokens(Vec<u32>)` always works with no tokenizer configured.

The dictionary registry, substring registry, and near-dedup LSH index are managed via dedicated APIs (`train_dictionary`, `train_substrings`, etc.), not config — they're stateful and persistent.

---

## Reproducing the benchmarks

The numbers in this README are reproducible from scripts in `scripts/`:

```sh
# Crowd-sourced unique content (zstd-only territory):
pip install datasets
python scripts/fetch_openassistant.py --output oasst.jsonl
stowken bench oasst.jsonl --retrieve 1000

# Synthetic drift-shaped corpus (near-dedup + substring territory):
python scripts/synth_drift_corpus.py \
    --output drift.jsonl --conversations 1000 \
    --system-tokens 1000 --rag-tokens 2000
stowken bench drift.jsonl --retrieve 500
stowken --near-dedup 0.85 bench drift.jsonl --retrieve 500

# Real production-mix (LMSys-Chat-1M; gated, requires HF terms acceptance):
huggingface-cli login
python scripts/fetch_lmsys.py --output lmsys.jsonl --max 10000
stowken bench lmsys.jsonl --retrieve 1000
```

Each `bench` run reports the three-stage savings breakdown (naive → dedup-only → zstd), per-segment-type detail, and (if `--retrieve N` is set) decompression throughput + p50/p95/p99 latency.

---

## Workspace

```
stowken/
├── crates/
│   ├── stowken/         # core library (Apache 2.0)
│   ├── stowken-cli/     # command-line tool
│   └── stowken-python/  # Python bindings (PyO3 + maturin)
├── scripts/
│   ├── fetch_openassistant.py
│   ├── fetch_lmsys.py
│   └── synth_drift_corpus.py
├── ROADMAP.md
├── CHANGELOG.md
└── CONTRIBUTING.md
```

## Running the test suite

```sh
# Rust workspace (172 tests across all crates)
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# S3 integration tests (8 extra tests; need MinIO or real S3)
docker run -d --rm -p 9000:9000 \
    -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
    minio/minio server /data
aws --endpoint-url http://localhost:9000 s3 mb s3://stowken-test
STOWKEN_S3_TEST_BUCKET=stowken-test \
    STOWKEN_S3_TEST_ENDPOINT=http://localhost:9000 \
    AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
    AWS_REGION=us-east-1 \
    cargo test -p stowken --features s3 --test integration_tests s3 \
        -- --test-threads 1

# Python (63 pytest tests)
cd crates/stowken-python
PYO3_PYTHON=python3.13 maturin build --release
pip install --force-reinstall ../../target/wheels/stowken-*.whl
pytest python/tests/
```

CI runs the first two of these on every push (see `.github/workflows/ci.yml`).

---

## Roadmap & changelog

- **[ROADMAP.md](ROADMAP.md)** — milestone-by-milestone plan and status. v0.1 → v0.6 are all shipped at the core level.
- **[CHANGELOG.md](CHANGELOG.md)** — what landed in each version.

## License

Apache 2.0. See `LICENSE`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issue templates and a security policy land before the first tagged release.
