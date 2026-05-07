# Getting Started

Stowken stores LLM conversations as compressed, deduplicated token segments and indexes them in SQLite so you can query and search your traffic without rebuilding it from logs.

This guide gets you from zero to a working vault with semantic search in about ten minutes.

---

## Install

**CLI**

```bash
git clone https://github.com/stowken/stowken
cd stowken
cargo install --path crates/stowken-cli --features semantic-search
```

**Python**

```bash
cd crates/stowken-python
maturin build --release --features semantic-search
pip install ../../target/wheels/stowken-*.whl
```

**Rust library**

```toml
[dependencies]
stowken = { version = "0.6", features = ["semantic-search"] }
```

---

## Store your first conversations

Every conversation is a JSON object with a `model`, `tokenizer`, and `messages` array.

```json
{
  "id": "conv-001",
  "model": "gpt-4",
  "tokenizer": "cl100k_base",
  "application": "my-app",
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "How do I reverse a list in Python?"},
    {"role": "assistant", "content": "Use `my_list[::-1]` or `list(reversed(my_list))`."}
  ]
}
```

**Single conversation:**

```bash
stowken store conversation.json
```

**Bulk from a JSONL file** (one conversation per line):

```bash
stowken ingest conversations.jsonl --application my-app
```

**Bulk from a directory** of `.json` / `.jsonl` files:

```bash
stowken ingest ./export-dir/ --application my-app
```

---

## Check what's stored

```bash
stowken stats
```

```
+---------------+---------+
| metric        | value   |
+---------------+---------+
| Conversations | 10000   |
| Total tokens  | 1350353 |
| Unique segs   | 26740   |
| Total segs    | 48815   |
| Dedup ratio   | 45.2%   |
| Savings       | 65.5%   |
+---------------+---------+
```

`Unique segs` is how many compressed blobs are on disk. `Total segs` is the total references across all conversations. The gap is deduplication: segments shared across conversations are stored once.

```bash
stowken segment-stats      # breakdown by role
stowken audit-system-prompts   # which system prompts exist and how often
stowken cost --by model,application   # token spend attribution
```

---

## Retrieve a conversation

```bash
stowken retrieve conv-001
stowken retrieve conv-001 --segments   # raw token arrays
```

**Read multiple conversations at once** (pairs naturally with search results):

```bash
stowken retrieve-batch conv-001 conv-002 conv-003
```

Or pipe IDs from stdin:

```bash
echo -e "conv-001\nconv-002" | stowken retrieve-batch --stdin
```

---

## Enable semantic search

Semantic search requires an embedding model. Stowken ships a built-in adapter for OpenAI's embedding API.

```bash
export OPENAI_API_KEY=sk-...

# Embed all stored conversations (run once; future stores embed automatically)
stowken embed-all --openai-key $OPENAI_API_KEY
```

This embeds:
1. Every unique segment (once per hash — deduplication applies to embedding too)
2. A text summary of every conversation (detokenized, role-labeled, no extra API calls)

Then search:

```bash
stowken semantic-search "debugging session with multiple errors"
stowken semantic-search "user frustrated with AI responses" --limit 5

# Read the actual conversations that matched
stowken semantic-search "production failures" --limit 3 --json \
  | jq -r '.[].conversation_id' \
  | stowken retrieve-batch --stdin
```

---

## Python quickstart

```python
import stowken

vault = stowken.Stowken.open("/var/lib/stowken")

# Store a conversation
vault.store({
    "model": "gpt-4",
    "tokenizer": "cl100k_base",
    "messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]
})

# Enable semantic search
vault.use_openai_embeddings(api_key="sk-...", model="text-embedding-3-small")
vault.embed_all()

# Search
hits = vault.semantic_search("math help", limit=5)
for h in hits:
    print(h["conversation_id"], h["score"])

# Read the matching conversations
ids = [h["conversation_id"] for h in hits]
convs = vault.retrieve_batch(ids)
for conv in convs:
    print(conv["text"])
```

---

## What's next

- [Semantic Search](semantic-search.md) — granularity, typed filters, score thresholds
- [Analytics](analytics.md) — clustering, outlier detection, retrieve_batch workflows
- [Python Cookbook](python-cookbook.md) — DataFrame integration, Jupyter examples
- [Architecture](architecture.md) — how dedup, compression, and embedding interact
