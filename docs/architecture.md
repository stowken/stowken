# Architecture

This document explains how Stowken's storage, deduplication, compression, and semantic search layers fit together — and why the design choices interact the way they do.

---

## The core insight: content-addressed segments

Every LLM conversation is a sequence of messages. Stowken breaks each message into a **segment** (classified by role: system prompt, user turn, assistant turn, tool call, tool result) and assigns each segment a content address: the SHA-256 hash of its token bytes in little-endian order.

```
conversation
├── message: role=system   → segment hash abc123
├── message: role=user     → segment hash def456
└── message: role=assistant → segment hash ghi789
```

The **manifest** for a conversation is just an ordered list of these hashes. It never embeds token data. The manifest is small (a few hundred bytes) regardless of how long the conversation is.

The **segment store** maps each hash to a single compressed blob on disk. If two conversations share the same system prompt, they share the same hash, which points to the same blob. The blob exists exactly once.

This is why deduplication is structural, not opportunistic: it's baked into the addressing scheme. There's no diff or similarity check — identical content produces identical hashes by construction.

---

## Storage layout

```
vault-root/
├── segments/
│   └── ab/
│       └── abc123...  ← compressed token blob
├── manifests/
│   └── conv-001.json  ← ordered list of segment hashes
├── dictionaries/      ← zstd dictionaries (optional)
├── substrings/        ← shared substring registry (optional)
└── metadata.db        ← SQLite analytics index
```

The `metadata.db` is a **derived cache** — it can always be rebuilt from the manifests and segments with `stowken reindex`. This means the source of truth is always the on-disk blobs, and the SQLite index is an acceleration structure for analytics.

---

## Compression pipeline

Each segment blob is written using the smallest of four encodings:

| Frame | Codec | Used when |
|---|---|---|
| `0x01` | varint only | compression disabled |
| `0x02` | varint + zstd | default |
| `0x03` | varint + zstd + dictionary | after `dict train --activate` |
| `0x04` | delta against a canonical | near-duplicate detected (opt-in) |
| `0x05` | substring references | after `substrings train + compact` |

The cost model is strict: a frame is only written if it produces fewer bytes than the previous best. There is never a regression in stored bytes.

**Varint encoding** compresses token IDs (uint32 values) by storing only as many bytes as needed — common small token IDs take 1–2 bytes instead of 4.

**zstd compression** is applied after varint encoding. It works best on tokens that have statistical regularity — domain-specific vocabulary, repeated patterns.

**Dictionary compression** trains a shared zstd dictionary on a sample of your corpus, then uses it for subsequent segments. Effective for domain-specific corpora (medical, legal, code) where vocabulary is consistent. Adds retrieval latency.

**Near-dedup** (delta frames) stores near-duplicate segments as diffs against a canonical. Useful only if your traffic has systematic prompt variations (A/B testing, user name interpolation). Has negligible benefit on diverse real-world corpora — measure before enabling.

**Substring registry** discovers recurring token sub-sequences across your corpus and promotes them to shared blobs. Effective for boilerplate that repeats within otherwise-unique segments. Post-ingest batch step; no retrieval overhead.

---

## SQLite metadata index

The `metadata.db` SQLite file tracks:

```sql
conversations         -- id, model, application, tokenizer, created_at, total_tokens
segments_meta         -- hash, segment_type, token_count, compressed_bytes, ref_count
segment_refs          -- conversation_id, segment_hash, position, segment_type
near_dedup_sigs       -- MinHash signatures for near-dedup (opt-in)
lsh_band_buckets      -- LSH index for near-neighbor lookup (opt-in)
segment_embeddings    -- embedding vectors per segment hash (semantic-search feature)
conversation_embeddings -- embedding vectors per conversation (semantic-search feature)
```

`segments_meta.ref_count` is incremented every time a segment hash is referenced by a new conversation. When `ref_count` drops to zero (after deleting conversations), GC can reclaim the blob.

The `segment_refs` join table is what makes the analytics queries possible: given a set of segment hashes, find all conversations that contain them. This is the same join used by semantic search to go from matching embeddings back to conversation IDs.

---

## Semantic search layer

The semantic search layer sits on top of the content-addressed store without modifying it.

### Dedup-aware embedding

Because segment content is deduplicated by hash, the embedding index is also deduplicated automatically: a system prompt used in 5,000 conversations has one hash, one compressed blob, and one embedding vector. The embedding is computed once.

At query time, the scan finds the embedding, then follows the same `segment_refs` join to surface all 5,000 conversations. This is fundamentally different from document-level embedding systems, where each of the 5,000 copies would be embedded independently.

### Two-level embedding

```
store(conversation)
    │
    ├── per-segment embeddings
    │   └── embed_batch([text_of_segment]) → 1 API call per unique segment
    │       stored in: segment_embeddings (hash → vector)
    │
    └── per-conversation embedding
        └── detokenize all segments → join with [role] markers → truncate
            embed_batch([summary_text]) → 1 API call
            stored in: conversation_embeddings (conv_id → vector)
```

**Segment embeddings** capture the meaning of individual segments in isolation. Effective for typed queries: "which assistant responses contain refusals?" scans only the assistant-turn embedding space.

**Conversation embeddings** capture the full arc of an exchange. Effective for intent queries: "debugging session that ended with a successful fix" requires understanding the whole conversation, not any single segment.

`semantic_search(granularity="both")` scans both tables and merges results, keeping the highest score per conversation across granularities.

### Cosine similarity via dot product

All embedding vectors are L2-normalized at write time:

```
v_stored = v_raw / ||v_raw||
```

Cosine similarity between two unit vectors is just their dot product:

```
cosine(q, v) = q · v    (when both are unit vectors)
```

This means the search scan is a sequence of dot products — no square root, no division. The brute-force scan over all conversation embeddings for a 10,000-conversation vault completes in milliseconds in-process.

### Summary strategies

The conversation-level embedding is computed from a text summary of the conversation:

**ConcatTruncate** (default):
```
[system] You are a helpful assistant.
[user] My payment keeps failing.
[assistant] I can see there's a declined transaction...
```
Truncated to fit the embedding model's context window (24,000 characters by default). No extra API calls. Effective for topic and content retrieval.

**LlmGenerated** (opt-in):
```
"User encountered repeated payment failures. The agent identified the card
issuer was blocking the charge, walked the user through contacting their bank..."
```
One LLM call per conversation. Compresses long, complex conversations into a single paragraph that captures intent and outcome. Better for behavioral queries but slower and costlier to index.

---

## Retrieval path

```
retrieve("conv-001")
    │
    ├── load manifest from backend → [hash1, hash2, hash3, ...]
    │
    └── for each hash:
            load segment blob from backend
            decompress (frame dispatch: varint / zstd / dict / delta / substring)
            → tokens: Vec<u32>

    → RetrievedConversation { manifest, segments }
```

`retrieve_batch` adds a detokenization step:

```
tokens → detokenize → text
    │
    ├── if vault.tokenizer is set: use it
    ├── elif built-in tiktoken adapter exists for tokenizer_name: use it
    └── else: emit decimal token IDs (fallback, semantically weak)
```

The result is a `ConversationText` with `turns` (per-role text) and `text` (full conversation with `[role]` markers joined).

---

## Multi-writer safety (S3 backend)

The S3 backend uses optimistic concurrency control for segment writes:

- **New segment** (first writer): `PUT` with `If-None-Match: *` — succeeds only if the key doesn't exist yet
- **Existing segment** (ref-count increment): `GET` to read current state, `PUT` with `If-Match: <etag>` — succeeds only if no concurrent writer changed it
- **Conflict**: `412 Precondition Failed` → exponential backoff retry (up to 8 attempts)

Two writers deduplicating the same segment simultaneously produce the correct final ref count — neither write is lost. The test `s3_concurrent_increments_no_lost_writes` validates this.

The manifest write is a `PUT` with `If-None-Match: *` keyed on conversation ID, so duplicate ingest of the same conversation ID is safely idempotent.

---

## Rebuild and recovery

If `metadata.db` is lost or corrupted:

```bash
stowken reindex
```

This walks all manifests, reads all segment blobs, and rebuilds the SQLite index from scratch. The semantic embedding tables are also preserved (they're in `metadata.db` but stored alongside the structural index — reindex preserves them if they exist, or they can be regenerated with `embed-all`).

The source of truth is always the on-disk blobs. SQLite is always a cache.
