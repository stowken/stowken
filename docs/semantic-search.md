# Semantic Search

Stowken's semantic search finds conversations by meaning, not keywords. "Find sessions where the agent struggled with authentication" works even if those words never appear verbatim in the conversation.

---

## How it works

Two embedding tables are maintained:

| Table | What's stored | When |
|---|---|---|
| `segment_embeddings` | One embedding per unique segment hash | At store time (embed-on-store) or via `embed-all` |
| `conversation_embeddings` | One embedding per conversation summary | At store time or via `embed-all` |

Because Stowken already deduplicates segments by content hash, the embedding index is also deduplicated: a system prompt shared across 5,000 conversations is embedded exactly once. At query time, the join from segment → conversation surfaces all 5,000 conversations from that single embedding.

Embeddings are L2-normalized at write time so cosine similarity reduces to a dot product at search time.

---

## Setup

```bash
export OPENAI_API_KEY=sk-...
stowken embed-all --openai-key $OPENAI_API_KEY
```

The default summary strategy (`concat`) detokenizes all segments with role markers and truncates to fit the embedding context window. No additional API calls beyond the embedding itself.

Future `store` calls embed automatically (embed-on-store is on by default). For bulk ingest without per-conversation overhead:

```bash
stowken ingest big.jsonl --no-embed
stowken embed-all --openai-key $OPENAI_API_KEY
```

---

## Basic search

```bash
stowken semantic-search "user frustrated with AI responses" --openai-key $OPENAI_API_KEY
```

```
conv-8821   score=0.847   via=conversation
conv-3104   score=0.831   via=conversation
conv-0092   score=0.819   via=segment(user_turn)
```

Default: searches both segment and conversation embeddings, returns top 10.

---

## Search granularity

| Granularity | Searches | Best for |
|---|---|---|
| `both` (default) | Segment + conversation embeddings | General queries |
| `conversation` | Conversation summaries only | Intent / outcome queries |
| `segment` | Individual segment embeddings only | Specific phrasing, role-specific content |

```bash
# Find conversations where the overall arc matches
stowken semantic-search "debugging session ending in success" --granularity conversation

# Find conversations containing a specific kind of user message
stowken semantic-search "user gives up and asks to start over" --granularity segment
```

---

## Typed segment search

Search within a specific message role only:

```bash
stowken search-prompts "billing and payment instructions"
stowken search-responses "I cannot help with that request"
stowken search-user-questions "how do I cancel my subscription"
stowken search-tool-uses "database query execution"
stowken search-conversations "multi-step agentic task completion"
```

These are thin wrappers around `semantic-search` that set `--granularity segment` and filter to the named segment type.

---

## Filters

All search commands accept:

```bash
--application my-app          # filter to a specific application
--model gpt-4                 # filter to a specific model
--limit 20                    # number of results (default 10)
--min-score 0.75              # minimum cosine similarity threshold
```

```bash
stowken semantic-search "production failures" \
  --application backend-agent \
  --model gpt-4 \
  --min-score 0.8 \
  --limit 5
```

---

## Reading the matching conversations

Search returns IDs and scores. Use `retrieve-batch` to read the actual conversations:

```bash
# Shell one-liner: search → retrieve the top 3
stowken semantic-search "payment errors" --limit 3 --json \
  | jq -r '.[].conversation_id' \
  | stowken retrieve-batch --stdin
```

Output:

```
━━━ conv-8821 (gpt-4) ━━━
[USER]
My payment keeps failing, I've tried three times.

[ASSISTANT]
I can see there's a declined transaction on your account. The error code
suggests your card issuer is blocking the charge — this can happen with...
```

---

## Summary strategies

The summary strategy controls how each conversation is compressed into a single text before being embedded.

### ConcatTruncate (default, recommended)

```bash
stowken embed-all --openai-key $OPENAI_API_KEY
# --summary-strategy concat is the default, no flag needed
```

Detokenizes all segments with role markers:

```
[system] You are a helpful assistant.
[user] My payment keeps failing.
[assistant] I can see there's a declined transaction...
```

Truncates to 24,000 characters (fits any major embedding model's context window). Free — no extra API calls. Works well for the vast majority of retrieval queries.

### LlmGenerated (opt-in)

```bash
stowken embed-all \
  --openai-key $OPENAI_API_KEY \
  --summary-strategy llm \
  --summary-model gpt-4o-mini
```

Calls GPT-4o-mini once per conversation to generate a one-paragraph summary like:

> "User encountered repeated payment failures. The agent identified the card issuer was blocking the charge due to a fraud flag, walked the user through contacting their bank, and confirmed the issue was resolved when the user successfully completed a test transaction."

Better for behavioral queries ("sessions where the agent apologized and escalated") because it captures intent and outcome rather than raw content. Costs one LLM call per conversation and is slower to index.

**When to use LLM summaries:** if your conversations are long (>2K tokens), have complex multi-turn structure, or you're doing behavioral analysis rather than topic retrieval. For most use cases, ConcatTruncate is the right starting point.

---

## Plugging in a different embedding model

Implement `EmbeddingAdapter` in Rust:

```rust
use stowken::types::EmbeddingAdapter;

struct MyEmbedder { /* ... */ }

impl EmbeddingAdapter for MyEmbedder {
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        // call your model
    }
    fn dimension(&self) -> usize { 768 }
    fn model_name(&self) -> &str { "my-model-v1" }
}

vault.set_embedding_adapter(MyEmbedder { /* ... */ });
```

Or in Python:

```python
def my_embed(texts: list[str]) -> list[list[float]]:
    return my_model.encode(texts).tolist()

vault.set_embedding_adapter(my_embed, dimension=768, model_name="my-model-v1")
```

The model name is stored alongside every embedding, so you can switch models and re-embed without losing old embeddings — old and new model embeddings coexist, and search always uses the currently configured model.

---

## Python API reference

```python
# Setup
vault.use_openai_embeddings(api_key, model="text-embedding-3-small")
vault.use_concat_summary(max_chars=24_000)          # default, free
vault.use_openai_summarizer(api_key, model="gpt-4o-mini")  # opt-in LLM summaries
vault.set_embed_on_store(True)                      # default True

# Indexing
stats = vault.embed_all()
# {"segments_embedded": N, "conversations_embedded": M, "embedding_model": "...", ...}

# Search
hits = vault.semantic_search(
    "production failures",
    granularity="both",     # "both" | "segment" | "conversation"
    limit=10,
    min_score=0.0,
    model=None,             # filter by model name
    application=None,       # filter by application tag
)
# [{"conversation_id": "...", "score": 0.91, "matched_via": {...}, "model": "...", ...}]

# Typed wrappers
vault.search_prompts("billing instructions", limit=10)
vault.search_responses("I cannot help", limit=10)
vault.search_user_questions("how do I", limit=10)
vault.search_tool_uses("sql query", limit=10)
vault.search_conversations("debugging session", limit=10)

# Read the results
ids = [h["conversation_id"] for h in hits]
convs = vault.retrieve_batch(ids)
# [{"conversation_id": "...", "text": "[user] ...\n[assistant] ...", "turns": [...]}]
```
