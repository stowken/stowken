# Python Cookbook

Practical recipes for common Stowken workflows in Python. These are designed to run in a Jupyter notebook or script.

---

## Setup

```python
import stowken

# Open a persistent vault
vault = stowken.Stowken.open("/var/lib/stowken")

# Enable semantic search
vault.use_openai_embeddings(api_key="sk-...", model="text-embedding-3-small")

# Index existing conversations (run once; new stores embed automatically)
stats = vault.embed_all()
print(f"Indexed {stats['conversations_embedded']} conversations, "
      f"{stats['segments_embedded']} unique segments")
```

---

## Recipe 1: Search and read

Find conversations on a topic, then read the top results.

```python
hits = vault.semantic_search("user struggling with authentication", limit=10)

print(f"Found {len(hits)} hits\n")
for h in hits:
    print(f"  {h['conversation_id']}  score={h['score']:.3f}  via={h['matched_via']['kind']}")

# Read the top 3
convs = vault.retrieve_batch([h["conversation_id"] for h in hits[:3]])
for conv in convs:
    print(f"\n{'='*60}")
    print(f"ID: {conv['conversation_id']}  Model: {conv['model']}")
    print(conv["text"][:1000])
```

---

## Recipe 2: Typed role search

Search within specific message roles.

```python
# Find system prompts related to customer support
support_prompts = vault.search_prompts("customer support refund policy", limit=10)

# Find assistant responses that decline requests
refusals = vault.search_responses("I cannot help with that", limit=20)

# Find user messages showing frustration
frustrated = vault.search_user_questions("this isn't working I give up", limit=10)

# Find tool calls involving databases
db_calls = vault.search_tool_uses("sql query execute", limit=10)

# Print what the matched segment was for each hit
for h in refusals:
    via = h["matched_via"]
    print(f"{h['conversation_id']}  score={h['score']:.3f}  "
          f"segment_type={via.get('segment_type', 'conversation')}")
```

---

## Recipe 3: Build a pandas DataFrame from search results

```python
import pandas as pd

hits = vault.semantic_search("production error debugging", limit=100, min_score=0.6)

# Basic hit DataFrame
df = pd.DataFrame([{
    "conversation_id": h["conversation_id"],
    "score": h["score"],
    "model": h["model"],
    "application": h.get("application"),
    "matched_via": h["matched_via"]["kind"],
    "created_at": h["created_at"],
} for h in hits])

df["created_at"] = pd.to_datetime(df["created_at"])
df = df.sort_values("score", ascending=False)
print(df.head(10))

# Score distribution
print(df["score"].describe())
print(df["model"].value_counts())
```

---

## Recipe 4: Cluster analysis with readable labels

Cluster conversations and label each cluster by reading its top representative.

```python
clusters = vault.cluster_conversations(k=12, representative_count=2)

# Sort by size, largest first
clusters = sorted(clusters, key=lambda c: -c["size"])

labeled = []
for cluster in clusters:
    # Read first representative to get a sense of the cluster
    reps = vault.retrieve_batch([cluster["representative_ids"][0]])
    if not reps:
        continue
    first_user = next(
        (t["text"] for t in reps[0]["turns"] if t["role"] == "user"),
        "(no user turn)"
    )
    labeled.append({
        "cluster_id": cluster["cluster_id"],
        "size": cluster["size"],
        "preview": first_user[:120],
        "representative_id": cluster["representative_ids"][0],
        "all_ids": cluster["members"],
    })
    print(f"Cluster {cluster['cluster_id']} ({cluster['size']} convs): {first_user[:80]}")

# Save cluster assignments
rows = [
    {"cluster_id": c["cluster_id"], "conversation_id": cid}
    for c in clusters
    for cid in c["all_ids"]
]
pd.DataFrame(rows).to_csv("cluster_assignments.csv", index=False)
```

---

## Recipe 5: Find and inspect outliers

```python
outliers = vault.find_outliers(k=12, threshold=0.5, limit=20)

print(f"Found {len(outliers)} outlier conversations\n")

# Read them
outlier_ids = [o["conversation_id"] for o in outliers]
outlier_convs = vault.retrieve_batch(outlier_ids)

for conv, meta in zip(outlier_convs, outliers):
    print(f"\n{'='*60}")
    print(f"ID: {conv['conversation_id']}  isolation={meta['isolation_score']:.3f}")
    print(conv["text"][:600])
    print("...")
```

---

## Recipe 6: Cost and usage breakdown

```python
# Token usage by model and application
stats = vault.stats()
print(f"Total tokens: {stats['total_tokens']:,}")
print(f"Unique segments: {stats['unique_segments']:,}")
print(f"Dedup ratio: {stats['dedup_ratio']:.1%}")

# Per-role breakdown
seg_stats = vault.segment_stats()
for s in seg_stats:
    print(f"{s['segment_type']:15}  {s['unique']:6} unique  "
          f"{s['refs']:6} refs  {s['dedup_pct']:.0f}% dedup  "
          f"avg {s['avg_tokens']} tokens")
```

---

## Recipe 7: Async usage in an event loop

All methods have `_async` variants for use inside `asyncio` event loops.

```python
import asyncio
import stowken

async def analyze():
    vault = stowken.Stowken.open("/var/lib/stowken")
    vault.use_openai_embeddings(api_key="sk-...", model="text-embedding-3-small")

    # Embed and search concurrently
    hits = await vault.semantic_search_async("payment errors", limit=10)
    ids = [h["conversation_id"] for h in hits]
    convs = await vault.retrieve_batch_async(ids)

    for conv in convs:
        print(conv["conversation_id"], conv["text"][:100])

asyncio.run(analyze())
```

---

## Recipe 8: Store conversations from the OpenAI API

If you're calling the OpenAI API directly, archive conversations on the fly:

```python
from openai import OpenAI
import stowken

client = OpenAI()
vault = stowken.Stowken.open("/var/lib/stowken")
vault.use_openai_embeddings(api_key="sk-...", model="text-embedding-3-small")

def chat_and_store(user_message: str, system_prompt: str, application: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    # Store the full conversation
    vault.store({
        "model": "gpt-4o-mini",
        "tokenizer": "cl100k_base",
        "application": application,
        "messages": messages,
    })
    return assistant_reply

reply = chat_and_store(
    "How do I reverse a list in Python?",
    "You are a helpful Python tutor.",
    "python-tutor",
)
print(reply)
```

---

## Recipe 9: Export for fine-tuning

Export user/assistant pairs for supervised fine-tuning:

```python
# Search for high-quality examples on a specific topic
hits = vault.semantic_search("clear step-by-step explanation", limit=200, min_score=0.7)
ids = [h["conversation_id"] for h in hits]
convs = vault.retrieve_batch(ids)

# Build JSONL fine-tuning dataset
import json

with open("finetune.jsonl", "w") as f:
    for conv in convs:
        turns = conv["turns"]
        # Build message pairs
        messages = [{"role": t["role"], "content": t["text"]} for t in turns]
        if len(messages) >= 2:
            f.write(json.dumps({"messages": messages}) + "\n")

print(f"Exported {len(convs)} conversations")
```

Or use the built-in export:

```python
# Built-in export (all conversations, deduped)
with open("pairs.jsonl", "w") as f:
    vault.export_training_data(output=f)
```

---

## Reference: return shapes

**`semantic_search()` / typed search wrappers**
```python
{
    "conversation_id": str,
    "score": float,               # cosine similarity [0, 1]
    "matched_via": {
        "kind": "conversation" | "segment",
        "segment_type": str,      # only present for segment matches
        "hash": str,              # only present for segment matches
    },
    "model": str,
    "application": str | None,
    "created_at": str,            # ISO 8601
}
```

**`retrieve_batch()`**
```python
{
    "conversation_id": str,
    "model": str,
    "application": str | None,
    "text": str,                  # full conversation with [role] markers
    "turns": [
        {"role": str, "text": str},  # in order
    ],
    "created_at": str,
}
```

**`cluster_conversations()`**
```python
{
    "cluster_id": int,
    "size": int,
    "representative_ids": [str],  # closest to centroid, length = representative_count
    "members": [str],             # all conversation IDs in this cluster
}
```

**`find_outliers()`**
```python
{
    "conversation_id": str,
    "isolation_score": float,     # distance from nearest centroid; higher = more outlier
    "model": str,
    "application": str | None,
    "created_at": str,
}
```

**`embed_all()`**
```python
{
    "segments_embedded": int,
    "segments_skipped": int,
    "segments_already_done": int,
    "conversations_embedded": int,
    "conversations_already_done": int,
    "embedding_model": str,
    "summary_strategy": str,      # "concat" or "llm:gpt-4o-mini"
}
```
