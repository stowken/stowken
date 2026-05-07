# Analytics

Beyond search, Stowken supports cluster analysis and outlier detection over your full conversation archive. These workflows answer questions like "what topics do my users actually talk about?" and "which conversations don't fit any pattern?"

All analytics operations use conversation-level embeddings. Run `embed-all` first if you haven't already.

---

## Clustering

Group conversations into topic clusters using in-process k-means++ on conversation embeddings.

### CLI

```bash
stowken cluster --k 10 --representatives 3 --openai-key $OPENAI_API_KEY
```

Output:

```
Cluster 0 (87 conversations)
  Representatives: conv-0421, conv-1832, conv-0093
  Sample: [user] How do I reverse a list in Python? ...

Cluster 1 (134 conversations)
  Representatives: conv-2918, conv-0441, conv-3302
  Sample: [user] My payment keeps failing ...

Cluster 2 (62 conversations)
  Representatives: conv-0877, conv-2103, conv-0512
  Sample: [user] Can you write a poem about autumn ...
...
```

### Python

```python
clusters = vault.cluster_conversations(k=10, representative_count=3)

for c in clusters:
    print(f"Cluster {c['cluster_id']}: {c['size']} conversations")
    print(f"  Top members: {c['representative_ids']}")
    print(f"  All: {len(c['members'])} IDs")
```

### Reading cluster representatives

The `representative_ids` are the conversations closest to each cluster centroid — the most "typical" example of that cluster. Read them to understand what a cluster is about:

```python
clusters = vault.cluster_conversations(k=10, representative_count=2)

for cluster in clusters:
    reps = vault.retrieve_batch(cluster["representative_ids"])
    print(f"\n=== Cluster {cluster['cluster_id']} ({cluster['size']} conversations) ===")
    for conv in reps:
        # Show just the first user turn as a preview
        first_user = next((t for t in conv["turns"] if t["role"] == "user"), None)
        if first_user:
            print(f"  [{conv['conversation_id']}] {first_user['text'][:120]}")
```

### Choosing k

There's no universal answer. Practical guidance:

- Start with `k = sqrt(N / 2)` as a rough heuristic (e.g. k=10 for 200 conversations, k=70 for 10,000)
- Run with a few values and read the representatives — you'll know if clusters feel distinct
- Clusters with 2–3 conversations are probably over-segmented; merge with a lower k
- Clusters where representatives look unrelated to each other need a higher k

---

## Outlier detection

Find conversations that don't fit any cluster — typically the most interesting ones.

### CLI

```bash
stowken outliers --k 10 --threshold 0.4 --limit 20 --openai-key $OPENAI_API_KEY
```

Output:

```
conv-7721   isolation=0.89   (gpt-4, my-app, 2024-03-12)
conv-2083   isolation=0.84   (gpt-4, my-app, 2024-03-08)
conv-9912   isolation=0.81   (gpt-4, my-app, 2024-03-15)
...
```

`isolation_score` is the distance from the conversation's nearest cluster centroid. Higher means more unusual.

### Python

```python
outliers = vault.find_outliers(k=10, threshold=0.4, limit=20)

for o in outliers:
    print(f"{o['conversation_id']}  score={o['isolation_score']:.3f}")
```

### Reading outliers

```python
outliers = vault.find_outliers(k=10, threshold=0.4, limit=10)
ids = [o["conversation_id"] for o in outliers]
convs = vault.retrieve_batch(ids)

for conv, outlier in zip(convs, outliers):
    print(f"\n=== Outlier (isolation={outlier['isolation_score']:.2f}) ===")
    print(conv["text"][:500])
```

Outlier conversations are worth reading manually. They often reveal:
- Jailbreak attempts or unexpected use cases
- Bug reports or user frustration not captured in typical flows
- Rare but important edge cases (e.g. a user who successfully used a complex multi-step workflow)
- Data quality issues (malformed conversations, test data in production)

---

## End-to-end analysis workflow

Here's a complete workflow for understanding a corpus of 10,000 agent conversations:

```python
import stowken
import json

vault = stowken.Stowken.open("/var/lib/stowken")
vault.use_openai_embeddings(api_key="sk-...", model="text-embedding-3-small")

# Step 1: Index (once)
stats = vault.embed_all()
print(f"Indexed {stats['conversations_embedded']} conversations")

# Step 2: Find clusters
clusters = vault.cluster_conversations(k=12, representative_count=3)
print(f"\nFound {len(clusters)} clusters:")
for c in sorted(clusters, key=lambda x: -x["size"]):
    print(f"  Cluster {c['cluster_id']}: {c['size']} conversations")

# Step 3: Read the top representative from each cluster to label them
print("\nCluster topics (first representative each):")
for cluster in sorted(clusters, key=lambda x: -x["size"])[:5]:
    reps = vault.retrieve_batch([cluster["representative_ids"][0]])
    if reps:
        first_user = next((t["text"] for t in reps[0]["turns"] if t["role"] == "user"), "")
        print(f"  [{cluster['size']}] {first_user[:100]}")

# Step 4: Search within a cluster of interest
# (e.g. after reading reps you decide cluster 3 is "payment issues")
payment_hits = vault.semantic_search(
    "payment failure declined card",
    granularity="conversation",
    limit=20,
)
payment_ids = [h["conversation_id"] for h in payment_hits]
payment_convs = vault.retrieve_batch(payment_ids)

# Step 5: Find outliers
outliers = vault.find_outliers(k=12, threshold=0.5, limit=10)
print(f"\nTop outlier: {outliers[0]['conversation_id']} "
      f"(isolation={outliers[0]['isolation_score']:.2f})")
odd_conv = vault.retrieve_batch([outliers[0]["conversation_id"]])
if odd_conv:
    print(odd_conv[0]["text"][:800])
```

---

## Exporting for downstream analysis

The output of any search or cluster call is plain Python dicts, so it integrates directly with pandas, JSON files, or any downstream tool:

```python
import pandas as pd

# Cluster membership as a DataFrame
clusters = vault.cluster_conversations(k=10, representative_count=1)
rows = [
    {"cluster_id": c["cluster_id"], "conversation_id": cid}
    for c in clusters
    for cid in c["members"]
]
df = pd.DataFrame(rows)
df.to_csv("cluster_assignments.csv", index=False)

# Search results as a DataFrame
hits = vault.semantic_search("user frustrated", limit=100)
df_hits = pd.DataFrame(hits)
print(df_hits[["conversation_id", "score", "model"]].head(10))

# Full conversation text for LLM analysis
convs = vault.retrieve_batch([h["conversation_id"] for h in hits[:10]])
for conv in convs:
    # Pass conv["text"] to any LLM for further analysis
    pass
```

---

## CLI reference

```bash
# Cluster
stowken cluster --k 10 --representatives 3 --openai-key $OPENAI_API_KEY

# Outliers
stowken outliers --k 10 --threshold 0.4 --limit 20 --openai-key $OPENAI_API_KEY

# Read results from search (pipe pattern)
stowken semantic-search "payment errors" --limit 5 --json \
  | jq -r '.[].conversation_id' \
  | stowken retrieve-batch --stdin
```
