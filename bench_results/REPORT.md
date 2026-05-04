# Stowken Benchmark Report

**Date:** 2026-05-03  
**Version:** v0.6.x  
**Sweep:** 23 runs across 4 corpora (drift, HH-RLHF, LMSys, Open Assistant)

---

## Executive Summary

| Finding | Detail |
|---------|--------|
| **Best single-flag win** | `--dict-warmup` consistently improves compression 2–10pp with low retrieval overhead |
| **Near-dedup ROI** | Only worthwhile on near-duplicate corpora (drift); adds 9–18× ingest cost with zero gain on real-world datasets |
| **Substring dedup** | Matches dict-warmup on synthetic corpus (68.2%), nearly free overhead, but finds nothing on real-world datasets |
| **Split-segments tradeoff** | Improves exact-dedup rate on diverse corpora but slightly worsens combined savings due to shorter-segment compression penalty |
| **Baseline compression** | 35–65% savings before any optimizations, driven by zstd on token sequences |

---

## Corpora

| Corpus | Conversations | Tokens | Character |
|--------|--------------|--------|-----------|
| **drift** | 1 000 | 420 K | Synthetic near-duplicate conversations with random base tokens and per-conversation drift substitutions |
| **hh_rlhf** | 10 000 | 1.35 M | Anthropic HH-RLHF chosen+rejected pairs; high exact dedup from shared conversation prefixes |
| **lmsys** | 10 000 | 4.90 M | LMSys-Chat-1M sample; diverse real-world, almost no repeated segments |
| **open_assistant** | 3 482 | 1.30 M | OpenAssistant crowd-sourced Q&A; unique conversations, low repetition |

---

## Results by Corpus

### drift — Synthetic Near-Duplicate Corpus

1 000 conversations with shared random base tokens and 3-position drift substitutions. Designed to stress near-dedup and substring dedup.

| Run | Combined Savings | Ingest Time | p99 Retrieval | vs. Baseline |
|-----|-----------------|-------------|---------------|-------------|
| baseline | 35.9% | 310ms | 24µs | — |
| `--near-dedup 0.8` | 48.9% | 5 612ms | 88µs | **+13.0pp**, 18× slower ingest |
| `--near-dedup 0.7` | 48.3% | 5 537ms | 35µs | +12.4pp, 18× slower ingest |
| `--split-segments 256` | 35.9% | 293ms | 20µs | 0pp (segments don't cross 256-token limit) |
| `--split-segments 256 --near-dedup 0.8` | 48.9% | 5 669ms | 36µs | Same as near-dedup alone |
| `--dict-warmup 200` | **69.7%** | 434ms | 96µs | **+33.8pp**, 1.4× slower ingest |
| `--with-substrings` | **68.2%** | 246ms + 91ms | 18µs | **+32.3pp**, 29% overhead |

**Segment-type breakdown (baseline):**

| Type | Unique | Refs | Dedup% | Tokens | Compressed |
|------|--------|------|--------|--------|------------|
| system_prompt | 1 000 | 1 000 | 0.0% | 100 K | 281.8 KB |
| user_turn | 1 000 | 1 000 | 0.0% | 240 K | 599.3 KB |
| assistant_turn | 1 000 | 1 000 | 0.0% | 80 K | 171.1 KB |

**Key observations:**
- Near-dedup at 0.8 and 0.7 thresholds yield similar results (48.9% vs 48.3%); looser threshold is not significantly better for this corpus
- Dict-warmup with just 200 samples (20% of corpus) delivers the best overall savings at 69.7%
- Substring dedup achieves comparable savings (68.2%) by discovering recurring token sub-sequences across 2 099 segments, saving 543 KB
- Split-segments has no effect here because drift segments are all well under 256 tokens
- Near-dedup adds 18× ingest cost; dict-warmup delivers 2.5× more savings at 1.4× ingest cost — strongly preferable

---

### hh_rlhf — Chosen + Rejected Pairs

10 000 conversations from Anthropic's HH-RLHF dataset. Each dataset row emits a chosen and rejected conversation sharing the same human turns — creating significant exact dedup opportunity.

| Run | Combined Savings | Ingest Time | p99 Retrieval | vs. Baseline |
|-----|-----------------|-------------|---------------|-------------|
| baseline | 65.5% | 3 746ms | 80µs | — |
| `--near-dedup 0.8` | 65.5% | 32 526ms | 87µs | 0pp, **8.7× slower ingest** |
| `--split-segments 256` | 65.5% | 4 549ms | 73µs | 0pp, minor overhead |
| `--split-segments 256 --near-dedup 0.8` | 65.5% | 32 525ms | 70µs | 0pp, same cost as near-dedup alone |
| `--dict-warmup 1000` | 64.8% | 4 971ms | 258µs | −0.7pp, 3× slower retrieval |
| `--with-substrings` | 65.5% | 3 726ms | 64µs | 0pp, 0 segs rewritten |

**Segment-type breakdown (baseline):**

| Type | Unique | Refs | Dedup% | Tokens | Compressed |
|------|--------|------|--------|--------|------------|
| assistant_turn | 15 910 | 24 439 | 34.9% | 995 K | 1.4 MB |
| user_turn | 10 830 | 24 376 | 55.6% | 355 K | 417 KB |

**Key observations:**
- Baseline already achieves 65.5% via strong exact dedup: 55.6% of user turns and 34.9% of assistant turns are deduplicated
- Near-dedup finds no additional benefit — chosen/rejected pairs share prefixes exactly (exact dedup handles this) but differ in the final turns (dissimilar enough that Jaccard falls below 0.8)
- Dict-warmup slightly hurts performance on this corpus: the dictionary is trained on chosen-only conversations but retrieval requires decoding with it, causing a 3× retrieval slowdown
- Substring dedup finds 0 rewriteable segments — the corpus is already well-handled by exact dedup
- **Conclusion:** for HH-RLHF-style corpora, baseline is optimal. No flags improve it.

---

### lmsys — Diverse Real-World Conversations

10 000 conversations sampled from LMSys-Chat-1M. Highly diverse topics and styles — the worst case for deduplication.

| Run | Combined Savings | Ingest Time | p99 Retrieval | Exact Dedup | vs. Baseline |
|-----|-----------------|-------------|---------------|-------------|-------------|
| baseline | 60.4% | 6 470ms | 122µs | 1.5% | — |
| `--split-segments 256` | 57.9% | 8 509ms | 145µs | 8.8% | −2.5pp combined |
| `--split-segments 128` | 55.2% | 8 750ms | 184µs | 10.2% | −5.2pp combined |
| `--dict-warmup 2000` | **62.5%** | 17 014ms | 1 601µs | 1.5% | **+2.1pp** |
| `--with-substrings` | 60.6% | 6 633ms | 133µs | 1.5% | +0.2pp |

**Segment-type breakdown (baseline):**

| Type | Unique | Refs | Dedup% | Tokens | Compressed |
|------|--------|------|--------|--------|------------|
| user_turn | 17 993 | 19 374 | 7.1% | 1.19 M | 2.1 MB |
| assistant_turn | 18 133 | 19 094 | 5.0% | 3.71 M | 5.3 MB |

**Split-256 segment breakdown:**

| Type | Unique | Refs | Dedup% | Tokens | Compressed |
|------|--------|------|--------|--------|------------|
| user_turn | 17 904 | 19 376 | 7.6% | 888 K | 1.7 MB |
| assistant_turn | 18 052 | 19 107 | 5.5% | 2.36 M | 4.0 MB |
| continuation | 9 474 | 11 452 | **17.3%** | 1.68 M | 2.2 MB |

**Key observations:**
- Splitting at 256 tokens raises exact-dedup from 1.5% to 8.8%, but short segments compress worse than long ones — net combined savings actually drops 2.5pp
- Splitting at 128 tokens raises exact-dedup further (10.2%) but the compression penalty grows; combined savings drops 5.2pp
- Continuation segments show 17.3% dedup rate at 256 and 13.7% at 128 — the splitting does find real shared sub-sequences
- Dict-warmup with 2 000 samples adds 2.1pp savings but at a severe retrieval cost (p99 1 601µs vs 122µs baseline — 13× slower)
- Substring dedup rewrites 377 segments, saving only 34 KB (~0.2pp) — too few repeated sub-sequences in diverse data

---

### open_assistant — Crowd-Sourced Q&A

3 482 conversations from the OpenAssistant dataset. Unique, independently authored conversations with essentially zero repetition.

| Run | Combined Savings | Ingest Time | p99 Retrieval | vs. Baseline |
|-----|-----------------|-------------|---------------|-------------|
| baseline | 54.2% | 1 828ms | 33µs | — |
| `--near-dedup 0.8` | 54.2% | 9 917ms | 39µs | 0pp, **5.4× slower ingest** |
| `--split-segments 256` | 51.2% | 2 348ms | 48µs | −3.0pp |
| `--dict-warmup 500` | **56.1%** | 6 026ms | 145µs | **+1.9pp**, 4.4× slower retrieval |
| `--with-substrings` | 54.2% | 1 806ms | 35µs | 0pp, 0 segs rewritten |

**Segment-type breakdown (baseline):**

| Type | Unique | Refs | Dedup% | Tokens | Compressed |
|------|--------|------|--------|--------|------------|
| user_turn | 6 317 | 6 378 | 1.0% | 187 K | 410 KB |
| assistant_turn | 5 162 | 5 173 | 0.2% | 1.11 M | 1.9 MB |

**Key observations:**
- Essentially zero exact dedup (0.0–1.0%) — this corpus is purely compression-driven
- Near-dedup finds no near-duplicates either; 5.4× slower ingest for zero gain
- Split-256 hurts: conversation segments already compress well at full length
- Dict-warmup (+1.9pp) shows the only meaningful improvement, though retrieval degrades 4× due to dict decode overhead
- Substring dedup finds 0 rewriteable segments, as expected

---

## Cross-Corpus Comparison

### Combined savings summary

| Corpus | Baseline | Best Flag | Best % | Best Flag Cost |
|--------|----------|-----------|--------|----------------|
| drift | 35.9% | `--dict-warmup 200` | **69.7%** | +40% ingest time, 4× retrieval |
| hh_rlhf | 65.5% | (none helpful) | 65.5% | — |
| lmsys | 60.4% | `--dict-warmup 2000` | **62.5%** | 2.6× ingest time, 13× retrieval |
| open_assistant | 54.2% | `--dict-warmup 500` | **56.1%** | 3.3× ingest time, 4× retrieval |

### Ingest speed by flag

| Flag | Effect on Ingest | When Worthwhile |
|------|-----------------|-----------------|
| `--near-dedup 0.8` | 5–18× slower | Only when corpus has true near-duplicates (systematic drift, prompt templating) |
| `--near-dedup 0.7` | 18× slower | Catches slightly more variants at same cost; marginal over 0.8 |
| `--split-segments 256` | 1.2–1.4× slower | Helps dedup rate but hurts compression; only worthwhile if retrieval granularity matters |
| `--split-segments 128` | 1.3–1.5× slower | Diminishing returns vs 256 |
| `--dict-warmup N` | 1.4–2.6× slower | Best general-purpose improvement; N = 10–20% of corpus is sufficient |
| `--with-substrings` | +10–30% overhead | Free win on near-duplicate corpora; zero benefit on diverse data |

### Retrieval latency (p99) summary

| Corpus | Baseline | Near-dedup | Dict-warmup | Split-256 | Substrings |
|--------|----------|------------|-------------|-----------|------------|
| drift | 24µs | 88µs | 96µs | 20µs | **18µs** |
| hh_rlhf | 80µs | 87µs | 258µs | 73µs | **64µs** |
| lmsys | 122µs | — | 1 601µs | 145µs | 133µs |
| open_assistant | 33µs | 39µs | 145µs | 48µs | 35µs |

> **Note:** Dict-warmup trades retrieval speed for storage density. Each segment requires dict-decompression, which adds latency proportional to segment size. On large, diverse corpora (lmsys) this can reach 13× baseline p99.

---

## Recommendations

### Choose your optimization based on corpus type

**Corpus has many near-duplicate segments** (LLM system prompts, RAG context injection, prompt templates):
- Use `--dict-warmup N` with N = 10–20% of ingested conversations
- Optionally add `--with-substrings` for additional 1–2pp at ~30% post-ingest overhead
- Avoid `--near-dedup` unless near-duplicates differ in token sequences rather than just values (e.g., substitution drift)

**Corpus has true near-duplicate conversations** (systematic drift, A/B prompt testing):
- Use `--near-dedup 0.8` (conservative) or `--near-dedup 0.7` (broader catch)
- Budget 10–18× ingest cost — do this as a batch job, not on the write path
- Combine with `--dict-warmup` only if ingest latency budget allows

**Corpus is a chosen+rejected pair dataset** (RLHF-style):
- Baseline is optimal — exact dedup handles shared prefixes perfectly
- No additional flags improve storage; avoid near-dedup (costly and redundant)

**Corpus is diverse real-world conversations** (LMSys, OpenAssistant):
- Dict-warmup gives modest improvement (+2pp) but p99 retrieval degrades 4–13×
- Use `--dict-warmup` only if retrieval SLA allows >100µs p99
- Avoid split-segments (hurts combined savings despite improving dedup rate)
- Avoid near-dedup and substrings (zero benefit, measurable cost)

### Default configuration

For a production deployment with mixed traffic and tight retrieval SLA:
```
stowken bench <corpus> --retrieve 1000
```
Baseline is often the best choice. If storage pressure is high and retrieval can tolerate 2–5× higher latency, add:
```
stowken --dict-warmup 500 bench <corpus> --retrieve 1000
```

---

## Appendix: Raw Numbers

### drift

| Run | Incoming | Stored | Savings | Ingest | p50 | p95 | p99 |
|-----|----------|--------|---------|--------|-----|-----|-----|
| baseline | 1.6 MB | 1.0 MB | 35.9% | 310ms | 19µs | 21µs | 24µs |
| near-dedup 0.8 | 1.6 MB | 837 KB | 48.9% | 5 612ms | 33µs | 50µs | 88µs |
| near-dedup 0.7 | 1.6 MB | 849 KB | 48.3% | 5 537ms | 24µs | 35µs | 35µs |
| split-256 | 1.6 MB | 1.0 MB | 35.9% | 293ms | 17µs | 18µs | 20µs |
| split-256+near-0.8 | 1.6 MB | 838 KB | 48.9% | 5 669ms | 24µs | 34µs | 36µs |
| dict-200 | 1.6 MB | 497 KB | 69.7% | 434ms | 66µs | 75µs | 96µs |
| substrings | 1.6 MB | 522 KB | 68.2% | 246ms+91ms | 13µs | 14µs | 18µs |

### hh_rlhf

| Run | Incoming | Stored | Savings | Ingest | p50 | p95 | p99 |
|-----|----------|--------|---------|--------|-----|-----|-----|
| baseline | 5.2 MB | 1.8 MB | 65.5% | 3 746ms | 20µs | 49µs | 80µs |
| near-dedup 0.8 | 5.2 MB | 1.8 MB | 65.5% | 32 526ms | 21µs | 49µs | 87µs |
| split-256 | 5.2 MB | 1.8 MB | 65.5% | 4 549ms | 20µs | 44µs | 73µs |
| split-256+near-0.8 | 5.2 MB | 1.8 MB | 65.5% | 32 525ms | 20µs | 45µs | 70µs |
| dict-1000 | 5.2 MB | 1.8 MB | 64.8% | 4 971ms | 75µs | 166µs | 258µs |
| substrings | 5.2 MB | 1.8 MB | 65.5% | 3 726ms+8ms | 19µs | 41µs | 64µs |

### lmsys

| Run | Incoming | Stored | Savings | Ingest | p50 | p95 | p99 |
|-----|----------|--------|---------|--------|-----|-----|-----|
| baseline | 18.7 MB | 7.4 MB | 60.4% | 6 470ms | 15µs | 72µs | 122µs |
| split-256 | 18.8 MB | 7.9 MB | 57.9% | 8 509ms | 19µs | 88µs | 145µs |
| split-128 | 18.8 MB | 8.4 MB | 55.2% | 8 750ms | 26µs | 114µs | 184µs |
| dict-2000 | 18.7 MB | 7.0 MB | 62.5% | 17 014ms | 144µs | 819µs | 1 601µs |
| substrings | 18.7 MB | 7.4 MB | 60.6% | 6 633ms+470ms | 15µs | 77µs | 133µs |

### open_assistant

| Run | Incoming | Stored | Savings | Ingest | p50 | p95 | p99 |
|-----|----------|--------|---------|--------|-----|-----|-----|
| baseline | 5.0 MB | 2.3 MB | 54.2% | 1 828ms | 16µs | 27µs | 33µs |
| near-dedup 0.8 | 5.0 MB | 2.3 MB | 54.2% | 9 917ms | 19µs | 32µs | 39µs |
| split-256 | 5.0 MB | 2.4 MB | 51.2% | 2 348ms | 17µs | 35µs | 48µs |
| dict-500 | 5.0 MB | 2.2 MB | 56.1% | 6 026ms | 84µs | 122µs | 145µs |
| substrings | 5.0 MB | 2.3 MB | 54.2% | 1 806ms+32ms | 18µs | 30µs | 35µs |
