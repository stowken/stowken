# Stowken Roadmap

This is a *delivery plan*, not a wishlist. Each milestone has a theme, a small number of items, a clear value statement, and a rationale for its position in the sequence.

The three "deferred-with-intent" features cut from v0.1 — **versioned dictionaries**, **near-duplicate detection**, **token-level shared-substring dedup** — are tracked as v0.3 / v0.4 / v0.5 respectively. They were removed from v0.1 because the original implementations had correctness bugs (data loss on dictionary rotation, in-memory near-dedup index lost on restart, near-dedup linear-memory leak). They come back here with the architectural fixes baked in from the start.

---

## v0.1 — Local vault foundation *(shipped)*

Everything currently in the repo: segmentation, exact dedup, varint+zstd, frame versioning, manifest schema versioning, CLI, Python binding, three differentiator commands (`audit-system-prompts`, `cost`, `find-duplicates`). 94 Rust + 28 Python tests green.

---

## v0.2 — Validation & ergonomics *(complete)*

**Theme:** prove the v0.1 thesis on real production data; close obvious correctness gaps before adding new features.

- [x] **Real-world benchmark on public datasets.** *Shipped: `stowken bench <input-dir>` + `scripts/fetch_openassistant.py`. Headline number on OpenAssistant English (~3.5K conversations, 1.3M tokens): 54.2% saved, all from zstd, ~0% from dedup (no shared prompts in crowd-sourced data). Synthetic shared-prompt benchmark still hits 99.3%. README leads with the spectrum framing.*
- [x] **`stowken reindex`.** *Shipped: rebuilds SQLite from on-disk manifests by replaying through the same batch path `store()` uses. Idempotent. Verified end-to-end (delete metadata.db → reindex → identical stats). Exposed via CLI and Python.*
- [x] **Async Python API.** *Shipped: `_async` suffixed variant for every I/O method (`store_async`, `retrieve_async`, `delete_async`, `stats_async`, `segment_stats_async`, `gc_async`, `gc_candidates_async`, `reindex_async`) via pyo3-async-runtimes 0.21. Sync API unchanged. Sync + async share one Tokio runtime and can be mixed on the same vault. 10 new async tests including 50-task concurrent stores.*

**Definition of done:** real-world benchmark numbers in README; `stowken reindex` rebuilds a working index from a deleted DB; async Python passes the existing test suite plus an async-specific test file. *All three met.*

---

## v0.3 — Cross-segment compression

**Theme:** bring back zstd dictionaries with the rotation bug fixed at the format layer.

- [ ] **Frame-versioned dictionaries.** New frame byte `0x03 = varint + zstd + dict(id)`. The dict ID is encoded in the frame (4 bytes from `zstd::dict::DictId`). Multiple dictionaries coexist at runtime; the decompressor selects by ID. Rotation is safe because old frames carry their own ID and resolve to the right dict.
- [ ] **`vault.train_dictionary()` returns a versioned dict bytes blob.** Dictionaries are stored alongside `metadata.db` in a dict registry (path + ID-keyed lookup).
- [ ] **`vault.apply_dictionary(id)`** makes a registered dict the active writer. Old frames stay readable forever (their ID resolves through the registry).
- [ ] **Bounded in-memory dictionary cache.** LRU keyed by ID; cold dictionaries faulted in from disk on demand.
- [ ] **Optional auto-train policy** that periodically trains a new dict from recent segments and adds it to the registry. Old dicts kept; never silently replaced.

**Value:** typical +5–15 percentage points compression for unique-per-call content where dedup gets nothing. Most useful for domain-specific corpora (medical, code, legal) where token frequency distributions differ substantially from general text.

**Why this position:** smallest of the three deferred features, clearest mechanism (frame versioning is already in v0.1). Sets a pattern (versioned codecs + registry) that v0.4 and v0.5 reuse.

---

## v0.4 — Near-duplicate detection

**Theme:** catch prompts that should be the same but aren't (interpolated names, formatting drift, prompt versioning).

- [ ] **Persistent MinHash LSH index.** New SQLite tables for signatures and band buckets. Survives restart, rebuildable from segment data via `stowken reindex --near-dedup`. Fixes v0.1's in-memory-only bug.
- [ ] **Bounded memory.** No keeping every canonical segment's full token sequence in RAM. Tokens are fetched from the backend on demand when computing deltas. Fixes v0.1's linear-memory leak.
- [ ] **Delta-encoded variant storage.** New frame byte `0x04 = delta(canonical_hash, edit_script)`. Variants store the edit script against a canonical segment; retrieval reconstructs by applying the delta to the canonical's tokens.
- [ ] **`stowken find-near-duplicates --threshold 0.85`.** Surfaces variant clusters: "you have 47 system prompts that are 90%+ similar to this canonical."
- [ ] **Configurable similarity threshold per segment type.** System prompts probably want 0.95 (be conservative); tool results might tolerate 0.80.

**Value:** catches what exact dedup misses — interpolated user names, slight RAG context differences, prompt versioning drift. On corpora with realistic prompt drift, typically 5–20% additional savings on top of exact dedup.

**Why this position:** moderate lift, addresses a real gap that v0.3 doesn't touch (dictionaries help with byte-level repetition; near-dedup helps with structural variants).

---

## v0.5 — Token-level shared-substring dedup

**Theme:** extract savings from sub-segment repetition. The most ambitious of the three.

- [ ] **Suffix array (or FM-index) over hot segments.** Periodic background job identifies long common substrings shared across otherwise-distinct segments. E.g., "as a large language model trained by OpenAI" appearing across 50,000 unrelated assistant turns.
- [ ] **Promotion policy.** A substring earns its own blob when `(occurrences × substring_size) > (occurrences × reference_overhead + substring_size)`. Tunable threshold; default conservative.
- [ ] **Reference-by-substring frame format.** New frame byte `0x05`; segments can interleave inline tokens with `(substring_id, position)` references. Storage format must be append-only friendly.
- [ ] **Substring registry.** New SQLite table for promoted substrings, indexed by ID. Hot substrings cached in memory.
- [ ] **Backwards-compatible retrieval.** Frames written before substring promotion stay readable forever. Newly-stored segments can reference substrings; previously-stored segments are not retroactively rewritten (would require a separate compaction pass).
- [ ] **Optional compaction job.** `stowken compact` rewrites old segments to use the substring registry. Opt-in because it temporarily doubles storage.

**Value:** catches what segment-level dedup misses entirely. Most valuable on creative / unique-per-call content where exact dedup gets nothing but token sequences still repeat through the corpus. Could be the largest win of the three on the right workload.

**Why this position:** hardest of the three. Largest correctness surface (interleaved frames), needs a cost model, needs compaction strategy. Doing it last means we have v0.2's real-world data to know whether the substring distribution justifies the complexity.

**Risks to watch:** if substring promotion is too aggressive, retrieval becomes a graph walk and gets slow. If too conservative, we leave value on the table. Needs careful benchmarking.

---

## v0.6 — Cloud backends

**Theme:** scale beyond a single host.

- [ ] **S3 backend** with ETag-based optimistic ref-count writes (`If-Match` header, retry on `412`). Multi-writer safe. Re-add the v0.1 implementation that we cut, with the concurrency bug fixed.
- [ ] **Region-aware manifest cache** for hot conversations.
- [ ] **GCS / Azure backends** if there's demand. Same trait, same pattern.

**Why this position:** cloud is an operational problem, not a product problem. The differentiator is the dedup / compression / analytics, not the backend. Local-first is fine for v0.x; cloud once the product itself is proven.

---

## Out of scope (for now)

- **Hosted service.** No SaaS / managed Stowken. Library + CLI only.
- **Drop-in OpenAI proxy.** Capture-side integration belongs in a separate tool.
- **Multi-tenant isolation as a product feature.** Single vault per process. Multi-tenant is the customer's job.
- **Compliance modules** (audit log, retention, GDPR deletion verification). Useful, but not differentiating; better as a thin layer customers build over the API.
- **License / JWT validation** for paid features. Single open-source license (Apache 2.0). Charge for hosting and support if there's a business there, not for code.

---

## Tracking

This file is the source of truth for sequencing. When checking off an item, move it to the corresponding section of `CHANGELOG.md` under the version it shipped in. When adding new items, slot them by *theme*, not by *priority order*; promote the whole milestone if you want to reorder.

If a v0.x slips, slip the whole milestone — don't re-shuffle individual items across milestones, or the sequencing rationale stops making sense.
