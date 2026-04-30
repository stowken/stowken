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

## v0.3 — Cross-segment compression *(complete)*

**Theme:** bring back zstd dictionaries with the rotation bug fixed at the format layer.

- [x] **Frame-versioned dictionaries.** *Shipped: new frame byte `0x03 = varint + zstd + dict(id)`. The 4-byte LE dict ID directly follows the version tag. Multiple dictionaries coexist at runtime; the decompressor selects by ID. Rotation safety is now a property of the frame format — pinned by `segments_compressed_with_old_dict_remain_readable_after_rotation`.*
- [x] **`vault.train_dictionary()` + `activate_dictionary()` + `list_dictionaries()`.** *Shipped via `Stowken`, the CLI (`stowken dict {train,list,activate}`), and Python (sync + `_async`). Dicts persist under `<vault-root>/dictionaries/{id}.zstd` plus `{id}.json` info sidecars and an `active` marker.*
- [x] **DictRegistry caches bytes per ID.** *Shipped: lazy load on first decompress, in-memory cache via `Arc<RwLock<HashMap<DictId, Arc<Vec<u8>>>>>`. LRU eviction deferred — typical vaults have few enough dicts that unbounded growth isn't a concern; revisit if pressure becomes real.*
- [ ] **Optional auto-train policy** that periodically trains a new dict from recent segments and adds it to the registry. Old dicts kept; never silently replaced. *Deferred to v0.3.x — manual `dict train --activate` is fine for a first release; auto-train policy is mostly orchestration.*

**Value (measured):** modest on OpenAssistant where dedup contributes ~0% and base zstd already gets 54%. Larger gains expected on domain-specific corpora (medical, code, legal) where token frequency distributions differ substantially from general text — to be benchmarked in v0.3.x or v0.4 with a domain corpus.

---

## v0.4 — Near-duplicate detection *(complete)*

**Theme:** catch prompts that should be the same but aren't (interpolated names, formatting drift, prompt versioning).

- [x] **Persistent MinHash LSH index.** *Shipped: two SQLite tables (`near_dedup_signatures` BLOB-keyed, `near_dedup_bands` for LSH bucket lookup). Survives restart. `reindex` rebuilds it from on-disk manifests. Pinned by `lsh_index_survives_vault_reopen`.*
- [x] **Bounded memory.** *Shipped: tokens are fetched from the backend on demand at delta-compute time, never cached in memory. Memory grows with one in-flight ingest, not corpus size.*
- [x] **Delta-encoded variant storage.** *Shipped: frame byte `0x04 = canonical_hash(32 bytes) || streaming delta`. Three opcodes (Skip / Copy / Insert) with a single cursor through the canonical. Compact varint-encoded. The vault refuses to chain deltas — variants always reference root canonicals.*
- [x] **`stowken find-near-duplicates --threshold 0.85`.** *Shipped, plus the matching Python `find_near_duplicates(...)` / `_async` variant. Surfaces clusters with similarity-sorted variants.*
- [x] **Opt-in only — falls back gracefully when delta isn't a win.** *`StowkenConfig::near_dedup_threshold = None` by default. When enabled, delta storage is rejected if its size exceeds a fresh full compression — no cross-ref overhead for negative gain.*
- [ ] **Configurable similarity threshold per segment type.** *Deferred — single global threshold is the right starting point; per-type tuning is an optimisation once we have real corpus data.*

**Value:** catches what exact dedup misses — interpolated user names, slight RAG context differences, prompt versioning drift. On corpora with realistic prompt drift, typically 5–20% additional savings on top of exact dedup.

---

## v0.5 — Token-level shared-substring dedup *(complete)*

**Theme:** extract savings from sub-segment repetition. The most ambitious of the three.

- [x] **Discovery via rolling-hash window counting.** *Shipped in v0.5.0: every `MIN_LENGTH`-token window across a sample of segments is hashed; windows clearing the distinct-segment threshold are promoted. Simpler than a full suffix-array; v0.5.x can extend with variable-length matching.*
- [x] **Promotion policy via cost-model fallback.** *Shipped: substring frame is accepted only if its byte length beats a fresh full compression of the same tokens. Strict opt-in win — never net regression.*
- [x] **Reference-by-substring frame format.** *Shipped: new frame byte `0x05` with `Inline(tokens)` and `Ref(substring_id)` ops, varint-encoded.*
- [x] **Substring registry.** *Shipped under `<vault-root>/substrings/{blobs.dat, index.json}`. Packed-file storage (one concatenated blob, not a file per substring) so a 3,000-substring registry costs ~600 KB instead of ~24 MB. In-memory cache populated eagerly at open.*
- [x] **Backwards-compatible retrieval.** *Shipped: frames written before substring promotion stay readable forever. The vault's substring-aware compress only runs at write time.*
- [x] **Optional compaction job.** *Shipped in v0.5.1: `stowken substrings compact` walks every segment, decompresses, retries the substring path, and replaces the stored bytes when smaller. Atomic per-segment via `StorageBackend::replace_segment_data`. Skips 0x04 and 0x05 frames (already in their final form).*
- [x] **Substring GC.** *Shipped in v0.5.1: `stowken substrings gc` walks every segment, decodes 0x05 frames to collect referenced substring ids, and prunes the registry. Atomic packed-file rebuild via `SubstringRegistry::retain`.*
- [ ] **Variable-length substring extension.** *Deferred — discovery currently registers fixed `MIN_LENGTH`-token windows. Extending outward when surrounding tokens also recur would tighten the registry. Optimization, not a correctness gap.*

**Value (measured):** on the synthetic large-segment drift corpus, substrings push savings from 58.8% (baseline) to **78.8%** (`train` only) to **83.8%** (`train` + `compact` for retroactive rewrite of existing data). On OpenAssistant the cost-model promotes zero substrings — no regression on workloads without genuine substring drift.

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
