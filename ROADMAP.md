# Stowken Roadmap

This is a *delivery plan*, not a wishlist. Each milestone has a theme, a small number of items, a clear value statement, and a rationale for its position in the sequence.

The three "deferred-with-intent" features cut from v0.1 — **versioned dictionaries**, **near-duplicate detection**, **token-level shared-substring dedup** — are tracked as v0.3 / v0.4 / v0.5 respectively. They were removed from v0.1 because the original implementations had correctness bugs (data loss on dictionary rotation, in-memory near-dedup index lost on restart, near-dedup linear-memory leak). They come back here with the architectural fixes baked in from the start.

---

## v0.1 — Local vault foundation *(shipped)*

Everything currently in the repo: segmentation, exact dedup, varint+zstd, frame versioning, manifest schema versioning, CLI, Python binding, three differentiator commands (`audit-system-prompts`, `cost`, `find-duplicates`). 94 Rust + 28 Python tests green.

---

## v0.2 — Validation & ergonomics *(shipped)*

**Theme:** prove the v0.1 thesis on real production data; close obvious correctness gaps before adding new features.

- [x] **Real-world benchmark on public datasets.** *Shipped: `stowken bench <input-dir>` + `scripts/fetch_openassistant.py`. Headline number on OpenAssistant English (~3.5K conversations, 1.3M tokens): 54.2% saved, all from zstd, ~0% from dedup (no shared prompts in crowd-sourced data). Synthetic shared-prompt benchmark still hits 99.3%. README leads with the spectrum framing.*
- [x] **`stowken reindex`.** *Shipped: rebuilds SQLite from on-disk manifests by replaying through the same batch path `store()` uses. Idempotent. Verified end-to-end (delete metadata.db → reindex → identical stats). Exposed via CLI and Python.*
- [x] **Async Python API.** *Shipped: `_async` suffixed variant for every I/O method via pyo3-async-runtimes 0.21. Sync API unchanged. Sync + async share one Tokio runtime and can be mixed on the same vault. 10 new async tests including 50-task concurrent stores.*

---

## v0.3 — Cross-segment compression *(shipped)*

**Theme:** bring back zstd dictionaries with the rotation bug fixed at the format layer.

- [x] **Frame-versioned dictionaries.** *Shipped: new frame byte `0x03 = varint + zstd + dict(id)`. Multiple dictionaries coexist at runtime; the decompressor selects by ID. Rotation safety is now a property of the frame format — pinned by `segments_compressed_with_old_dict_remain_readable_after_rotation`.*
- [x] **`vault.train_dictionary()` + `activate_dictionary()` + `list_dictionaries()`.** *Shipped via `Stowken`, CLI (`stowken dict {train,list,activate}`), and Python (sync + `_async`).*
- [x] **DictRegistry caches bytes per ID.** *Lazy load on first decompress, in-memory cache via `Arc<RwLock<HashMap>>`. LRU eviction deferred — typical vaults have few enough dicts that unbounded growth isn't a concern.*
- [ ] **Optional auto-train policy.** *Deferred to v0.3.x — manual `dict train --activate` is fine for a first release.*

**Value (measured, v0.6.1 sweep):** +2pp on LMSys (60.4% → 62.5% with 2,000-conv warmup), +1.9pp on OpenAssistant, +34pp on synthetic drift corpus. Retrieval latency cost is significant on diverse large corpora (LMSys p99: 122µs → 1,601µs). Best on domain-specific token distributions; not worth enabling for highly diverse traffic without measuring retrieval SLA impact.

---

## v0.4 — Near-duplicate detection *(shipped)*

**Theme:** catch prompts that are structurally almost identical but hash differently — interpolated values, slight formatting changes, A/B prompt versioning.

- [x] **Persistent MinHash LSH index.** *Shipped: two SQLite tables (`near_dedup_signatures`, `near_dedup_bands`). Survives restart. `reindex` rebuilds it. Pinned by `lsh_index_survives_vault_reopen`.*
- [x] **Bounded memory.** *Tokens fetched from backend on demand at delta-compute time, never cached. Memory grows with one in-flight ingest, not corpus size.*
- [x] **Delta-encoded variant storage.** *Frame byte `0x04 = canonical_hash(32 bytes) || streaming delta`. Three opcodes (Skip / Copy / Insert) with a single cursor through the canonical. The vault refuses to chain deltas — variants always reference root canonicals.*
- [x] **`stowken find-near-duplicates --threshold 0.85`.** *Surfaces drift clusters with similarity-sorted variants. Python `find_near_duplicates(...)` / `_async`.*
- [x] **Opt-in only — falls back gracefully when delta isn't a win.** *`StowkenConfig::near_dedup_threshold = None` by default.*
- [x] **SQLite query optimization (v0.6.1).** *Replaced 16 per-band queries with a single UNION ALL, and per-candidate `spawn_blocking` calls with a single batch signature fetch. Near-dedup ingest overhead dropped from ~55× to ~8–18× over baseline.*
- [ ] **Configurable threshold per segment type.** *Deferred — single global threshold is the right starting point.*

**Value (measured, v0.6.1 sweep):** +13pp on a synthetic corpus designed to have systematic per-conversation token drift. Zero additional savings over baseline on HH-RLHF, LMSys, and OpenAssistant — real-world traffic does not exhibit near-duplicate structure at the token level at these similarity thresholds. Ingest overhead remains 8–18× over baseline even after the SQLite optimization. **Use `stowken find-near-duplicates` to confirm near-duplicate structure in your data before enabling.**

---

## v0.5 — Token-level shared-substring dedup *(shipped)*

**Theme:** extract savings from sub-segment repetition — boilerplate that recurs inside otherwise-distinct segments.

- [x] **Discovery via rolling-hash window counting.** *Every `MIN_LENGTH`-token window across a sample of segments is hashed; windows clearing the distinct-segment threshold are promoted.*
- [x] **Promotion policy via cost-model fallback.** *Substring frame accepted only if its byte length beats a fresh full compression. Strict opt-in win — never net regression.*
- [x] **Frame format `0x05`.** *`Inline(tokens)` and `Ref(substring_id)` ops, varint-encoded.*
- [x] **Substring registry.** *Packed-file storage under `<vault-root>/substrings/{blobs.dat, index.json}`. 3,000-substring registry costs ~600 KB instead of ~24 MB with per-file storage. Eagerly cached in memory at open.*
- [x] **Compaction job (v0.5.1).** *`stowken substrings compact` retroactively rewrites pre-train segments. Atomic per-segment via `StorageBackend::replace_segment_data`. Skips 0x04 and 0x05 frames.*
- [x] **Substring GC (v0.5.1).** *`stowken substrings gc` prunes unreferenced substrings. Atomic packed-file rebuild.*
- [ ] **Variable-length substring extension.** *Deferred — fixed `MIN_LENGTH`-token windows are the right starting point; adaptive extension is an optimization.*

**Value (measured, v0.6.1 sweep):** +32pp on the synthetic drift corpus (35.9% → 68.2%) with 2,099 segments rewritten, 543 KB saved. Zero rewritten segments on HH-RLHF and OpenAssistant — the cost-model correctly finds nothing to promote when there are no recurring token-level sub-sequences. No retrieval overhead (substring frames decompress at the same speed as plain zstd).

---

## v0.6 — Cloud backends + bench enhancements *(shipped)*

**Theme:** scale beyond a single host; validate optimization flags across real corpora.

- [x] **S3 backend** with ETag-based optimistic ref-count writes. *`If-None-Match: *` for first-time meta writes, `If-Match: <etag>` for ref-count updates, 8-retry capped exponential backoff. Pinned by `s3_concurrent_increments_no_lost_writes`. Compatible with AWS S3, MinIO, localstack, R2, Spaces.*
- [x] **Python S3 integration.** *`stowken.Stowken.open_s3(bucket, local_db_path, ...)`. SQLite index stays local (rebuildable from S3 manifests via `reindex`).*
- [x] **Retrieval benchmarks** (`--retrieve N` flag). *Post-ingest sampling pass with p50/p95/p99 latency and throughput reporting.*
- [x] **LMSys-Chat-1M fetcher** (`scripts/fetch_lmsys.py`).
- [x] **HH-RLHF fetcher** (`scripts/fetch_hh_rlhf.py`).
- [x] **Sub-segment splitting** (`--split-segments N`, `SegmentType::Continuation`). *Paragraph-boundary text split before tokenizing; hard token-count split fallback.*
- [x] **Bench enhancements** (`--dict-warmup N`, `--with-substrings`). *Full training overhead measured inline.*
- [x] **23-run four-corpus sweep** with full analysis in `bench_results/REPORT.md`.
- [ ] **CLI integration for S3.** *Deferred to v0.7 — requires making each command generic over `<B: StorageBackend>`. Python is the more important interface for cloud users in the meantime.*
- [ ] **GCS / Azure backends.** *Deferred — same trait, same pattern. Add when demand exists.*

---

## v0.7 — Production hardening

**Theme:** close the remaining gaps between "works in benchmarks" and "runs in production."

- [ ] **CLI S3 integration.** *`--storage s3 --s3-bucket NAME` for all CLI commands. Contained refactor.*
- [ ] **Capture-side integration guide.** *A minimal pattern for intercepting OpenAI / Anthropic API calls and routing conversations to Stowken, without making Stowken a proxy. Drop-in middleware examples for FastAPI, LangChain, and LlamaIndex.*
- [ ] **Retention and GC policy.** *`stowken gc --older-than 90d` and configurable ref-count decay. Today GC is manual; production deployments need scheduled cleanup.*
- [ ] **Vault-level metrics.** *Prometheus-compatible `/metrics` endpoint or structured log output for ingest rate, dedup ratio, compression ratio, and retrieval latency. Needed for production observability.*
- [ ] **Dict auto-train policy.** *Periodic retraining on recent segments without disrupting reads. Deferred from v0.3.*

---

## Out of scope (for now)

- **Hosted service.** Library + CLI only.
- **Drop-in OpenAI / Anthropic proxy.** Capture belongs in a separate thin layer, not in Stowken itself.
- **Multi-tenant isolation as a first-class feature.** Single vault per process. Multi-tenant is the operator's job.
- **Compliance modules** (audit log, GDPR deletion verification). Useful thin layers on top of the API, not Stowken's core.

---

## Tracking

This file is the source of truth for sequencing. When checking off an item, move the detail to the corresponding section in `CHANGELOG.md`. When adding new items, slot by *theme* not by priority order; promote the whole milestone if you want to reorder.
