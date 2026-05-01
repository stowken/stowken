# Changelog

For planned future work see [ROADMAP.md](ROADMAP.md). Items move from the roadmap to a version section in this file when they ship.

## 0.6.0 — Unreleased *(in progress)*

### Added — multi-writer S3 backend

Last roadmap item. Stowken can now run against S3 with the v0.1 ref-count race fixed at the protocol layer.

- **`stowken::storage::s3::S3Backend`** behind the new `s3` Cargo feature. Implements the full `StorageBackend` trait including `replace_segment_data`, `list_garbage`, `garbage_collect`, and `storage_size_bytes`.
- **Multi-writer ref-count safety via S3 conditional writes:**
  - `put_object` with `If-None-Match: *` for first-time `.meta` writes (only the first writer succeeds).
  - `put_object` with `If-Match: <etag>` for every ref-count update; on `412 Precondition Failed`, re-read and retry up to `MAX_CAS_RETRIES = 8` times with capped exponential backoff (10ms, 20ms, ..., 320ms).
  - The `s3_concurrent_increments_no_lost_writes` integration test fires 20 concurrent `increment_ref` calls against the same segment and pins `final_ref_count == 21`. The v0.1 implementation would silently lose writes here.
- **`StorageBackend::replace_segment_data`** for S3 — same semantics as filesystem, atomic from the caller's perspective; CAS-updates the `.meta`'s `compressed_size` afterwards so a concurrent ref-count update doesn't race with size update.
- **`S3Backend::with_options(bucket, prefix, endpoint, region)`** — single helper covering AWS, MinIO, localstack, Cloudflare R2, etc. When `endpoint` is set, path-style addressing is forced.
- **Python: `stowken.Stowken.open_s3(bucket, local_db_path, *, prefix, endpoint, region, ...)`**. The metadata index (SQLite) stays local — it's a derived cache rebuildable from on-S3 manifests via `reindex()`. Explicit so multiple readers can each have their own.
- **Integration tests gated by `STOWKEN_S3_TEST_BUCKET` env var.** Run against MinIO / localstack / real S3 with `cargo test -p stowken --features s3 --test integration_tests s3 -- --test-threads 1`. Eight tests cover: roundtrip, idempotent put, concurrent first-writers, concurrent increments, decrement-signals-gc, replace-keeps-ref-count, garbage_collect, full vault store/retrieve/delete.

### Deferred to v0.6.1

- **CLI integration for S3.** Adding `--storage s3 --s3-bucket NAME` requires turning every CLI command into a generic-over-backend function. Contained refactor, but big enough to deserve its own commit. Python is the more important interface for cloud-storage users in the meantime.

### Tests

- 99 core (unchanged) + 49 integration + **8 S3 integration (env-gated)** + 14 CLI + 63 Python = **225 + 8 = 233 green** when MinIO is available; 225 in normal CI runs.
- Clippy clean across both feature configurations (default and `--features s3`).

## 0.5.1 — Unreleased *(in progress)*

### Added — substring compaction & GC

The two pieces v0.5.0 deferred. Together they close the substring story: pre-train segments now get the substring win retroactively, and dead substrings get cleaned up.

- **`stowken substrings compact`** + **`Stowken::compact_substrings`** + Python `vault.compact_substrings()` (sync + async). Walks every unique segment in the index, decompresses to native tokens, tries `0x05` encoding, and replaces the stored bytes when smaller. Atomic per-segment via the new `StorageBackend::replace_segment_data`. Skips `0x04` (delta) and `0x05` (already-substring) frames — they aren't candidates.
- **`stowken substrings gc`** + **`Stowken::gc_substrings`** + Python `vault.gc_substrings()` (sync + async). Walks every segment, decodes any `0x05` frame to collect referenced substring ids, and prunes the registry. Returns `{registry_size_before, registry_size_after, substrings_dropped}`.
- **`SubstringRegistry::retain`** — atomic packed-file rebuild. The kept substrings are re-concatenated into a new `blobs.dat`; the new `index.json` is renamed atomically over the old one. A crash mid-rebuild leaves the old blobs.dat in place; the index.json rename is the commit point.
- **`StorageBackend::replace_segment_data`** — overwrite a segment's compressed bytes without touching ref_count. Atomic-ish (tmp-file rename) on FilesystemBackend; trivial on MemoryBackend. Used by compaction to keep pre-train segments' hash and ref-count stable while updating their on-disk encoding.
- **`MetadataIndex::update_segment_compressed_size`** + **`all_segment_hashes`** — small bookkeeping helpers used by compaction and GC.

### Measured impact

Same drift corpus as v0.5.0, with `compact` after `train`:

| Configuration | Stored | Saved |
|---|---:|---:|
| baseline (varint + zstd) | 4.9 MB | 58.8% |
| `substrings train` (new ingests only) | 2.5 MB | 78.8% (+20 pp) |
| `substrings train` + `compact` (retroactive) | **1.9 MB** | **83.8%** (+25 pp) |

3.0 MB freed by compaction on this corpus — `Compaction: examined 3000 segment(s), rewrote 2000, skipped 0`. Round-trip verified on every rewritten segment.

### Tests

- 99 core (unchanged) + **49** integration (was 44, +5 compact + GC) + **14** CLI (unchanged) + **63** Python (was 60, +3 compact + GC sync/async) = **225 green** (was 217, +8). Clippy clean.

## 0.5.0 — Unreleased *(in progress)*

### Added — token-level shared-substring dedup

The third and last of the three deferred-with-intent features from v0.1 returns. Long boilerplate that recurs across thousands of otherwise-distinct segments — "as a large language model, I cannot...", boilerplate disclaimers, RAG context fragments — is now promoted to shared blobs and replaced in segments by tiny references.

- **New frame format `0x05`**: sequence of two ops, `Inline(tokens)` and `Ref(substring_id)`. Compact varint-encoded body. Decompression streams the ops, emitting inline tokens directly and resolving refs from the registry.
- **`SubstringRegistry`** under `<vault-root>/substrings/{blobs.dat, index.json}`. **Packed-file storage** — substrings concatenate into one `blobs.dat` rather than one file per substring, so a registry of 3,000 typical 16-token substrings takes ~600 KB on disk instead of ~24 MB (the per-file approach paid 4 KiB cluster cost on each tiny file). In-memory cache is populated eagerly at open so the write hot path is a single hashmap lookup.
- **`Stowken::train_substrings(samples, min_occurrences)`** discovers high-frequency MIN_LENGTH-token windows across a sample of stored segments and promotes those clearing the threshold. Subsequent `store()` calls automatically use registered substrings when the resulting `0x05` frame is smaller than the baseline; otherwise it falls back to `0x02`/`0x03`. Strict opt-in win — never net regression.
- **`Stowken::list_substrings`** for inspection.
- **CLI**: `stowken substrings train [--samples N] [--min-occurrences M]` and `stowken substrings list`. Both support `--json`.
- **Python**: `vault.train_substrings(samples=500, min_occurrences=5)`, `vault.list_substrings()`, plus `_async` variants.
- **Compression pipeline** holds an optional `Arc<SubstringRegistry>` and exposes `try_compress_substring_frame(tokens, fallback_len)` — returns `Some(frame)` only when a substring frame ends up smaller than `fallback_len`.

### Measured impact

On the synthetic large-segment drift corpus (1000 convs, 1000-token system prompts + 2000-token RAG with interpolated names):

| Configuration | Stored | Saved vs incoming |
|---|---:|---:|
| Baseline (varint + zstd) | 4.9 MB | 58.8% |
| `--near-dedup 0.85` | 3.5 MB | 71.0% |
| `substrings train` | **2.5 MB** | **78.8%** |

That's a 49% reduction in stored bytes vs. the baseline, on a corpus where the per-segment zstd context can't see across the many shared canonical fragments. Substrings catch what zstd misses.

On OpenAssistant (1000 convs, genuinely unique crowd-sourced content) the cost model correctly promotes **zero** substrings — there's no recurring 16-token window worth promoting — and stored bytes match the baseline exactly. No regression on workloads without substring drift.

### Not yet — deferred to v0.5.x

- **Compaction pass** for retroactive rewrite of pre-train segments. Currently new ingests benefit; old segments are unaffected unless re-stored. The compaction has the largest correctness surface (atomic rewrites + ref-count migration + partial-failure recovery), so it gets its own focused milestone.
- **Variable-length substring extension.** Discovery currently registers exactly `MIN_LENGTH`-token windows. Extending promising windows outward when the surrounding tokens also recur would tighten the registry and increase per-ref savings.
- **GC of unused substrings.** Once a substring's only references are GC'd from the segments backend, the substring blob is currently kept. Add ref-count tracking + cleanup in v0.5.x.

### Tests

- 99 core (was 84, +9 substring registry + +6 substring frame) + **44** integration (was 37, +7 substring) + **14** CLI (unchanged) + **60** Python (was 55, +4 sync substring + +1 async substring) = **217 green** (was 191, +26).
- Clippy clean across the workspace.

## 0.4.0 — Unreleased *(in progress)*

### Added — persistent MinHash near-duplicate detection

The second of the three deferred-with-intent features from v0.1 returns. The v0.1 implementation kept the LSH index in a process-local `HashMap` (lost on every restart) and cached the full token sequence of every canonical (linear memory growth). v0.4 fixes both: the index lives in SQLite, and tokens are fetched from the backend on demand when computing deltas.

- **New frame format `0x04`** = `canonical_hash (32 raw bytes) || streaming delta`. The streaming delta uses a single cursor through the canonical with three opcodes: `Skip(n)`, `Copy(n)`, `Insert([tokens])`. Compact varint-encoded.
- **`crate::near_dedup` module**: `MinHashSignature` (128 × u64, with byte-roundtrip for SQLite BLOB storage), `band_hash` for LSH bucketing, `compute_delta` / `apply_delta` / `encode_delta` / `decode_delta` for the streaming delta format.
- **Two new SQLite tables** in `metadata.db`:
  - `near_dedup_signatures(segment_hash PRIMARY KEY, signature BLOB)` — 1024-byte MinHash per segment.
  - `near_dedup_bands(band_index, band_hash, segment_hash)` — LSH bucket index, primary keyed for idempotent inserts.
- **`StowkenConfig::near_dedup_threshold: Option<f64>`** — opt-in. `None` (default) disables; `Some(0.85)` is conservative.
- **`Stowken::find_near_duplicates`** + **`near_dedup_index_size`** — read-only analytics over the LSH index.
- **`stowken find-near-duplicates [--threshold] [--min-refs] [--limit]`** — surfaces drift clusters in human or `--json` form.
- **Global `--near-dedup <threshold>` CLI flag** so `store` invocations can opt in without rebuilding.
- **Python:** `Stowken(near_dedup=0.85)` and `Stowken.open(path, near_dedup=0.85)` constructors; `find_near_duplicates(...)` and `near_dedup_index_size()` plus `_async` variants.
- **`reindex` now rebuilds the LSH index too** by walking each non-delta segment, decompressing, recomputing signature + bands, and upserting. Delta segments (0x04) are skipped — they aren't canonicals and shouldn't seed the LSH.
- **`CompressionPipeline::compress_delta` / `frame_canonical_hash`** for use by the vault layer.
- **`CanonicalResolver` trait** in compression — pluggable callback for resolving canonical tokens during 0x04 decompression. The vault uses an async path that fetches from the backend; the sync `decompress` returns `NoCanonicalResolver` for 0x04 frames since resolving requires I/O.

### Why the v0.1 in-memory bug is structurally fixed

In v0.1 the `NearDedupIndex` was a process-local struct. After a restart, the `HashMap<SegmentHash, MinHashSignature>` was empty, so newly-arriving variants of pre-restart canonicals would be stored as full segments — silent regression in compression. The `tokens: HashMap<SegmentHash, Vec<Token>>` cache also grew unboundedly with corpus size.

In v0.4 the LSH state is two SQLite tables. Restart is a no-op. The `tokens` cache is gone entirely — when computing a delta, the canonical is fetched from the backend on demand and dropped after the delta is built. Memory grows with the working set of one in-flight ingest, not with corpus size.

Pinned by:
- `lsh_index_survives_vault_reopen` — store canonical, drop vault, reopen, store near-variant, confirm it goes through the delta path.
- `reindex_rebuilds_lsh_after_db_deletion` — wipe `metadata.db`, reindex, confirm subsequent near-variants are detected.
- `delta_only_used_when_actually_smaller` — for tiny segments where the 33-byte delta header exceeds a fresh full compression, fall back to full storage rather than paying cross-ref overhead for negative gain.

### Tests

- 84 core (was 71, +13 near_dedup primitives) + **37** integration (was 30, +7 near-dedup) + **14** CLI (unchanged) + **55** Python (was 49, +4 sync near-dup, +2 async near-dup) = **191 green** (was 165, +26).
- Clippy clean across the workspace.

### Measured impact

Bench results across three corpora make the workload-dependence visible:

| Corpus | Without `--near-dedup` | With `--near-dedup 0.85` | Verdict |
|---|---:|---:|---|
| OpenAssistant English (3,482 convs, unique content) | 54.2% | 54.2% (no detectable drift) | Don't enable — 5× ingest cost, zero gain. |
| Synthetic small-segment drift (1,000 convs, 100t system + 200t RAG) | 57.1% | 57.1% net (delta header ~ savings) | Marginal — 54× ingest cost for ~5% storage win. |
| Synthetic large-segment drift (1,000 convs, 1000t system + 2000t RAG) | 58.8% | **71.0%** | Worth it — 60× ingest cost for ~30% storage reduction. |

Near-dedup pays off when segments are large (delta header is small relative), drift is small (few token changes per segment), and ingest latency tolerates an LSH probe per segment. It's off by default; turn it on per workload after measuring.

### Bench output cleanup

The bench command's storage-layer report now distinguishes "Incoming (u32)" — what the user actually sent in, the apples-to-apples baseline regardless of whether near-dedup inflates the reference graph — from the dedup/zstd savings. Previously the headline used `naive_bytes` from the metadata index, which double-counted near-dedup's canonical cross-refs and made before/after comparisons confusing.

### Synthetic drift fixture

`scripts/synth_drift_corpus.py` generates production-shaped corpora with configurable segment sizes (`--system-tokens`, `--rag-tokens`) and a fixed-seed PRNG so bench numbers are reproducible.

## 0.3.0 — Unreleased *(in progress)*

### Added — frame-versioned zstd dictionaries

The first of the three deferred-with-intent features from v0.1 returns, this time with the rotation safety bug fixed at the format layer. Multiple dictionaries coexist; each compressed frame names the dict it was written with; rotation is non-destructive.

- **New frame format `0x03`** = varint + zstd-with-dictionary. Body: `dict_id (u32 LE) || zstd(varint(tokens), dict)`. Dispatches per-frame at decompression so old frames stay readable as long as their dict is in the registry.
- **`DictRegistry`** — per-vault registry for zstd compression dictionaries. Filesystem-backed vaults persist under `<vault-root>/dictionaries/{id}.zstd` plus a `{id}.json` info sidecar and an `active` marker file. Memory-backed vaults keep the registry in RAM only.
- **`Stowken::train_dictionary`**, **`activate_dictionary`**, **`train_and_activate_dictionary`**, **`list_dictionaries`**. Training pulls samples from the most recent conversations and runs `zstd::dict::from_samples` on a blocking thread.
- **`stowken dict train [--samples N] [--activate]`**, **`stowken dict list`**, **`stowken dict activate <id>`**. Each supports `--json` for scripting.
- **Python:** `vault.train_dictionary(samples=200, activate=False)`, `activate_dictionary(id)`, `list_dictionaries()` plus `_async` variants for all three.
- **`DictInfo`** type with `id`, `created_at`, `sample_count`, `size_bytes`, `is_active`.
- **`StowkenError::Dict`** and **`InsufficientTrainingSamples`** error variants.

### How the v0.1 data-loss bug is fixed

In v0.1 the active dictionary was held by-value in the compression pipeline; applying a new dictionary replaced the old one in memory and any frame compressed against the previous dict became unreadable on the next `retrieve()` call.

In v0.3 the compression pipeline holds an `Arc<DictRegistry>`. New compressions write the active dict's ID into the frame; decompression reads the ID and looks up the right dict in the registry. Activating a new dictionary is purely additive — old dictionaries stay registered and continue to serve their frames. Rotation safety is now a property of the frame format, not of the runtime configuration.

The integration test `segments_compressed_with_old_dict_remain_readable_after_rotation` pins this contract: train dict A → store conv X → train dict B → activate B → store conv Y → confirm both X and Y round-trip exactly. v0.1 lost X here; v0.3 doesn't.

### Tests

- 71 core (was 60, +11 dict registry + dict frame) + **30** integration (was 25, +5 rotation safety) + **14** CLI (was 13, +1 dict round-trip) + **49** Python (was 40, +7 sync dict + +2 async dict) = **165 green** (was 139, +26).
- Clippy clean across the workspace.

## 0.2.0 — Unreleased *(in progress)*

### Added

- **`stowken bench <input-dir>`** — read-only benchmark on a directory of `.json` (one conversation per file) or `.jsonl` (one per line) inputs. Reports the three-stage savings breakdown (naive → dedup → zstd) plus per-segment-type detail. Doesn't touch the configured data directory. Both human-readable tables and `--json` output.
- **`scripts/fetch_openassistant.py`** — converts the HuggingFace `OpenAssistant/oasst1` dataset to Stowken JSONL by walking the highest-rank path through each message tree.
- **`MetadataIndex::dedup_only_bytes`** and **`Stowken::dedup_only_bytes`** — exposes the "after dedup, before compression" size needed to separate dedup savings from zstd savings.
- **`stowken reindex`** + `Stowken::reindex` + Python `vault.reindex()` — rebuild the SQLite metadata index from on-disk manifests. Closes the recovery loop: deleting `metadata.db` or restoring from a backup that didn't include it no longer means losing all analytics. Idempotent. Reports `(conversations_indexed, unique_segments_indexed, segments_missing)`. Replays manifests through the same batch path that `store()` uses so the resulting index is byte-identical to one built incrementally.
- **Async Python API.** Every I/O method on `Stowken` now has an `_async` suffix variant that returns an awaitable: `store_async`, `retrieve_async`, `delete_async`, `stats_async`, `segment_stats_async`, `gc_async`, `gc_candidates_async`, `reindex_async`. Safe inside asyncio event loops (FastAPI, async workers). Sync and async methods share the same vault instance and the same Tokio runtime; they can be mixed freely on one vault.
- **`MetadataIndex::clear`** — wipe every index table in one transaction. Used by reindex.
- **`ReindexStats`** type.
- **README headline numbers** replaced with measured benchmarks: 54.2% on full English OpenAssistant (zstd-dominated, no shared prompts) and 99.3% on the synthetic shared-prompt corpus (dedup-dominated). Spectrum framing instead of a single marketing number.

### Tests

- 60 core + **25** integration (was 23, +2 reindex) + **13** CLI (was 10, +2 bench, +1 reindex) + **40** Python (was 28, +2 reindex, +10 async — including a 50-task concurrent `store_async` test) = **139 green**.

## 0.1.0 — Unreleased

First focused release. The project was previously a 5-crate workspace with a wide enterprise feature surface (cloud backends, near-dedup via MinHash LSH, AES-256-GCM encryption, multi-tenant isolation, license validation, compliance modules, replication / distributed-lock abstractions, parquet export). This release drops everything that wasn't truly working and focuses on a clean local-vault product.

### Cut

- `stowken-enterprise` crate (entire crate removed). Anything that lived there is on a branch in git history if needed: near-dedup, encryption, multi-tenant, S3/GCS/Azure backends, parquet export, compliance, replication, license JWT validation, advanced analytics.
- `stowken-node` (Node.js binding). Deferred. Python covers the analytics audience for v1.
- Auto-train zstd dictionaries. The previous implementation silently corrupted historical data on rotation (each new dictionary replaced the previous one without preserving the ID). Removed pending a versioned dictionary scheme.
- Manual `train_dictionary` / `apply_dictionary` API. Removed for the same reason; bring back in v0.2 with frame-level dictionary IDs.
- Tenant-scoped hashing. The previous prefix-based scheme didn't deliver real isolation; cleaner to drop than misrepresent.
- Stub methods on `StorageBackend`: `get_stats`, `get_segment_type_stats`, `query_analytics`, `list_unique_system_prompts`, `find_conversations_by_segment`, `has_segments`. These were dead-coded by the vault (analytics live on `MetadataIndex`) and returned silent zeros on `FilesystemBackend` and S3.
- All cloud backend implementations (S3, GCS, Azure). The S3 backend had an unsolved concurrency bug in ref-count updates and was unused by the bindings; the others were sketches.

### Added

- **Frame format versioning.** Every compressed segment now starts with a 1-byte version tag (`0x01` = varint, `0x02` = varint+zstd). Future codecs can be introduced without breaking historical data.
- **Manifest schema versioning.** `ConversationManifest.schema_version` (default 1, defaults backfilled on read). Retrieval rejects newer-version manifests with a clear error.
- **`stowken audit-system-prompts`** — surface unique system prompts by reference count to detect prompt drift.
- **`stowken cost --by application,model[,segment_type,day]`** — token attribution across arbitrary dimensions.
- **`stowken find-duplicates --threshold N`** — list heavily-referenced segments with computed `wasted_bytes` (what dedup is saving on each).
- **Real `gc --dry-run`.** Lists actual zero-ref candidates instead of printing aggregate stats.
- **`StorageBackend::list_garbage`** — backend-level support for the dry-run command. Default impl returns empty.
- **`Stowken::find_duplicates`** and `MetadataIndex::find_duplicates`.
- **`DuplicateSegment` type** with `ref_count`, `token_count`, `wasted_bytes`.

### Changed

- **Python binding overhaul.**
  - `retrieve()` now returns the actual `u32` token IDs in `seg["tokens"]` (previously returned only metadata).
  - New `Stowken.open(path)` classmethod for filesystem-backed persistence (previously memory-only).
  - Singleton Tokio runtime via `OnceLock` (previously a fresh runtime was created per call).
  - New `gc_candidates()` method.
- **`StowkenConfig` slimmed** to a single field: `enable_compression`. The dictionary, tenant, and auto-train fields are gone.
- **`StorageBackend` trait slimmed** to pure storage operations. Analytics live exclusively on `MetadataIndex`.
- **`md5_tokens` → `hash_pair_key`** in the export module (the original name lied — it never used MD5).
- **README rewritten** to match shipped reality. The "55–85% with a trained dictionary" range was replaced with measured numbers from the `compression_benchmark` example. License framing simplified to single Apache-2.0.

### Tests

- 60 core unit tests + 23 integration tests + 10 CLI tests + 28 Python tests = **121 green**, no skipped, no ignored.
- New compression module tests cover frame-version dispatch and cross-pipeline decompression.
- New Python tests cover token roundtrip, filesystem persistence across instances, dedup-after-reopen, and `gc_candidates`.

### Removed artifacts

- Stale `tokenvault.darwin-x64.node` and `tokenvault.abi3.so` files (project was renamed mid-build; legacy binaries persisted in source tree).
- Empty placeholder directories: `docs/`, `benches/`, `dictionaries/`, `tests/`.
