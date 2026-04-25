# Changelog

For planned future work see [ROADMAP.md](ROADMAP.md). Items move from the roadmap to a version section in this file when they ship.

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
