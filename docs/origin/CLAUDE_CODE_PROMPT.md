# Claude Code Prompt — TokenVault Implementation

You are building **TokenVault**, a Rust library for compressed storage and retrieval of LLM token sequences. The complete implementation specification is in `TOKENVAULT_IMPLEMENTATION.md` in the project root. Read that file thoroughly before writing any code — it contains every type definition, trait signature, algorithm, schema, and dependency you need.

## What to build

A Rust workspace with three crates:
- `tokenvault-core` — all logic (compression, dedup, segmentation, storage, indexing, export)
- `tokenvault-python` — Python bindings via PyO3 + maturin
- `tokenvault-node` — Node.js bindings via napi-rs

## How to build it

Follow the **Implementation Order** in Section 17 of the spec exactly. Build each module, write tests for it, and confirm they pass before moving to the next:

1. `types.rs` — all data types
2. `compression/varint.rs` — variable-width token encoding with roundtrip tests
3. `compression/dictionary.rs` — zstd wrapper
4. `dedup/exact.rs` — SHA-256 segment hashing
5. `storage/backend.rs` — trait definition
6. `storage/memory.rs` — in-memory backend
7. `segmenter.rs` — conversation decomposition
8. `index/metadata.rs` — SQLite metadata store
9. `vault.rs` — main orchestrator (store + retrieve pipeline)
10. Integration tests — roundtrip, dedup, compression
11. `storage/filesystem.rs` — filesystem backend
12. `index/analytics.rs` — analytics queries
13. `export/training.rs` — JSONL export
14. `dedup/near.rs` — MinHash near-dedup (behind `near-dedup` feature flag)
15. `export/parquet.rs` — Parquet export (behind `export-parquet` feature flag)
16. `storage/s3.rs` — S3 backend (behind `s3` feature flag)
17. Python bindings
18. Node.js bindings
19. CLI binary
20. Criterion benchmarks
21. Cloud backends (GCS, Azure)

After completing each step, run `cargo test` and `cargo clippy` before proceeding. Fix any warnings or failures before moving on.

## Key constraints

- **No tokenizer dependency.** TokenVault does not tokenize text. It stores and retrieves `Vec<u32>` token arrays. Users provide pre-tokenized data or supply a `TokenizerAdapter` trait implementation.
- **Async storage, sync SQLite.** The `StorageBackend` trait is async. The SQLite metadata index uses `rusqlite` synchronously. Bridge with `spawn_blocking` in the vault orchestrator.
- **Tenant-scoped dedup.** When `tenant_id` is set, prefix segment hashes with the tenant ID before storage.
- **Feature flags for optional backends.** S3, GCS, Azure, near-dedup, and Parquet export are behind cargo feature flags. The default feature set is `["filesystem"]`.
- **Segments are immutable.** Once stored, segment content never changes. Edits create new segments and update the manifest.
- **Test every layer.** Each module should have unit tests. Integration tests go in `tests/integration/`. Use the fixture data patterns described in Section 14 of the spec.

## Quality bar

- All public types and methods have doc comments
- `cargo clippy -- -D warnings` passes
- `cargo test` passes with no failures
- Error types use `thiserror` with descriptive messages
- No `unwrap()` in library code — propagate errors with `?`
- Use `tracing` for structured logging at appropriate levels
