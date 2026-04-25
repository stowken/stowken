# Contributing to Stowken

Thank you for your interest in contributing! This document covers the open-source
`stowken` crate. The `stowken-enterprise` crate is proprietary and not
open to external contributions.

## Development setup

```bash
# Install Rust (stable)
curl https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/stowken/stowken
cd stowken
cargo build --workspace

# Run tests
cargo test -p stowken

# Run clippy (must be clean with -D warnings)
cargo clippy -p stowken -- -D warnings
```

## Pull request guidelines

- **One logical change per PR.** Bug fixes, features, and refactors should be
  separate PRs.
- **Tests required.** New functionality must have unit or integration tests.
- **Clippy clean.** `cargo clippy -p stowken -- -D warnings` must pass.
- **No new dependencies** without discussion. The open-source crate has a
  carefully curated dependency footprint.
- **No unsafe code** without a detailed justification and a comment explaining
  the invariants that make it safe.

## Commit style

```
<type>(<scope>): <short summary>

Types: feat, fix, refactor, test, docs, chore
Scope: compression, dedup, storage, vault, export, index
```

Examples:
- `feat(storage): add S3 presigned URL support`
- `fix(varint): handle u32::MAX correctly in 5-byte escape path`
- `test(dedup): add exact-dedup collision test`

## What belongs in the open-source crate

The `stowken` crate (Apache 2.0) includes:

- Core types (`types.rs`)
- Varint encoding and zstd compression
- Exact deduplication (SHA-256)
- `StorageBackend` trait + `MemoryBackend` + `FilesystemBackend`
- Conversation segmenter
- SQLite metadata index
- `Stowken` orchestrator
- JSONL training export
- Analytics queries

The following are enterprise-only and not accepted as open-source contributions:
- Cloud backends (S3, GCS, Azure)
- Near-dedup (MinHash LSH)
- Encryption at rest
- Multi-tenant isolation
- GDPR compliance tooling
- Advanced analytics
- Parquet export

## Reporting bugs

Open an issue on GitHub with:
1. Your Rust version (`rustc --version`)
2. Your OS and architecture
3. A minimal reproducible example
4. The full error output

## Security vulnerabilities

Please do **not** open a public issue for security vulnerabilities. Email
security@stowken.dev instead.
