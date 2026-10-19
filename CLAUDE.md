# Stowken

Storage library for LLM conversations: content-addressed token segments (identical prompts stored once), compression, a SQLite index for querying traffic, plus semantic search and analytics. Formerly called **TokenVault** (original spec: `docs/origin/`). Published as a Rust crate and a Python package under the `stowken` GitHub org (public).

## Stack
Rust workspace + Python bindings (PyO3 / maturin).
- `crates/stowken/` — core library (`segmenter`, `tokenizer`, `dedup/`, `near_dedup*`, `compression/`, `storage/`, `index/`, `semantic`, `clustering`, `export/`, `vault.rs`)
- `crates/stowken-cli/` — CLI · `crates/stowken-python/` — Python bindings + `python/tests/`
- Docs: `docs/` (getting started, semantic search, analytics, Python cookbook, architecture), `ROADMAP.md`, `CHANGELOG.md`, `CONTRIBUTING.md`

## Commands
```sh
cargo build --workspace
cargo test --workspace            # verified 2026-10-18: 194 tests pass
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
# Python bindings (see CONTRIBUTING.md):
PYO3_PYTHON=python3.13 maturin build --release && pytest crates/stowken-python/python/tests/
```

## Known state (2026-10-18)
- On Rust 1.99 (latest stable), `cargo fmt --check` reports differences and `clippy -D warnings` fails (15 errors, from the `async_trait` macro). CI uses `stable` too (`.github/workflows/ci.yml`), so **CI will likely fail the same way** until these are fixed. Tests themselves pass. Ask before making the fixes.
- Python bindings not yet built on this Mac (needs `maturin` + Python 3.13).

## Rules
- CONTRIBUTING.md requires clippy-clean code and a CHANGELOG entry for user-facing changes.
- Public repo: never commit data from `data/` that contains real conversations, or any keys.
