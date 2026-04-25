# Contributing to Stowken

Stowken is a single Apache-2.0 workspace. All three crates (`stowken`, `stowken-cli`, `stowken-python`) accept contributions.

## Development setup

```bash
# Install Rust (stable)
curl https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/stowken/stowken
cd stowken
cargo build --workspace

# Run tests
cargo test --workspace

# Run clippy (should be clean)
cargo clippy --workspace -- -D warnings

# Python bindings (optional)
cd crates/stowken-python
PYO3_PYTHON=python3.13 maturin build --release
pip install --force-reinstall ../../target/wheels/stowken-*.whl
pytest python/tests/
```

## Pull request guidelines

- **One logical change per PR.** Bug fixes, features, and refactors should be separate.
- **Tests required.** New functionality must have unit or integration coverage.
- **Clippy clean.** `cargo clippy --workspace -- -D warnings` must pass.
- **No new dependencies** without discussion. The dependency footprint is intentionally small.
- **No unsafe code** without a detailed justification and a comment explaining the invariants that make it safe.

## Commit style

```
<type>(<scope>): <short summary>

Types: feat, fix, refactor, test, docs, chore
Scope: compression, dedup, storage, vault, export, index, cli, python
```

Examples:
- `fix(varint): handle u32::MAX correctly in 5-byte escape path`
- `test(dedup): add exact-dedup collision test`
- `feat(cli): add --since flag to find-duplicates`

## On-disk format changes

Stowken's frame format and manifest schema are versioned. Any change that alters how segments or manifests are written on disk must:

1. Bump the version (new frame byte, or `MANIFEST_SCHEMA_VERSION + 1`).
2. Keep the old version readable.
3. Add a roundtrip test that covers both old-format and new-format data.

## Reporting bugs

Open an issue on GitHub with:
1. Your Rust version (`rustc --version`)
2. Your OS and architecture
3. A minimal reproducible example
4. The full error output

## Security vulnerabilities

Please do not open a public issue for security vulnerabilities. Email security@stowken.dev instead.
