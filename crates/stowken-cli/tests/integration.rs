//! CLI integration tests.
//!
//! These spawn the `stowken` binary against a temporary directory and
//! assert on stdout/stderr/exit-codes.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn bin() -> PathBuf {
    // `cargo test` sets CARGO_BIN_EXE_stowken when the bin target is listed.
    // Fall back to building manually in case the env var isn't set.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_stowken") {
        return PathBuf::from(p);
    }
    // Build in release mode so tests are fast enough.
    let output = Command::new(env!("CARGO"))
        .args(["build", "-p", "stowken-cli", "--bin", "stowken"])
        .output()
        .expect("cargo build failed");
    assert!(output.status.success(), "build failed: {:?}", output);

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.parent().unwrap().parent().unwrap();
    workspace
        .join("target")
        .join("debug")
        .join("stowken")
}

fn run(data_dir: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .output()
        .expect("failed to run stowken binary")
}

fn run_stdin(data_dir: &Path, args: &[&str], stdin_data: &str) -> Output {
    use std::process::Stdio;
    let mut child = Command::new(bin())
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn stowken");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin_data.as_bytes())
        .expect("write to stdin");
    child.wait_with_output().expect("wait failed")
}

fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn sample_conversation(id: &str) -> String {
    serde_json::json!({
        "id": id,
        "model": "gpt-4",
        "tokenizer": "cl100k_base",
        "messages": [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"}
        ]
    })
    .to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn store_and_retrieve_roundtrip() {
    let dir = temp_dir();
    let conv = sample_conversation("test-conv-1");

    // Store via stdin.
    let out = run_stdin(dir.path(), &["store", "-"], &conv);
    assert!(out.status.success(), "store failed: {}", String::from_utf8_lossy(&out.stderr));

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("test-conv-1"), "id not in output: {stdout}");

    // Retrieve by id (--json for parseable output).
    let out = run(dir.path(), &["retrieve", "test-conv-1", "--json"]);
    assert!(out.status.success(), "retrieve failed: {}", String::from_utf8_lossy(&out.stderr));

    let retrieved: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("retrieve output not JSON");
    assert_eq!(retrieved["id"].as_str(), Some("test-conv-1"));
    assert_eq!(retrieved["model"].as_str(), Some("gpt-4"));
}

#[test]
fn store_json_flag_emits_valid_json() {
    let dir = temp_dir();
    let conv = sample_conversation("json-test");

    let out = run_stdin(dir.path(), &["store", "-", "--json"], &conv);
    assert!(out.status.success(), "store --json failed: {}", String::from_utf8_lossy(&out.stderr));

    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("output is not valid JSON");
    assert_eq!(parsed["id"].as_str(), Some("json-test"));
    assert!(parsed["new_segments"].as_u64().unwrap_or(0) > 0);
}

#[test]
fn dedup_second_store_reports_deduped_segments() {
    let dir = temp_dir();
    let conv = sample_conversation("dedup-test");

    let out1 = run_stdin(dir.path(), &["store", "-", "--json"], &conv);
    assert!(out1.status.success());
    let r1: serde_json::Value = serde_json::from_slice(&out1.stdout).unwrap();
    let new1 = r1["new_segments"].as_u64().unwrap();
    assert!(new1 > 0, "first store should create new segments");

    // Store the exact same conversation again with a different id.
    let conv2 = serde_json::json!({
        "id": "dedup-test-2",
        "model": "gpt-4",
        "tokenizer": "cl100k_base",
        "messages": [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"}
        ]
    })
    .to_string();

    let out2 = run_stdin(dir.path(), &["store", "-", "--json"], &conv2);
    assert!(out2.status.success());
    let r2: serde_json::Value = serde_json::from_slice(&out2.stdout).unwrap();
    assert!(
        r2["deduped_segments"].as_u64().unwrap_or(0) > 0,
        "second store should report deduped segments"
    );
}

#[test]
fn stats_returns_nonzero_after_store() {
    let dir = temp_dir();
    let conv = sample_conversation("stats-test");

    run_stdin(dir.path(), &["store", "-"], &conv);

    let out = run(dir.path(), &["stats", "--json"]);
    assert!(out.status.success(), "stats failed: {}", String::from_utf8_lossy(&out.stderr));

    let stats: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(stats["total_conversations"].as_u64().unwrap_or(0) >= 1);
    assert!(stats["unique_segments"].as_u64().unwrap_or(0) >= 1);
}

#[test]
fn delete_removes_conversation() {
    let dir = temp_dir();
    let conv = sample_conversation("delete-test");

    run_stdin(dir.path(), &["store", "-"], &conv);

    // Delete with --yes to skip interactive confirmation.
    let out = run(dir.path(), &["delete", "delete-test", "--yes"]);
    assert!(out.status.success(), "delete failed: {}", String::from_utf8_lossy(&out.stderr));

    // Retrieve should now fail.
    let out = run(dir.path(), &["retrieve", "delete-test"]);
    assert!(!out.status.success(), "retrieve after delete should fail");
}

#[test]
fn gc_runs_without_error() {
    let dir = temp_dir();
    // Store then delete to leave dangling segments.
    let conv = sample_conversation("gc-test");
    run_stdin(dir.path(), &["store", "-"], &conv);
    run(dir.path(), &["delete", "gc-test", "--yes"]);

    let out = run(dir.path(), &["gc"]);
    assert!(out.status.success(), "gc failed: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn segment_stats_lists_segment_types() {
    let dir = temp_dir();
    let conv = sample_conversation("seg-stats-test");
    run_stdin(dir.path(), &["store", "-"], &conv);

    let out = run(dir.path(), &["segment-stats", "--json"]);
    assert!(out.status.success(), "segment-stats failed: {}", String::from_utf8_lossy(&out.stderr));

    let arr: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert!(!arr.is_empty(), "expected at least one segment type");
    let types: Vec<_> = arr.iter().filter_map(|e| e["segment_type"].as_str()).collect();
    assert!(
        types.iter().any(|t| t.contains("system") || t.contains("user")),
        "expected system/user segment types, got {types:?}"
    );
}

#[test]
fn data_dir_subcommand_prints_path() {
    let dir = temp_dir();
    let out = run(dir.path(), &["data-dir"]);
    assert!(out.status.success());
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    assert_eq!(printed, dir.path().to_str().unwrap());
}

#[test]
fn store_from_file() {
    let dir = temp_dir();
    let conv = sample_conversation("file-test");

    let tmp_file = dir.path().join("conv.json");
    std::fs::write(&tmp_file, conv).unwrap();

    let out = run(
        dir.path(),
        &["store", tmp_file.to_str().unwrap(), "--json"],
    );
    assert!(out.status.success(), "store from file failed: {}", String::from_utf8_lossy(&out.stderr));
    let r: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(r["id"].as_str(), Some("file-test"));
}

#[test]
fn bench_runs_on_jsonl_directory() {
    let dir = temp_dir();
    let bench_dir = tempfile::tempdir().unwrap();

    // Two conversations sharing a system prompt — exercises the dedup path.
    let jsonl_path = bench_dir.path().join("corpus.jsonl");
    let mut file = std::fs::File::create(&jsonl_path).unwrap();
    for id in ["bench-a", "bench-b"] {
        let conv = serde_json::json!({
            "id": id,
            "model": "gpt-4",
            "tokenizer": "cl100k_base",
            "messages": [
                {"role": "system", "content": "Shared system prompt for benchmarks."},
                {"role": "user", "content": format!("Question {id}")},
                {"role": "assistant", "content": format!("Answer for {id}")}
            ]
        });
        writeln!(file, "{}", conv).unwrap();
    }
    drop(file);

    let out = run(
        dir.path(),
        &["--json", "bench", bench_dir.path().to_str().unwrap()],
    );
    assert!(
        out.status.success(),
        "bench failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&out.stdout)
        .expect("bench --json should emit valid JSON");

    assert_eq!(result["ingested"], 2, "expected 2 conversations ingested");
    assert_eq!(result["failures"], 0, "expected 0 failures");

    // Shared system prompt → at least one segment with refs > unique.
    assert!(
        result["dedup_ratio"].as_f64().unwrap() > 0.0,
        "expected dedup ratio > 0 with a shared system prompt"
    );

    // The three storage layers are present and ordered correctly.
    let naive = result["naive_bytes"].as_u64().unwrap();
    let dedup = result["dedup_only_bytes"].as_u64().unwrap();
    let storage = result["storage_bytes"].as_u64().unwrap();
    assert!(dedup < naive, "dedup ({dedup}) should be smaller than naive ({naive})");
    // storage may be larger than dedup on tiny corpora due to zstd frame
    // overhead — that's expected and the report shows it honestly. Only
    // sanity-check that storage is positive and not absurd.
    assert!(storage > 0 && storage < naive * 2);
}

#[test]
fn bench_errors_on_empty_directory() {
    let dir = temp_dir();
    let empty = tempfile::tempdir().unwrap();
    let out = run(dir.path(), &["bench", empty.path().to_str().unwrap()]);
    assert!(!out.status.success(), "bench should fail on empty directory");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no conversations found"),
        "expected 'no conversations found' error, got: {stderr}"
    );
}

#[test]
fn export_jsonl_format() {
    let dir = temp_dir();
    // Store two conversations so export has something to emit.
    for id in ["exp-1", "exp-2"] {
        let conv = serde_json::json!({
            "id": id,
            "model": "gpt-4",
            "tokenizer": "cl100k_base",
            "messages": [
                {"role": "system",    "content": "System prompt."},
                {"role": "user",      "content": "User message."},
                {"role": "assistant", "content": "Assistant reply."}
            ]
        })
        .to_string();
        run_stdin(dir.path(), &["store", "-"], &conv);
    }

    let out = run(dir.path(), &["export", "--format", "jsonl"]);
    assert!(out.status.success(), "export failed: {}", String::from_utf8_lossy(&out.stderr));

    let stdout = String::from_utf8_lossy(&out.stdout);
    // Each line in JSONL output should be valid JSON.
    let lines: Vec<_> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty(), "export produced no output");
    for line in &lines {
        let _: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("invalid JSONL line: {e}\n{line}"));
    }
}
