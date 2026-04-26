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
fn dict_train_list_and_activate_round_trip() {
    let dir = temp_dir();

    // Seed enough segments to train a dictionary.
    for i in 0..40u32 {
        let conv = serde_json::json!({
            "id": format!("dict-seed-{i}"),
            "model": "gpt-4",
            "tokenizer": "cl100k_base",
            "messages": [
                {
                    "role": "system",
                    "tokens": (0..100u32).map(|t| (t * 13 + i * 7) % 50_000).collect::<Vec<_>>(),
                },
                {
                    "role": "user",
                    "tokens": (0..30u32).map(|t| (t * 5 + i * 3) % 50_000).collect::<Vec<_>>(),
                },
            ]
        })
        .to_string();
        let out = run_stdin(dir.path(), &["store", "-"], &conv);
        assert!(out.status.success(), "seed store failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    // Empty list to start.
    let list_out = run(dir.path(), &["--json", "dict", "list"]);
    assert!(list_out.status.success());
    let list: serde_json::Value = serde_json::from_slice(&list_out.stdout).unwrap();
    assert_eq!(list.as_array().map(|v| v.len()), Some(0));

    // Train + activate in one shot.
    let train_out = run(dir.path(), &["--json", "dict", "train", "--samples", "20", "--activate"]);
    assert!(train_out.status.success(), "train failed: {}", String::from_utf8_lossy(&train_out.stderr));
    let info: serde_json::Value = serde_json::from_slice(&train_out.stdout).unwrap();
    let id = info["id"].as_u64().expect("id should be u64");
    assert!(info["is_active"].as_bool().unwrap());

    // List should show one active dict.
    let list2: serde_json::Value =
        serde_json::from_slice(&run(dir.path(), &["--json", "dict", "list"]).stdout).unwrap();
    let arr = list2.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["id"].as_u64(), Some(id));
    assert!(arr[0]["is_active"].as_bool().unwrap());

    // Train a second dict (without --activate) and confirm the first stays active.
    // (Need to seed more data first or sampling will overlap.)
    for i in 100..140u32 {
        let conv = serde_json::json!({
            "id": format!("dict-seed-{i}"),
            "model": "gpt-4",
            "tokenizer": "cl100k_base",
            "messages": [
                {
                    "role": "system",
                    "tokens": (0..100u32).map(|t| (t * 17 + i * 11) % 50_000).collect::<Vec<_>>(),
                },
                {
                    "role": "user",
                    "tokens": (0..30u32).map(|t| (t * 7 + i * 5) % 50_000).collect::<Vec<_>>(),
                },
            ]
        })
        .to_string();
        run_stdin(dir.path(), &["store", "-"], &conv);
    }

    let train2 = run(dir.path(), &["--json", "dict", "train", "--samples", "20"]);
    assert!(train2.status.success());
    let info2: serde_json::Value = serde_json::from_slice(&train2.stdout).unwrap();
    let id2 = info2["id"].as_u64().unwrap();
    assert_ne!(id, id2, "different training corpora should produce different ids");
    assert!(!info2["is_active"].as_bool().unwrap(), "second dict trained without --activate");

    // Now activate the second one.
    let act_out = run(dir.path(), &["dict", "activate", &id2.to_string()]);
    assert!(act_out.status.success(), "activate failed: {}", String::from_utf8_lossy(&act_out.stderr));

    // Verify activation flipped.
    let list3: serde_json::Value =
        serde_json::from_slice(&run(dir.path(), &["--json", "dict", "list"]).stdout).unwrap();
    let arr3 = list3.as_array().unwrap();
    assert_eq!(arr3.len(), 2);
    let active_ids: Vec<u64> = arr3
        .iter()
        .filter(|d| d["is_active"].as_bool().unwrap())
        .map(|d| d["id"].as_u64().unwrap())
        .collect();
    assert_eq!(active_ids, vec![id2], "exactly one dict active, must be id2");

    // Pre-dict retrievals still work after activations + rotations.
    let r = run(dir.path(), &["--json", "retrieve", "dict-seed-0"]);
    assert!(r.status.success(), "pre-dict retrieve failed: {}", String::from_utf8_lossy(&r.stderr));
}

#[test]
fn reindex_restores_stats_after_db_deletion() {
    let dir = temp_dir();

    // Seed two conversations sharing a system prompt.
    for id in ["reidx-a", "reidx-b"] {
        let conv = serde_json::json!({
            "id": id,
            "model": "gpt-4",
            "tokenizer": "cl100k_base",
            "messages": [
                {"role": "system",    "content": "Shared reindex prompt."},
                {"role": "user",      "content": format!("Question for {id}")},
                {"role": "assistant", "content": format!("Answer for {id}")}
            ]
        })
        .to_string();
        let out = run_stdin(dir.path(), &["store", "-"], &conv);
        assert!(out.status.success(), "seed store failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let stats_before = run(dir.path(), &["--json", "stats"]);
    assert!(stats_before.status.success());
    let before: serde_json::Value = serde_json::from_slice(&stats_before.stdout).unwrap();

    // Nuke the index. Manifests + segment files survive.
    std::fs::remove_file(dir.path().join("metadata.db")).unwrap();

    // Stats should be empty now.
    let stats_empty = run(dir.path(), &["--json", "stats"]);
    let empty: serde_json::Value = serde_json::from_slice(&stats_empty.stdout).unwrap();
    assert_eq!(empty["total_conversations"].as_u64(), Some(0));

    // Reindex.
    let reidx = run(dir.path(), &["--json", "reindex"]);
    assert!(reidx.status.success(), "reindex failed: {}", String::from_utf8_lossy(&reidx.stderr));
    let report: serde_json::Value = serde_json::from_slice(&reidx.stdout).unwrap();
    assert_eq!(report["conversations_indexed"].as_u64(), Some(2));

    // Stats should now match the original.
    let stats_after = run(dir.path(), &["--json", "stats"]);
    let after: serde_json::Value = serde_json::from_slice(&stats_after.stdout).unwrap();
    assert_eq!(after["total_conversations"], before["total_conversations"]);
    assert_eq!(after["unique_segments"], before["unique_segments"]);
    assert_eq!(after["total_segments"], before["total_segments"]);
    assert_eq!(after["total_tokens"], before["total_tokens"]);
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
