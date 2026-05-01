//! S3Backend integration tests.
//!
//! These run against a real S3-compatible endpoint (real S3, MinIO, or
//! localstack). They are gated behind environment variables so they
//! don't run in normal `cargo test` invocations.
//!
//! ```sh
//! # Against MinIO:
//! docker run -d --rm -p 9000:9000 -e MINIO_ROOT_USER=minioadmin \
//!     -e MINIO_ROOT_PASSWORD=minioadmin minio/minio server /data
//! # Create bucket via mc (or aws-cli):
//! aws --endpoint-url http://localhost:9000 s3 mb s3://stowken-test
//!
//! STOWKEN_S3_TEST_BUCKET=stowken-test \
//!     STOWKEN_S3_TEST_ENDPOINT=http://localhost:9000 \
//!     AWS_ACCESS_KEY_ID=minioadmin \
//!     AWS_SECRET_ACCESS_KEY=minioadmin \
//!     AWS_REGION=us-east-1 \
//!     cargo test -p stowken --features s3 --test integration_tests s3 -- --test-threads 1
//! ```
//!
//! Without those env vars, every test in this file is skipped via an
//! early return after a stderr note.

#![cfg(feature = "s3")]

use std::sync::Arc;

use stowken::{
    storage::{s3::S3Backend, StorageBackend},
    types::{Conversation, Message, MessageContent, SegmentHash, SegmentType, StoredSegment, StowkenConfig},
    Stowken,
};

/// Build an `S3Backend` from env vars, or `None` if the test should
/// be skipped.
async fn s3_backend_or_skip(test_name: &str) -> Option<S3Backend> {
    let bucket = match std::env::var("STOWKEN_S3_TEST_BUCKET") {
        Ok(b) if !b.is_empty() => b,
        _ => {
            eprintln!(
                "skipping {test_name}: set STOWKEN_S3_TEST_BUCKET to enable"
            );
            return None;
        }
    };

    let endpoint = std::env::var("STOWKEN_S3_TEST_ENDPOINT").ok();
    let prefix = format!(
        "stowken-tests/{}/{}/",
        test_name,
        chrono::Utc::now().timestamp_millis()
    );

    use aws_sdk_s3::config::{Builder, Region};
    let mut cfg_loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
    if let Ok(region) = std::env::var("AWS_REGION") {
        cfg_loader = cfg_loader.region(Region::new(region));
    }
    let shared_cfg = cfg_loader.load().await;

    let mut s3_cfg = Builder::from(&shared_cfg);
    if let Some(ep) = endpoint {
        s3_cfg = s3_cfg.endpoint_url(ep).force_path_style(true);
    }
    let client = aws_sdk_s3::Client::from_conf(s3_cfg.build());
    Some(S3Backend::new(client, bucket, prefix))
}

fn make_segment(hash_hex: &str, payload: Vec<u8>, ref_count: u64) -> StoredSegment {
    StoredSegment {
        hash: SegmentHash(hash_hex.to_owned()),
        segment_type: SegmentType::UserTurn,
        tokenizer: "cl100k_base".to_owned(),
        token_count: payload.len() as u32 / 4,
        raw_size: payload.len() as u32,
        compressed_size: payload.len() as u32,
        compressed_data: payload,
        ref_count,
        created_at: chrono::Utc::now(),
    }
}

const HASH_A: &str = "aabbccddeeff001122334455667788990011223344556677889900112233445566";
const HASH_B: &str = "bbccddeeff00112233445566778899001122334455667788990011223344556677";

// ── Basic correctness ─────────────────────────────────────────────────

#[tokio::test]
async fn s3_segment_roundtrip() {
    let Some(backend) = s3_backend_or_skip("segment_roundtrip").await else {
        return;
    };

    let seg = make_segment(HASH_A, vec![1, 2, 3, 4, 5], 1);
    backend.put_segment(&seg).await.unwrap();
    assert!(backend.has_segment(&seg.hash).await.unwrap());

    let fetched = backend.get_segment(&seg.hash).await.unwrap();
    assert_eq!(fetched.compressed_data, vec![1, 2, 3, 4, 5]);
    assert_eq!(fetched.ref_count, 1);
}

#[tokio::test]
async fn s3_put_segment_idempotent_increments_ref() {
    let Some(backend) = s3_backend_or_skip("put_idempotent").await else {
        return;
    };

    let seg = make_segment(HASH_A, vec![42u8; 64], 1);
    backend.put_segment(&seg).await.unwrap();
    backend.put_segment(&seg).await.unwrap(); // second put → ref_count=2

    let fetched = backend.get_segment(&seg.hash).await.unwrap();
    assert_eq!(
        fetched.ref_count, 2,
        "second put on an existing hash must increment ref_count, not overwrite to 1"
    );
}

#[tokio::test]
async fn s3_decrement_ref_signals_gc_eligible() {
    let Some(backend) = s3_backend_or_skip("decrement_signals_gc").await else {
        return;
    };

    let seg = make_segment(HASH_A, vec![1, 2, 3], 1);
    backend.put_segment(&seg).await.unwrap();
    let zero = backend.decrement_ref(&seg.hash).await.unwrap();
    assert!(zero, "ref_count went 1->0, should signal GC-eligible");
}

// ── CAS-correctness under contention ──────────────────────────────────

#[tokio::test]
async fn s3_concurrent_increments_no_lost_writes() {
    // The v0.1 bug: 20 concurrent increments on the same segment could
    // produce a final ref_count well below 21 because of read-modify-
    // write races. With the v0.6 CAS loop, every increment must land.
    let Some(backend) = s3_backend_or_skip("concurrent_increments").await else {
        return;
    };

    let seg = make_segment(HASH_A, vec![7u8; 32], 1);
    backend.put_segment(&seg).await.unwrap();

    let backend = Arc::new(backend);
    let mut handles = Vec::new();
    for _ in 0..20 {
        let b = Arc::clone(&backend);
        let h = seg.hash.clone();
        handles.push(tokio::spawn(async move {
            b.increment_ref(&h).await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let fetched = backend.get_segment(&seg.hash).await.unwrap();
    assert_eq!(
        fetched.ref_count, 21,
        "expected initial 1 + 20 increments = 21; got {} \
         (silent race condition?)",
        fetched.ref_count
    );
}

#[tokio::test]
async fn s3_concurrent_first_writers_only_one_wins() {
    // Two concurrent put_segment calls on a fresh hash. The
    // If-None-Match guard means exactly one creates the meta sidecar
    // with ref_count=1; the other falls through to a CAS increment.
    // Final ref_count must be 2.
    let Some(backend) = s3_backend_or_skip("concurrent_first_writes").await else {
        return;
    };

    let backend = Arc::new(backend);
    let seg_a = make_segment(HASH_A, vec![100u8; 32], 1);
    let seg_b = seg_a.clone();

    let b1 = Arc::clone(&backend);
    let b2 = Arc::clone(&backend);
    let (r1, r2) = tokio::join!(
        tokio::spawn(async move { b1.put_segment(&seg_a).await }),
        tokio::spawn(async move { b2.put_segment(&seg_b).await }),
    );
    r1.unwrap().unwrap();
    r2.unwrap().unwrap();

    let fetched = backend
        .get_segment(&SegmentHash(HASH_A.to_owned()))
        .await
        .unwrap();
    assert_eq!(
        fetched.ref_count, 2,
        "concurrent put_segment on the same hash must produce ref_count=2"
    );
}

// ── replace_segment_data + storage_size ───────────────────────────────

#[tokio::test]
async fn s3_replace_segment_data_keeps_ref_count() {
    let Some(backend) = s3_backend_or_skip("replace_keeps_ref").await else {
        return;
    };

    let seg = make_segment(HASH_A, vec![1u8; 64], 1);
    backend.put_segment(&seg).await.unwrap();
    backend.increment_ref(&seg.hash).await.unwrap(); // ref_count = 2

    let new_data = vec![9u8; 16];
    backend
        .replace_segment_data(&seg.hash, new_data.clone())
        .await
        .unwrap();

    let fetched = backend.get_segment(&seg.hash).await.unwrap();
    assert_eq!(fetched.compressed_data, new_data);
    assert_eq!(fetched.compressed_size, 16);
    assert_eq!(
        fetched.ref_count, 2,
        "replace_segment_data must not touch ref_count"
    );
}

#[tokio::test]
async fn s3_garbage_collect_removes_zero_ref_segments() {
    let Some(backend) = s3_backend_or_skip("gc_zero_ref").await else {
        return;
    };

    let seg_a = make_segment(HASH_A, vec![1u8; 16], 1);
    let seg_b = make_segment(HASH_B, vec![2u8; 16], 1);
    backend.put_segment(&seg_a).await.unwrap();
    backend.put_segment(&seg_b).await.unwrap();

    // Zero out A; keep B.
    backend.decrement_ref(&seg_a.hash).await.unwrap();
    let candidates = backend.list_garbage().await.unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0], seg_a.hash);

    let deleted = backend.garbage_collect().await.unwrap();
    assert_eq!(deleted, 1);
    assert!(!backend.has_segment(&seg_a.hash).await.unwrap());
    assert!(backend.has_segment(&seg_b.hash).await.unwrap());
}

// ── Full vault integration over S3 ────────────────────────────────────

#[tokio::test]
async fn s3_full_vault_store_retrieve_delete() {
    let Some(backend) = s3_backend_or_skip("full_vault").await else {
        return;
    };

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("metadata.db");
    let vault = Stowken::open(backend, StowkenConfig::default(), db_path.to_str().unwrap())
        .await
        .unwrap();

    let tokens: Vec<u32> = (0..50).collect();
    let conv = Conversation {
        id: Some("s3-full".to_owned()),
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        application: None,
        messages: vec![Message {
            role: "system".to_owned(),
            content: MessageContent::Tokens(tokens.clone()),
            name: None,
            tool_call_id: None,
        }],
        metadata: None,
    };

    let result = vault.store(conv).await.unwrap();
    assert_eq!(result.id, "s3-full");
    assert_eq!(result.new_segments, 1);

    let retrieved = vault.retrieve("s3-full").await.unwrap();
    assert_eq!(retrieved.segments[0].tokens, tokens);

    vault.delete("s3-full").await.unwrap();
    let freed = vault.gc().await.unwrap();
    assert_eq!(freed, 1, "GC after delete should free the orphan");
}
