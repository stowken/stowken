use stowken::{
    compression::{varint, CompressionPipeline},
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

#[test]
fn varint_compression_reduces_size_for_small_tokens() {
    // Tokens 0–127 cost 1 byte each vs 4 bytes raw
    let tokens: Vec<u32> = (0u32..1000).map(|i| i % 128).collect();
    let encoded = varint::encode_tokens(&tokens);
    let raw_size = tokens.len() * 4;
    assert!(
        encoded.len() < raw_size,
        "varint should compress small tokens: {} < {}",
        encoded.len(),
        raw_size
    );
}

#[test]
fn full_pipeline_compress_decompress_roundtrip() {
    let pipeline = CompressionPipeline::new(true, 3);
    let tokens: Vec<u32> = (0u32..500).map(|i| i % 50_000).collect();
    let compressed = pipeline.compress(&tokens).unwrap();
    let decompressed = pipeline.decompress(&compressed).unwrap();
    assert_eq!(tokens, decompressed);
}

#[test]
fn compression_disabled_still_roundtrips() {
    let pipeline = CompressionPipeline::new(false, 3);
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let compressed = pipeline.compress(&tokens).unwrap();
    let decompressed = pipeline.decompress(&compressed).unwrap();
    assert_eq!(tokens, decompressed);
}

#[test]
fn zstd_achieves_meaningful_compression_ratio() {
    let pipeline = CompressionPipeline::new(true, 3);
    // Highly compressible: same pattern repeated
    let tokens: Vec<u32> = (0u32..2000).map(|i| i % 10).collect();
    let compressed = pipeline.compress(&tokens).unwrap();
    let raw_size = tokens.len() * 4; // naive storage
    let ratio = compressed.len() as f64 / raw_size as f64;
    assert!(
        ratio < 0.5,
        "expected compression ratio < 0.5 for repetitive data, got {ratio:.3}"
    );
}

#[tokio::test]
async fn vault_reports_nonzero_compression_ratio() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    // Highly compressible: same tokens repeated
    let tokens: Vec<u32> = vec![42u32; 500];
    let conv = Conversation {
        id: None,
        application: None,
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: MessageContent::Tokens(tokens),
            name: None,
            tool_call_id: None,
        }],
        metadata: None,
    };

    let result = vault.store(conv).await.unwrap();
    assert!(
        result.compression_ratio < 1.0,
        "compression ratio should be less than 1.0 for compressible data, got {}",
        result.compression_ratio
    );
}

#[tokio::test]
async fn vault_stats_show_storage_savings() {
    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
        .await
        .unwrap();

    // 5 conversations sharing the same large system prompt
    let system_tokens: Vec<u32> = (0u32..500).map(|i| i % 100).collect();

    for i in 0..5u32 {
        let conv = Conversation {
            id: None,
            application: None,
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k".to_owned(),
            messages: vec![
                Message {
                    role: "system".to_owned(),
                    content: MessageContent::Tokens(system_tokens.clone()),
                    name: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".to_owned(),
                    content: MessageContent::Tokens(vec![i * 1000, i * 1001]),
                    name: None,
                    tool_call_id: None,
                },
            ],
            metadata: None,
        };
        vault.store(conv).await.unwrap();
    }

    let stats = vault.stats().await.unwrap();
    assert!(
        stats.savings_percentage > 0.0,
        "expected positive savings, got {:.1}%",
        stats.savings_percentage
    );
    assert!(
        stats.storage_bytes < stats.naive_bytes,
        "actual storage should be less than naive storage"
    );
}
