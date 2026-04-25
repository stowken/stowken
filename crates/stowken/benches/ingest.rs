use criterion::{criterion_group, criterion_main, Criterion};
use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

fn make_conversation(i: u32) -> Conversation {
    Conversation {
        id: None,
        application: None,
        model: "gpt-4".to_owned(),
        tokenizer: "cl100k_base".to_owned(),
        messages: vec![
            Message {
                role: "system".to_owned(),
                content: MessageContent::Tokens(vec![1u32, 2, 3, 4, 5]),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_owned(),
                content: MessageContent::Tokens((i..(i + 500)).collect()),
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "assistant".to_owned(),
                content: MessageContent::Tokens((i + 1000..(i + 1500)).collect()),
                name: None,
                tool_call_id: None,
            },
        ],
        metadata: None,
    }
}

fn bench_ingest(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("store_conversation_5k_tokens", |b| {
        b.to_async(&runtime).iter(|| async {
            let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
                .await
                .unwrap();
            vault.store(make_conversation(0)).await.unwrap()
        });
    });
}

fn bench_ingest_high_dedup(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("store_conversation_high_dedup", |b| {
        b.to_async(&runtime).iter_batched(
            || {
                // Pre-populate vault
                runtime.block_on(async {
                    let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
                        .await
                        .unwrap();
                    vault.store(make_conversation(0)).await.unwrap();
                    vault
                })
            },
            |vault| async move {
                // Same conversation again — high dedup path
                vault.store(make_conversation(0)).await.unwrap()
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_ingest, bench_ingest_high_dedup);
criterion_main!(benches);
