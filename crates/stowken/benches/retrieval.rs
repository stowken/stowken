use criterion::{criterion_group, criterion_main, Criterion};
use stowken::{
    storage::MemoryBackend,
    types::{Conversation, Message, MessageContent, StowkenConfig},
    Stowken,
};

fn bench_retrieve(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("retrieve_conversation_5k_tokens", |b| {
        let (vault, id) = runtime.block_on(async {
            let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default())
                .await
                .unwrap();
            let conv = Conversation {
                id: None,
                application: None,
                model: "gpt-4".to_owned(),
                tokenizer: "cl100k_base".to_owned(),
                messages: vec![
                    Message {
                        role: "user".to_owned(),
                        content: MessageContent::Tokens((0u32..5000).collect()),
                        name: None,
                        tool_call_id: None,
                    },
                    Message {
                        role: "assistant".to_owned(),
                        content: MessageContent::Tokens((5000u32..10000).collect()),
                        name: None,
                        tool_call_id: None,
                    },
                ],
                metadata: None,
            };
            let r = vault.store(conv).await.unwrap();
            (vault, r.id)
        });

        b.to_async(&runtime)
            .iter(|| async { vault.retrieve(&id).await.unwrap() });
    });
}

criterion_group!(benches, bench_retrieve);
criterion_main!(benches);
