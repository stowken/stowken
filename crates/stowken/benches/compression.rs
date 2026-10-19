use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use stowken::compression::{varint, CompressionPipeline};

fn bench_varint_encode(c: &mut Criterion) {
    let tokens_1k: Vec<u32> = (0u32..1000).map(|i| i % 100_000).collect();
    let tokens_10k: Vec<u32> = (0u32..10_000).map(|i| i % 100_000).collect();

    let mut group = c.benchmark_group("varint_encode");
    for (name, tokens) in [("1k", &tokens_1k), ("10k", &tokens_10k)] {
        group.bench_with_input(BenchmarkId::from_parameter(name), tokens, |b, t| {
            b.iter(|| varint::encode_tokens(black_box(t)))
        });
    }
    group.finish();
}

fn bench_varint_decode(c: &mut Criterion) {
    let tokens: Vec<u32> = (0u32..10_000).map(|i| i % 100_000).collect();
    let encoded = varint::encode_tokens(&tokens);

    c.bench_function("varint_decode_10k", |b| {
        b.iter(|| varint::decode_tokens(black_box(&encoded)).unwrap())
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    let pipeline = CompressionPipeline::new(true, 3);
    let tokens: Vec<u32> = (0u32..5000).map(|i| i % 50_000).collect();

    let mut group = c.benchmark_group("compression_pipeline");
    group.bench_function("compress_5k", |b| {
        b.iter(|| pipeline.compress(black_box(&tokens)).unwrap())
    });

    let compressed = pipeline.compress(&tokens).unwrap();
    group.bench_function("decompress_5k", |b| {
        b.iter(|| pipeline.decompress(black_box(&compressed)).unwrap())
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_varint_encode,
    bench_varint_decode,
    bench_full_pipeline
);
criterion_main!(benches);
