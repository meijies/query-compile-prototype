use arrow::datatypes::ArrowNativeTypeOp;
use criterion::{criterion_group, criterion_main, Criterion};

fn add<T: ArrowNativeTypeOp>(a: T, b: T) -> T {
    return a.add_wrapping(b);
}

fn add_native() -> u32 {
    1 + 2
}

fn add_benchmark(c: &mut Criterion) {
    c.bench_function("add", |b| {
        b.iter(|| criterion::black_box(add(1_u32, 2_u32)))
    });
    c.bench_function("add_native", |b| {
        b.iter(|| criterion::black_box(add_native()))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
