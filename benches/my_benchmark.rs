use criterion::{criterion_group, criterion_main, Criterion};
use gsl_rust::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    // c.bench_function("x", |b| {
    //     b.iter_with_large_drop(|| todo!())
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
