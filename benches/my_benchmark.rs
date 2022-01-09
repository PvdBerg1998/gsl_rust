use criterion::{criterion_group, criterion_main, Criterion};
use gsl_rust::fft::fft64_packed;

pub fn criterion_benchmark(c: &mut Criterion) {
    // Prepare data
    let y = (0..2u64.pow(20))
        .map(|x| x as f64 / 2.0f64.powi(18) * std::f64::consts::TAU)
        .map(|x| x.cos())
        .collect::<Vec<_>>();
    c.bench_function("fft64 2^20", |b| {
        b.iter_with_large_drop(|| {
            let mut y = y.clone();
            fft64_packed(y.as_mut())
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
