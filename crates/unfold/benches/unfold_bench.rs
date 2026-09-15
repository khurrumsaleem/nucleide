//! Criterion benchmarks for SAND-II unfolding iteration.
//!
//! Synthetic response matrices only (no evaluated-library data, per the
//! crate contract): a deterministic diagonally-weighted response plus a
//! flat guess, exercising the O(iters x D x G) dense iteration.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nucleide_unfold::sandii::{unfold, SandII};

/// Deterministic pseudo-random value in (0, 1] from integer coordinates.
fn prand(i: usize, j: usize) -> f64 {
    let x = (i.wrapping_mul(0x9E3779B1) ^ j.wrapping_mul(0x85EBCA6B)) as u64;
    let v = x.wrapping_mul(0xC2B2AE35).wrapping_add(0x27D4EB2F);
    ((v >> 11) as f64) / ((1u64 << 53) as f64) * 0.999 + 0.001
}

/// 24-detector x 64-group synthetic response with a dominant diagonal band
/// so the iteration converges in a bounded number of steps.
fn synthetic_response(d: usize, g: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut response = Vec::with_capacity(d);
    for i in 0..d {
        let mut row = Vec::with_capacity(g);
        for j in 0..g {
            // Each detector sees its own group block strongly plus a weak
            // smooth background: diagonally dominant, SAND-II contracts fast.
            let band = if j * d / g == i { 5.0 } else { 0.0 };
            row.push(band + 0.01 * prand(i, j));
        }
        response.push(row);
    }
    // True spectrum: smooth bump; rates folded through the response.
    let truth: Vec<f64> = (0..g)
        .map(|j| 1.0 + (-(((j as f64) - (g as f64) / 2.0).powi(2)) / 200.0).exp() * 5.0)
        .collect();
    let rates: Vec<f64> = response
        .iter()
        .map(|row| row.iter().zip(&truth).map(|(r, t)| r * t).sum())
        .collect();
    let guess = vec![1.0; g];
    (response, rates, guess)
}

fn bench_unfold(c: &mut Criterion) {
    let (response, rates, guess) = synthetic_response(24, 64);

    let mut group = c.benchmark_group("unfold_sandii_24x64");
    group.throughput(Throughput::Elements(24 * 64));
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(8));

    group.bench_function("unfold_to_1e-6", |b| {
        b.iter(|| unfold(&response, &rates, &guess, 1e-6, 2000).expect("bench unfold converges"))
    });

    // Single-iteration cost (the hot loop body) with a loose cap.
    group.bench_function("ten_iterations", |b| {
        b.iter(|| {
            let mut run = SandII::new(&response, &rates, &guess, 1e-300, 10).expect("bench setup");
            let mut n = 0;
            for _ in &mut run {
                n += 1;
            }
            assert_eq!(n, 10);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_unfold);
criterion_main!(benches);
