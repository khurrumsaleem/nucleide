//! Criterion benchmarks for MCPL merge / extract / stats.
//!
//! Synthetic in-memory particle lists (no fixtures needed): 20 000 records
//! exercising the O(range) extract, the streaming stats tally, and the
//! reserved merge path.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nucleide_mcpl_io::{
    encode_file, extract_mcpl, mcpl_stats, merge_mcpl, ExtractSpec, Header, McplFile, Particle,
};

/// Deterministic synthetic particles: axis unit directions, linearly
/// ramped energies/weights (all finite, unit directions).
fn synthetic_particles(n: usize) -> Vec<Particle> {
    (0..n)
        .map(|i| {
            let f = i as f64;
            Particle {
                ekin: 0.5 + (f % 100.0) * 0.01,
                polarisation: [0.0; 3],
                position: [f * 0.01, -f * 0.005, 0.25],
                direction: [0.0, 0.0, 1.0],
                time: f * 1e-3,
                weight: 1.0 + (f % 7.0) * 0.1,
                pdgcode: 2112,
                userflags: 0,
            }
        })
        .collect()
}

fn bench_file(n: usize) -> McplFile {
    let bytes = encode_file(&Header::default(), &synthetic_particles(n)).unwrap();
    McplFile::from_bytes(bytes).unwrap()
}

fn bench_mcpl(c: &mut Criterion) {
    let n = 20_000;
    let file = bench_file(n);
    let other = bench_file(n / 2);

    let mut group = c.benchmark_group("mcpl_utils_20k");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(8));

    group.bench_function("stats_streaming", |b| {
        b.iter(|| mcpl_stats(&file).expect("bench stats"))
    });

    group.bench_function("extract_range_1k", |b| {
        let spec = ExtractSpec::Range(1000..2000);
        b.iter(|| extract_mcpl(&file, &spec).expect("bench extract"))
    });

    group.bench_function("merge_two_files", |b| {
        b.iter(|| merge_mcpl(&[file.clone(), other.clone()]).expect("bench merge"))
    });

    group.finish();
}

criterion_group!(benches, bench_mcpl);
criterion_main!(benches);
