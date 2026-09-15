//! Criterion benchmark for MCNP meshtal end-to-end parsing and WWINP text emission.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nucleide_mcnp_io::meshtal::Meshtal;
use nucleide_mcnp_io::wwinp::Wwinp;

fn fixture_path(name: &str) -> String {
    format!(
        "{}/../../fixtures/mcnp/meshtal/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn bench_mcnp_io(c: &mut Criterion) {
    let single_path = fixture_path("mcnp_meshtal_single_meshtal.txt");
    let multi_path = fixture_path("mcnp_meshtal_multiple_meshtal.txt");

    let single_text = std::fs::read_to_string(&single_path).expect("read single meshtal");
    let multi_text = std::fs::read_to_string(&multi_path).expect("read multiple meshtal");

    let mut group = c.benchmark_group("mcnp_io_parse_meshtal");
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.throughput(Throughput::Bytes(single_text.len() as u64));
    group.bench_function("single_tally", |b| {
        b.iter(|| Meshtal::parse(&single_text).expect("parse single"))
    });

    group.throughput(Throughput::Bytes(multi_text.len() as u64));
    group.bench_function("multiple_tallies", |b| {
        b.iter(|| Meshtal::parse(&multi_text).expect("parse multiple"))
    });

    group.bench_function("single_from_file", |b| {
        b.iter(|| Meshtal::from_file(&single_path).expect("from file"))
    });

    group.finish();

    // WWINP text emission over the largest committed fixture (the
    // streaming `to_text` path reworked in 0.12.0).
    let wwinp_path = format!(
        "{}/../../fixtures/mcnp/wwinp/mcnp_wwinp_wwinp_n.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let wwinp_text = std::fs::read_to_string(&wwinp_path).expect("read wwinp fixture");
    let wwinp = Wwinp::parse(&wwinp_text).expect("parse wwinp fixture");
    let n_values: u64 = wwinp.ww.iter().flatten().flatten().count() as u64;

    let mut wgroup = c.benchmark_group("mcnp_io_wwinp_to_text");
    wgroup.throughput(Throughput::Elements(n_values));
    wgroup.sample_size(20);
    wgroup.measurement_time(std::time::Duration::from_secs(5));
    wgroup.bench_function("wwinp_to_text", |b| {
        b.iter(|| wwinp.to_text().expect("wwinp to_text"))
    });
    wgroup.finish();
}

criterion_group!(benches, bench_mcnp_io);
criterion_main!(benches);
