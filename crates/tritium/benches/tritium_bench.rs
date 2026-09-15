//! Criterion benchmarks for the multi-layer tritium solver.
//!
//! Synthetic two-layer W/Cu-like stack (caller Arrhenius data, no property
//! tables): steady Thomas solve plus a short Dirichlet transient.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nucleide_tritium::layers::{Interface, LayerSpec, LayerStack};
use nucleide_tritium::solve::{InitialState, SolverOptions, TimeGrid};
use nucleide_tritium::{solve_layers, steady_layers, Boundary};

/// Two-layer stack, 256 + 256 cells, trap-free, uniform 500 K.
fn bench_stack() -> LayerStack {
    LayerStack::new(
        vec![
            LayerSpec::new(5e-4, 256, 1e-9, 0.0, 2.0, vec![], vec![500.0], vec![]).unwrap(),
            LayerSpec::new(5e-4, 256, 5e-10, 0.0, 0.5, vec![], vec![500.0], vec![]).unwrap(),
        ],
        vec![Interface::Sieverts],
    )
    .unwrap()
}

fn bench_tritium(c: &mut Criterion) {
    let stack = bench_stack();
    let n = stack.total_cells();
    let left = Boundary::Dirichlet(1.0);
    let right = Boundary::Dirichlet(0.0);

    let mut group = c.benchmark_group("tritium_layers_512");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("steady_layers", |b| {
        b.iter(|| steady_layers(&stack, &left, &right).expect("bench steady"))
    });

    // Short transient: 5 output times, default CN options.
    let grid = TimeGrid::new(vec![100.0, 200.0, 400.0, 800.0, 1600.0]).unwrap();
    let initial = InitialState {
        mobile: vec![0.0; n],
        trapped: vec![Vec::new(); n],
    };
    let opts = SolverOptions::default();
    group.bench_function("transient_5_outputs", |b| {
        b.iter(|| {
            solve_layers(&stack, &left, &right, &grid, &initial, &opts).expect("bench transient")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_tritium);
criterion_main!(benches);
