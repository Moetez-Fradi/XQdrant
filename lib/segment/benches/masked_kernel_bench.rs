//! Cross-cut X1 — gather vs repack masked-distance kernels.
//!
//! Compares per-op latency of:
//! - **repack**: copy focus dims into a contiguous scratch buffer, then contiguous SIMD metric
//! - **gather**: score focus dims in place (`vgatherdps` on x86_64 AVX2; scalar elsewhere)
//!
//! Emits CSV (`d_sub,kernel,ns_per_op`) for the Python plot script:
//! `DB_systems/research/xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py`
//!
//! ## Run
//!
//! ```bash
//! cd XQdrant
//! # Fast path: measure + write CSV only (skip Criterion plots)
//! XQDRANT_KERNEL_BENCH_CSV_ONLY=1 cargo bench -p segment --bench masked_kernel_bench
//!
//! # Full Criterion + CSV:
//! cargo bench -p segment --bench masked_kernel_bench
//!
//! # CSV default: target/masked_kernel_bench/kernel_bench.csv
//! # Override:    XQDRANT_KERNEL_BENCH_CSV=/path/to/kernel_bench.csv
//!
//! # Then plot:
//! python DB_systems/research/xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py \
//!   --kernel-csv target/masked_kernel_bench/kernel_bench.csv
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use segment::spaces::simple::dot_similarity;
use segment::types::Distance;
use segment::vector_storage::query_scorer::masked_kernels::{score_gather_f32, score_repack_f32};

/// Full vector width for the synthetic corpus (focus dims are a subset).
const FULL_DIM: usize = 1536;
const NUM_VECTORS: usize = 2048;
const WARMUP_ITERS: usize = 200;
const MEASURE_ITERS: usize = 50_000;

const D_SUBS: &[usize] = &[32, 128, 192, 384, 768];

fn csv_path() -> PathBuf {
    if let Ok(p) = std::env::var("XQDRANT_KERNEL_BENCH_CSV") {
        return PathBuf::from(p);
    }
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
    PathBuf::from(target)
        .join("masked_kernel_bench")
        .join("kernel_bench.csv")
}

fn csv_only() -> bool {
    matches!(
        std::env::var("XQDRANT_KERNEL_BENCH_CSV_ONLY").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn make_fixture(d_sub: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<Vec<f32>>, Vec<usize>) {
    let query: Vec<f32> = (0..d_sub).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vectors: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|_| (0..FULL_DIM).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    // Spread focus dims across the full vector (non-contiguous).
    let step = (FULL_DIM / d_sub).max(1);
    let dims: Vec<usize> = (0..d_sub).map(|i| (i * step) % FULL_DIM).collect();
    (query, vectors, dims)
}

fn measure_ns_per_op(kernel: &str, query: &[f32], vectors: &[Vec<f32>], dims: &[usize]) -> f64 {
    let mut scratch = vec![0.0f32; dims.len()];
    let mut vi = 0usize;

    let score_once = |vi: usize, scratch: &mut [f32]| -> f32 {
        match kernel {
            "repack" => score_repack_f32(query, &vectors[vi], dims, scratch, dot_similarity),
            "gather" => score_gather_f32(Distance::Dot, query, &vectors[vi], dims),
            _ => unreachable!(),
        }
    };

    for _ in 0..WARMUP_ITERS {
        black_box(score_once(vi % NUM_VECTORS, &mut scratch));
        vi += 1;
    }

    let start = Instant::now();
    for _ in 0..MEASURE_ITERS {
        black_box(score_once(vi % NUM_VECTORS, &mut scratch));
        vi += 1;
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / MEASURE_ITERS as f64
}

fn write_kernel_csv(rows: &[(usize, &'static str, f64)]) {
    let path = csv_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create CSV parent dir");
    }
    let mut f = File::create(&path).expect("create kernel_bench.csv");
    writeln!(f, "d_sub,kernel,ns_per_op").unwrap();
    for &(d_sub, kernel, ns) in rows {
        writeln!(f, "{d_sub},{kernel},{ns:.4}").unwrap();
    }
    eprintln!("[masked_kernel_bench] wrote {}", path.display());
}

fn collect_csv_rows() -> Vec<(usize, &'static str, f64)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut csv_rows: Vec<(usize, &'static str, f64)> = Vec::new();

    eprintln!("[masked_kernel_bench] measuring gather vs repack…");
    for &d_sub in D_SUBS {
        let (query, vectors, dims) = make_fixture(d_sub, &mut rng);
        for kernel in ["repack", "gather"] {
            let ns = measure_ns_per_op(kernel, &query, &vectors, &dims);
            csv_rows.push((d_sub, kernel, ns));
            eprintln!("  d_sub={d_sub:>4} kernel={kernel:<6} {ns:.2} ns/op");
        }
    }
    write_kernel_csv(&csv_rows);
    csv_rows
}

fn masked_kernel_bench(c: &mut Criterion) {
    // Always emit CSV first so a long Criterion run can be interrupted safely.
    let _rows = collect_csv_rows();

    if csv_only() {
        eprintln!("[masked_kernel_bench] CSV_ONLY=1 — skipping Criterion groups");
        // Register an empty/no-op so criterion_main still succeeds.
        c.bench_function("csv_only_noop", |b| b.iter(|| black_box(0u64)));
        return;
    }

    let mut rng = StdRng::seed_from_u64(42);
    for &d_sub in D_SUBS {
        let (query, vectors, dims) = make_fixture(d_sub, &mut rng);
        let mut group = c.benchmark_group(format!("masked_kernel_d{d_sub}"));
        group
            .sample_size(20)
            .measurement_time(Duration::from_secs(2));

        let query_c = query.clone();
        let vectors_c = vectors.clone();
        let dims_c = dims.clone();

        group.bench_function("repack", |b| {
            let mut scratch = vec![0.0f32; dims_c.len()];
            let mut i = 0usize;
            b.iter(|| {
                let v = &vectors_c[i % NUM_VECTORS];
                i += 1;
                black_box(score_repack_f32(
                    &query_c,
                    v,
                    &dims_c,
                    &mut scratch,
                    dot_similarity,
                ))
            });
        });

        group.bench_function("gather", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let v = &vectors_c[i % NUM_VECTORS];
                i += 1;
                black_box(score_gather_f32(Distance::Dot, &query_c, v, &dims_c))
            });
        });

        group.finish();
    }
}

criterion_group!(benches, masked_kernel_bench);
criterion_main!(benches);
