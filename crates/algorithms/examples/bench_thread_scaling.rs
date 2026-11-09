//! Thread-scaling benchmark for parallel-efficiency analysis (paper revision R1.7).
//!
//! Varies `RAYON_NUM_THREADS` for a single SurtGIS run; the driver script invokes
//! this binary once per thread count so the global thread pool is set fresh each
//! time (Rayon's `build_global` panics on second call).
//!
//! Compute-only timing (load + warmup + measured) is what feeds the speedup and
//! parallel-efficiency curves; read/write are reported separately for context but
//! excluded from the parallel analysis since they are largely serial.
//!
//! Run:
//!   cargo run -p surtgis-algorithms --example bench_thread_scaling --release -- \
//!       --threads 8 --size 5000 --reps 5
//!
//! CSV schema (appended):
//!   threads,algorithm,size,rep,read_ms,compute_ms,write_ms,total_ms

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use surtgis_algorithms::hydrology::{
    PriorityFloodParams, flow_accumulation, flow_direction, priority_flood,
};
use surtgis_algorithms::terrain::{
    AspectOutput, HillshadeParams, SlopeParams, TpiParams, aspect, hillshade, slope, tpi,
};
use surtgis_core::Raster;
use surtgis_core::io::{GeoTiffOptions, read_geotiff, write_geotiff};

fn main() {
    let args: Vec<String> = env::args().collect();

    let threads = parse_flag_usize(&args, "--threads").unwrap_or(0);
    let size = parse_flag_usize(&args, "--size").unwrap_or(5000);
    let reps = parse_flag_usize(&args, "--reps").unwrap_or(5);
    let warmup = parse_flag_usize(&args, "--warmup").unwrap_or(2);
    let csv_path = parse_flag_string(&args, "--csv")
        .unwrap_or_else(|| "output/bench_thread_scaling/results.csv".to_string());

    let algorithms_arg = parse_flag_string(&args, "--algos");
    let algorithms: Vec<&str> = match algorithms_arg.as_deref() {
        Some(s) => s.split(',').collect(),
        None => vec!["slope", "aspect", "hillshade", "tpi", "fill", "flow_acc"],
    };

    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("Rayon global pool already initialised — invoke binary once per thread count");
    }
    let actual_threads = rayon::current_num_threads();
    eprintln!(
        "[bench] threads_requested={} threads_actual={} size={} reps={} warmup={}",
        threads, actual_threads, size, reps, warmup
    );

    let dem_path = locate_dem(size).unwrap_or_else(|| {
        eprintln!(
            "[bench] FATAL: DEM not found for size={}. Looked under benchmarks/results/dems/.",
            size
        );
        std::process::exit(2);
    });
    eprintln!("[bench] dem={}", dem_path.display());

    let out_dir = Path::new("output/bench_thread_scaling");
    fs::create_dir_all(out_dir).expect("cannot create output dir");

    let csv_p = PathBuf::from(&csv_path);
    if let Some(parent) = csv_p.parent() {
        fs::create_dir_all(parent).expect("cannot create csv parent dir");
    }
    let need_header = !csv_p.exists();
    let mut csv = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_p)
        .expect("cannot open csv");
    if need_header {
        writeln!(
            csv,
            "threads,algorithm,size,rep,read_ms,compute_ms,write_ms,total_ms"
        )
        .unwrap();
    }

    let dem: Raster<f64> = read_geotiff(&dem_path, None).expect("failed to read DEM");
    eprintln!(
        "[bench] DEM loaded: {}x{} = {:.1}M cells",
        dem.rows(),
        dem.cols(),
        dem.rows() as f64 * dem.cols() as f64 / 1e6
    );

    for alg in &algorithms {
        eprintln!("[bench] >>> algorithm={alg}");
        for _ in 0..warmup {
            let _ = bench_one(&dem, alg, out_dir);
        }
        for rep in 1..=reps {
            let (t_read, t_compute, t_write, t_total) = bench_one(&dem, alg, out_dir);
            writeln!(
                csv,
                "{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
                actual_threads, alg, size, rep, t_read, t_compute, t_write, t_total
            )
            .unwrap();
            eprintln!(
                "[bench]   rep={} compute_ms={:.1} total_ms={:.1}",
                rep, t_compute, t_total
            );
        }
    }
    eprintln!("[bench] done -> {}", csv_p.display());
}

fn bench_one(dem: &Raster<f64>, algorithm: &str, out_dir: &Path) -> (f64, f64, f64, f64) {
    let t_read = 0.0;

    let t1 = Instant::now();
    let computed: Raster<f64> = match algorithm {
        "slope" => slope(dem, SlopeParams::default()).unwrap(),
        "aspect" => aspect(dem, AspectOutput::Degrees).unwrap(),
        "hillshade" => hillshade(dem, HillshadeParams::default()).unwrap(),
        "tpi" => tpi(dem, {
            let mut p = TpiParams::default();
            p.radius = 10;
            p
        })
        .unwrap(),
        "fill" => priority_flood(dem, PriorityFloodParams::default()).unwrap(),
        "flow_acc" => {
            let fdir = flow_direction(dem).unwrap();
            let acc = flow_accumulation(&fdir).unwrap();
            // cast accumulation (u32 typically) into f64 raster for write parity
            let mut out = Raster::<f64>::new(acc.rows(), acc.cols());
            out.set_transform(*acc.transform());
            for r in 0..acc.rows() {
                for c in 0..acc.cols() {
                    let v = acc.get(r, c).unwrap();
                    out.set(r, c, v as f64).unwrap();
                }
            }
            out
        }
        _ => panic!("unknown algorithm: {}", algorithm),
    };
    let t_compute = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let path = out_dir.join(format!("{}.tif", algorithm));
    write_geotiff(&computed, &path, Some(GeoTiffOptions::default())).unwrap();
    let t_write = t2.elapsed().as_secs_f64() * 1000.0;

    let total = t_read + t_compute + t_write;
    (t_read, t_compute, t_write, total)
}

fn locate_dem(size: usize) -> Option<PathBuf> {
    let raw = PathBuf::from(format!("benchmarks/results/dems/fbm_{}_raw.tif", size));
    if raw.exists() {
        return Some(raw);
    }
    let compressed = PathBuf::from(format!("benchmarks/results/dems/fbm_{}.tif", size));
    if compressed.exists() {
        return Some(compressed);
    }
    None
}

fn parse_flag_usize(args: &[String], flag: &str) -> Option<usize> {
    for i in 0..args.len() {
        if args[i] == flag {
            return args.get(i + 1).and_then(|s| s.parse().ok());
        }
    }
    None
}

fn parse_flag_string(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag {
            return args.get(i + 1).cloned();
        }
    }
    None
}
