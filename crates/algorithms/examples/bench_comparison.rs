//! Benchmark comparison: SurtGis vs GDAL vs GRASS vs WhiteboxTools
//!
//! Fair benchmark: ALL tools (including SurtGIS) measure the full pipeline:
//!   read GeoTIFF → compute → write GeoTIFF
//!
//! SurtGIS uses native Rust I/O (tiff crate, no GDAL dependency).
//! External tools use their own native I/O.
//!
//! Run:
//!   cargo run -p surtgis-algorithms --example bench_comparison --release
//!
//! Custom size:
//!   cargo run -p surtgis-algorithms --example bench_comparison --release -- --size 5000
//!
//! Multiple sizes for paper:
//!   cargo run -p surtgis-algorithms --example bench_comparison --release -- --paper
//!
//! Repetitions:
//!   cargo run -p surtgis-algorithms --example bench_comparison --release -- --reps 10

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use surtgis_algorithms::hydrology::{
    flow_accumulation, flow_direction, priority_flood, PriorityFloodParams,
};
use surtgis_algorithms::terrain::{
    aspect, hillshade, slope, tpi, AspectOutput, HillshadeParams, SlopeParams, TpiParams,
};
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};
use surtgis_core::{GeoTransform, Raster};

const WBT_PATH: &str = "/home/franciscoparrao/.local/lib/python3.12/site-packages/whitebox/WBT/whitebox_tools";

/// Timeout in seconds for external tools
const TIMEOUT_SECS: u64 = 300;

fn main() {
    let args: Vec<String> = env::args().collect();

    // --single-thread: restrict Rayon to 1 thread
    if args.contains(&"--single-thread".into()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
        eprintln!("(single-thread mode: RAYON_NUM_THREADS=1)");
    }

    let out_dir = Path::new("output/bench_comparison");
    fs::create_dir_all(out_dir).expect("Cannot create output directory");

    let reps = parse_flag_usize(&args, "--reps").unwrap_or(5);
    let warmup = parse_flag_usize(&args, "--warmup").unwrap_or(2);

    let sizes: Vec<usize> = if args.contains(&"--paper".into()) {
        vec![1000, 5000, 10000, 20000]
    } else {
        vec![parse_flag_usize(&args, "--size").unwrap_or(2048)]
    };

    // Check tool availability
    let has_gdal = check_tool("gdaldem", &["--help"]);
    let has_grass = check_tool("grass", &["--help"]);
    let has_wbt = check_tool(WBT_PATH, &["--version"]);

    println!("=== SurtGIS Benchmark Comparison (Fair I/O) ===");
    println!("Reps: {reps}, Warmup: {warmup}");
    println!(
        "Tools:  SurtGIS(native)  GDAL={}  GRASS={}  WBT={}",
        yn(has_gdal),
        yn(has_grass),
        yn(has_wbt),
    );
    println!();

    let algorithms = ["slope", "aspect", "hillshade", "tpi", "fill", "flow_acc"];

    // CSV output
    let csv_path = out_dir.join("benchmark_results.csv");
    let mut csv = fs::File::create(&csv_path).expect("Cannot create CSV");
    use std::io::Write;
    writeln!(csv, "algorithm,size,tool,run,time_ms,read_ms,compute_ms,write_ms").unwrap();

    for &size in &sizes {
        println!(
            "\n--- DEM {}x{} ({:.1}M cells) ---",
            size,
            size,
            size as f64 * size as f64 / 1e6
        );

        // Prefer uncompressed Float32 DEMs (fbm_SIZE_raw.tif) for fair I/O benchmarking.
        // Fall back to compressed originals, then generate if neither exists.
        let raw_dem = PathBuf::from(format!("benchmarks/results/dems/fbm_{}_raw.tif", size));
        let existing_dem = if raw_dem.exists() { raw_dem } else {
            PathBuf::from(format!("benchmarks/results/dems/fbm_{}.tif", size))
        };
        let dem_path = if existing_dem.exists() {
            println!("  Using DEM: {}", existing_dem.display());
            existing_dem
        } else {
            let native_path = out_dir.join(format!("dem_{}_native.tif", size));
            let dem_path = out_dir.join(format!("dem_{}.tif", size));
            if !dem_path.exists() {
                print!("  Generating fBm DEM...");
                std::io::stdout().flush().ok();
                let dem = generate_fbm_dem(size);
                write_geotiff(&dem, &native_path, Some(GeoTiffOptions::default())).unwrap();
                // Add proper CRS with gdal_translate so GRASS can read it
                let status = Command::new("gdal_translate")
                    .args([
                        "-a_srs", "EPSG:32719",
                        native_path.to_str().unwrap(),
                        dem_path.to_str().unwrap(),
                    ])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
                if status.map(|s| s.success()).unwrap_or(false) {
                    let _ = fs::remove_file(&native_path);
                } else {
                    // Fallback: use native without CRS (GRASS won't work)
                    let _ = fs::rename(&native_path, &dem_path);
                }
                println!(" done ({:.0}MB)", fs::metadata(&dem_path).unwrap().len() as f64 / 1e6);
            } else {
                println!("  Reusing DEM: {}", dem_path.display());
            }
            dem_path
        };

        for alg in &algorithms {
            // SurtGIS (native Rust I/O) — returns (total, read, compute, write)
            let surtgis_results = run_n_times_tuple(reps, warmup, || {
                bench_surtgis_io(alg, &dem_path, out_dir)
            });

            let surtgis_totals: Vec<f64> = surtgis_results.iter().map(|r| r.0).collect();
            let surtgis_reads: Vec<f64> = surtgis_results.iter().map(|r| r.1).collect();
            let surtgis_computes: Vec<f64> = surtgis_results.iter().map(|r| r.2).collect();
            let surtgis_writes: Vec<f64> = surtgis_results.iter().map(|r| r.3).collect();

            for (i, r) in surtgis_results.iter().enumerate() {
                writeln!(csv, "{alg},{size},surtgis,{},{:.3},{:.3},{:.3},{:.3}",
                    i + 1, r.0, r.1, r.2, r.3).unwrap();
            }

            let s_med = median(&surtgis_totals);
            let s_iqr = iqr(&surtgis_totals);
            let s_read = median(&surtgis_reads);
            let s_comp = median(&surtgis_computes);
            let s_write = median(&surtgis_writes);
            print!("  {:<12} SurtGIS: {:>8.1}ms (R:{:.0} C:{:.0} W:{:.0} IQR:{:.0})",
                alg, s_med, s_read, s_comp, s_write, s_iqr);

            // GDAL
            if has_gdal {
                if let Some(times) = run_tool_n_times(reps, warmup, || {
                    bench_gdal(alg, &dem_path, out_dir)
                }) {
                    for (i, &t) in times.iter().enumerate() {
                        writeln!(csv, "{alg},{size},gdal,{},{:.3},,,", i + 1, t).unwrap();
                    }
                    let med = median(&times);
                    print!("  GDAL: {:>8.1}ms ({:.1}x)", med, med / s_med);
                } else {
                    print!("  GDAL: {:>8}", "n/a");
                }
            }

            // GRASS
            if has_grass {
                if let Some(times) = run_tool_n_times(reps, warmup, || {
                    bench_grass(alg, &dem_path, out_dir)
                }) {
                    for (i, &t) in times.iter().enumerate() {
                        writeln!(csv, "{alg},{size},grass,{},{:.3},,,", i + 1, t).unwrap();
                    }
                    let med = median(&times);
                    print!("  GRASS: {:>8.1}ms ({:.1}x)", med, med / s_med);
                } else {
                    print!("  GRASS: {:>8}", "T/O");
                }
            }

            // WhiteboxTools
            if has_wbt {
                if let Some(times) = run_tool_n_times(reps, warmup, || {
                    bench_whitebox(alg, &dem_path, out_dir)
                }) {
                    for (i, &t) in times.iter().enumerate() {
                        writeln!(csv, "{alg},{size},wbt,{},{:.3},,,", i + 1, t).unwrap();
                    }
                    let med = median(&times);
                    print!("  WBT: {:>8.1}ms ({:.1}x)", med, med / s_med);
                } else {
                    print!("  WBT: {:>8}", "T/O");
                }
            }

            println!();
        }
    }

    println!("\nResults: {}", csv_path.display());
    println!("\"Nx\" = tool_time / surtgis_time (>1 means SurtGIS is faster)");
}

// ─── SurtGIS: full pipeline (native Rust I/O) ─────────────────────────

/// Returns (total_ms, read_ms, compute_ms, write_ms)
fn bench_surtgis_io(algorithm: &str, dem_path: &Path, out_dir: &Path) -> (f64, f64, f64, f64) {
    let t0 = Instant::now();

    // 1. Read GeoTIFF (native Rust, tiff crate)
    let dem: Raster<f64> = read_geotiff(dem_path, None).unwrap();
    let t_read = t0.elapsed().as_secs_f64() * 1000.0;

    // 2. Compute
    let t1 = Instant::now();
    match algorithm {
        "slope" => {
            let r = slope(&dem, SlopeParams::default()).unwrap();
            let t_compute = t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            let path = out_dir.join("surtgis_slope.tif");
            write_geotiff(&r, &path, Some(GeoTiffOptions::default())).unwrap();
            let t_write = t2.elapsed().as_secs_f64() * 1000.0;
            (t_read + t_compute + t_write, t_read, t_compute, t_write)
        }
        "aspect" => {
            let r = aspect(&dem, AspectOutput::Degrees).unwrap();
            let t_compute = t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            let path = out_dir.join("surtgis_aspect.tif");
            write_geotiff(&r, &path, Some(GeoTiffOptions::default())).unwrap();
            let t_write = t2.elapsed().as_secs_f64() * 1000.0;
            (t_read + t_compute + t_write, t_read, t_compute, t_write)
        }
        "hillshade" => {
            let r = hillshade(&dem, HillshadeParams::default()).unwrap();
            let t_compute = t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            let path = out_dir.join("surtgis_hillshade.tif");
            write_geotiff(&r, &path, Some(GeoTiffOptions::default())).unwrap();
            let t_write = t2.elapsed().as_secs_f64() * 1000.0;
            (t_read + t_compute + t_write, t_read, t_compute, t_write)
        }
        "tpi" => {
            let r = tpi(&dem, TpiParams { radius: 10 }).unwrap();
            let t_compute = t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            let path = out_dir.join("surtgis_tpi.tif");
            write_geotiff(&r, &path, Some(GeoTiffOptions::default())).unwrap();
            let t_write = t2.elapsed().as_secs_f64() * 1000.0;
            (t_read + t_compute + t_write, t_read, t_compute, t_write)
        }
        "fill" => {
            let r = priority_flood(&dem, PriorityFloodParams::default()).unwrap();
            let t_compute = t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            let path = out_dir.join("surtgis_fill.tif");
            write_geotiff(&r, &path, Some(GeoTiffOptions::default())).unwrap();
            let t_write = t2.elapsed().as_secs_f64() * 1000.0;
            (t_read + t_compute + t_write, t_read, t_compute, t_write)
        }
        "flow_acc" => {
            let fdir = flow_direction(&dem).unwrap();
            let r = flow_accumulation(&fdir).unwrap();
            let t_compute = t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            let path = out_dir.join("surtgis_flow_acc.tif");
            write_geotiff(&r, &path, Some(GeoTiffOptions::default())).unwrap();
            let t_write = t2.elapsed().as_secs_f64() * 1000.0;
            (t_read + t_compute + t_write, t_read, t_compute, t_write)
        }
        _ => unreachable!(),
    }
}

// ─── GDAL ────────────────────────────────────────────────────────────

fn bench_gdal(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let output = out_dir.join(format!("gdal_{}.tif", algorithm));
    let _ = fs::remove_file(&output);

    let args: Vec<&str> = match algorithm {
        "slope" => vec!["slope", dem_path.to_str()?, output.to_str()?],
        "aspect" => vec!["aspect", dem_path.to_str()?, output.to_str()?],
        "hillshade" => vec![
            "hillshade",
            dem_path.to_str()?,
            output.to_str()?,
            "-az",
            "315",
            "-alt",
            "45",
        ],
        "tpi" => vec!["TPI", dem_path.to_str()?, output.to_str()?],
        _ => return None,
    };

    let start = Instant::now();
    let status = Command::new("gdaldem")
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let _ = fs::remove_file(&output);
    status.success().then_some(ms)
}

// ─── GRASS GIS ───────────────────────────────────────────────────────

fn grass_tmp_flag() -> &'static str {
    // GRASS >= 8.4 uses --tmp-project, older uses --tmp-location
    let output = Command::new("grass")
        .arg("--version")
        .output()
        .ok();

    if let Some(out) = output {
        let text = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        for line in text.lines() {
            if line.contains("GRASS GIS") {
                for word in line.split_whitespace() {
                    if word.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                        let parts: Vec<&str> = word.split('.').collect();
                        if parts.len() >= 2 {
                            let major: u32 = parts[0].parse().unwrap_or(0);
                            let minor: u32 = parts[1].parse().unwrap_or(0);
                            if (major, minor) >= (8, 4) {
                                return "--tmp-project";
                            }
                        }
                        return "--tmp-location";
                    }
                }
            }
        }
    }
    "--tmp-location"
}

fn bench_grass(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let output = out_dir.join(format!("grass_{}.tif", algorithm));
    let _ = fs::remove_file(&output);

    let import_cmd = format!(
        "r.in.gdal input={} output=dem --overwrite && g.region raster=dem",
        dem_path.display()
    );
    let export_cmd = format!(
        "r.out.gdal input=result output={} format=GTiff --overwrite",
        output.display()
    );

    let grass_cmds = match algorithm {
        "slope" => format!(
            "{import_cmd} && r.slope.aspect elevation=dem slope=result --overwrite && {export_cmd}"
        ),
        "aspect" => format!(
            "{import_cmd} && r.slope.aspect elevation=dem aspect=result --overwrite && {export_cmd}"
        ),
        "hillshade" => format!(
            "{import_cmd} && r.relief input=dem output=result --overwrite && {export_cmd}"
        ),
        "fill" => format!(
            "{import_cmd} && r.fill.dir input=dem output=result direction=fdir --overwrite && {export_cmd}"
        ),
        "flow_acc" => format!(
            "{import_cmd} && r.watershed elevation=dem accumulation=result --overwrite && {export_cmd}"
        ),
        _ => return None,
    };

    let tmp_flag = grass_tmp_flag();

    let start = Instant::now();
    let status = Command::new("grass")
        .args([tmp_flag, "EPSG:32719", "--exec", "bash", "-c", &grass_cmds])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .env("GRASS_VERBOSE", "0")
        .status()
        .ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;

    let _ = fs::remove_file(&output);

    if ms > TIMEOUT_SECS as f64 * 1000.0 {
        return None;
    }
    status.success().then_some(ms)
}

// ─── WhiteboxTools ───────────────────────────────────────────────────

fn bench_whitebox(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let output = out_dir.join(format!("wbt_{}.tif", algorithm));
    let _ = fs::remove_file(&output);

    let abs_dem = fs::canonicalize(dem_path).ok()?;
    let abs_out = out_dir
        .canonicalize()
        .ok()?
        .join(format!("wbt_{}.tif", algorithm));

    let args: Vec<String> = match algorithm {
        "slope" => vec![
            "--run=Slope".into(),
            format!("--dem={}", abs_dem.display()),
            format!("--output={}", abs_out.display()),
        ],
        "aspect" => vec![
            "--run=Aspect".into(),
            format!("--dem={}", abs_dem.display()),
            format!("--output={}", abs_out.display()),
        ],
        "hillshade" => vec![
            "--run=Hillshade".into(),
            format!("--dem={}", abs_dem.display()),
            format!("--output={}", abs_out.display()),
            "--azimuth=315".into(),
            "--altitude=45".into(),
        ],
        "fill" => vec![
            "--run=FillDepressions".into(),
            format!("--dem={}", abs_dem.display()),
            format!("--output={}", abs_out.display()),
        ],
        "flow_acc" => {
            // WBT flow_acc requires fill first
            let filled = out_dir
                .canonicalize()
                .ok()?
                .join("wbt_flow_filled.tif");
            let _ = fs::remove_file(&filled);

            let fill_status = Command::new(WBT_PATH)
                .args([
                    "--run=FillDepressions",
                    &format!("--dem={}", abs_dem.display()),
                    &format!("--output={}", filled.display()),
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .ok()?;
            if !fill_status.success() {
                return None;
            }

            vec![
                "--run=D8FlowAccumulation".into(),
                format!("--input={}", filled.display()),
                format!("--output={}", abs_out.display()),
            ]
        }
        _ => return None,
    };

    let start = Instant::now();
    let status = Command::new(WBT_PATH)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;

    let _ = fs::remove_file(&output);
    let _ = fs::remove_file(out_dir.join("wbt_flow_filled.tif"));

    if ms > TIMEOUT_SECS as f64 * 1000.0 {
        return None;
    }
    status.success().then_some(ms)
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Generate fractal Brownian motion DEM (same methodology as Python benchmarks)
fn generate_fbm_dem(size: usize) -> Raster<f64> {
    let mut dem = Raster::new(size, size);
    dem.set_transform(GeoTransform::new(500_000.0, 4_500_000.0, 10.0, -10.0));

    // Deterministic fBm using spectral synthesis
    let hurst = 0.7;
    let seed: u64 = 42;

    // Simple mid-point displacement with deterministic noise
    let center = size as f64 / 2.0;
    for row in 0..size {
        for col in 0..size {
            let dx = col as f64 - center;
            let dy = row as f64 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            // Base elevation (mountain shape)
            let base = (2000.0 - dist * 0.2).max(100.0);

            // Multi-scale deterministic noise (simulate fBm)
            let mut noise = 0.0;
            let mut amp = 500.0;
            let mut freq = 0.001;
            for octave in 0..8u64 {
                let phase = seed.wrapping_mul(octave + 1);
                let nx = col as f64 * freq + phase as f64 * 0.1;
                let ny = row as f64 * freq + phase as f64 * 0.07;
                noise += amp * (nx.sin() * ny.cos() + (nx * 1.3 + ny * 0.7).sin() * 0.5);
                amp *= 2.0_f64.powf(-hurst);
                freq *= 2.0;
            }

            dem.set(row, col, (base + noise).max(0.0)).unwrap();
        }
    }
    dem
}

fn run_n_times_tuple<F>(reps: usize, warmup: usize, mut f: F) -> Vec<(f64, f64, f64, f64)>
where
    F: FnMut() -> (f64, f64, f64, f64),
{
    // Warmup
    for _ in 0..warmup {
        let _ = f();
    }
    // Measured runs
    (0..reps).map(|_| f()).collect()
}

fn run_tool_n_times<F>(reps: usize, warmup: usize, mut f: F) -> Option<Vec<f64>>
where
    F: FnMut() -> Option<f64>,
{
    // Warmup
    for _ in 0..warmup {
        if f().is_none() {
            return None;
        }
    }
    // Measured runs
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        match f() {
            Some(t) => times.push(t),
            None => return None,
        }
    }
    Some(times)
}

fn median(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn iqr(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n < 4 {
        return 0.0;
    }
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    q3 - q1
}

fn check_tool(name: &str, args: &[&str]) -> bool {
    Command::new(name)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn yn(b: bool) -> &'static str {
    if b { "YES" } else { "no" }
}

fn parse_flag_usize(args: &[String], flag: &str) -> Option<usize> {
    for i in 0..args.len() {
        if args[i] == flag {
            return args.get(i + 1).and_then(|s| s.parse().ok());
        }
    }
    None
}
