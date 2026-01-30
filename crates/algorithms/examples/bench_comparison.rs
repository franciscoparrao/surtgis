//! Benchmark comparison: SurtGis vs GDAL vs SAGA vs WhiteboxTools vs R (terra)
//!
//! Generates a synthetic DEM, runs slope/aspect/hillshade with each tool,
//! and prints a wall-clock comparison table.
//!
//! External tools are optional — only included if found on the system.
//!
//! Run:
//!   cargo run -p surtgis-algorithms --example bench_comparison --release
//!
//! Custom size (default 2048):
//!   cargo run -p surtgis-algorithms --example bench_comparison --release -- --size 4096

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use surtgis_algorithms::terrain::{
    aspect, hillshade, slope, AspectOutput, HillshadeParams, SlopeParams,
};
use surtgis_core::io::{write_geotiff, GeoTiffOptions};
use surtgis_core::{GeoTransform, Raster};

const SAGA_PATH: &str = "/home/franciscoparrao/proyectos/quinteros/saga-9.9.1_src/saga-9.9.1/saga-gis/build/src/saga_core/saga_cmd/saga_cmd";
const WBT_PATH: &str = "/home/franciscoparrao/.local/lib/python3.12/site-packages/whitebox/WBT/whitebox_tools";

fn main() {
    let size = parse_size();
    let out_dir = Path::new("output/bench_comparison");
    fs::create_dir_all(out_dir).expect("Cannot create output directory");

    println!("=== SurtGis Benchmark Comparison ===");
    println!("DEM size: {}x{} ({:.1}M cells)\n", size, size, size as f64 * size as f64 / 1e6);

    // 1. Generate and write DEM
    let dem = generate_dem(size);
    let dem_path = out_dir.join("dem.tif");
    write_geotiff(&dem, &dem_path, Some(GeoTiffOptions::default())).unwrap();

    // 2. Check tool availability
    let has_gdal = check_tool("gdaldem", &["--help"]);
    let has_saga = check_tool(SAGA_PATH, &["--version"]);
    let has_wbt = check_tool(WBT_PATH, &["--version"]);
    let has_r = check_r_terra();

    println!(
        "Tools:  SurtGis  GDAL={}  SAGA={}  Whitebox={}  R/terra={}",
        yn(has_gdal), yn(has_saga), yn(has_wbt), yn(has_r),
    );
    if has_saga {
        println!("  SAGA: 9.9.1 ({})", SAGA_PATH);
    }
    if has_wbt {
        println!("  Whitebox: 2.4.0 ({})", WBT_PATH);
    }
    println!();

    // 3. Run benchmarks
    let algorithms = ["slope", "aspect", "hillshade"];

    // Table header
    print!("{:<12} {:>10}", "Algorithm", "SurtGis");
    if has_gdal { print!(" {:>10} {:>7}", "GDAL", "vs"); }
    if has_saga { print!(" {:>10} {:>7}", "SAGA", "vs"); }
    if has_wbt { print!(" {:>10} {:>7}", "Whitebox", "vs"); }
    if has_r { print!(" {:>10} {:>7}", "R/terra", "vs"); }
    println!();

    print!("{:<12} {:>10}", "─────────", "────────");
    if has_gdal { print!(" {:>10} {:>7}", "────────", "─────"); }
    if has_saga { print!(" {:>10} {:>7}", "────────", "─────"); }
    if has_wbt { print!(" {:>10} {:>7}", "────────", "─────"); }
    if has_r { print!(" {:>10} {:>7}", "────────", "─────"); }
    println!();

    for alg in &algorithms {
        // SurtGis
        let surtgis_ms = bench_surtgis(alg, &dem, out_dir);
        print!("{:<12} {:>7.1} ms", alg, surtgis_ms);

        // GDAL
        if has_gdal {
            match bench_gdal(alg, &dem_path, out_dir) {
                Some(ms) => print!(" {:>7.1} ms {:>6.2}x", ms, ms / surtgis_ms),
                None => print!(" {:>10} {:>7}", "err", "-"),
            }
        }

        // SAGA
        if has_saga {
            match bench_saga(alg, &dem_path, out_dir) {
                Some(ms) => print!(" {:>7.1} ms {:>6.2}x", ms, ms / surtgis_ms),
                None => print!(" {:>10} {:>7}", "err", "-"),
            }
        }

        // WhiteboxTools
        if has_wbt {
            match bench_whitebox(alg, &dem_path, out_dir) {
                Some(ms) => print!(" {:>7.1} ms {:>6.2}x", ms, ms / surtgis_ms),
                None => print!(" {:>10} {:>7}", "err", "-"),
            }
        }

        // R/terra
        if has_r {
            match bench_r_terra(alg, &dem_path, out_dir) {
                Some(ms) => print!(" {:>7.1} ms {:>6.2}x", ms, ms / surtgis_ms),
                None => print!(" {:>10} {:>7}", "err", "-"),
            }
        }

        println!();
    }

    println!("\n\"vs\" = tool_time / surtgis_time (>1 means SurtGis is faster)");
    println!("Output files in {}/", out_dir.display());
}

// ─── SurtGis ────────────────────────────────────────────────────────────

fn bench_surtgis(algorithm: &str, dem: &Raster<f64>, out_dir: &Path) -> f64 {
    let start = Instant::now();
    let result: Raster<f64> = match algorithm {
        "slope" => slope(dem, SlopeParams::default()).unwrap(),
        "aspect" => aspect(dem, AspectOutput::Degrees).unwrap(),
        "hillshade" => hillshade(dem, HillshadeParams::default()).unwrap(),
        _ => unreachable!(),
    };
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let path = out_dir.join(format!("surtgis_{}.tif", algorithm));
    write_geotiff(&result, &path, Some(GeoTiffOptions::default())).unwrap();
    ms
}

// ─── GDAL ───────────────────────────────────────────────────────────────

fn bench_gdal(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let output = out_dir.join(format!("gdal_{}.tif", algorithm));
    let _ = fs::remove_file(&output);

    let args: Vec<&str> = match algorithm {
        "slope" => vec!["slope", dem_path.to_str()?, output.to_str()?],
        "aspect" => vec!["aspect", dem_path.to_str()?, output.to_str()?],
        "hillshade" => vec!["hillshade", dem_path.to_str()?, output.to_str()?, "-az", "315", "-alt", "45"],
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
    status.success().then_some(ms)
}

// ─── SAGA 9.9.1 ────────────────────────────────────────────────────────

fn bench_saga(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let output = out_dir.join(format!("saga_{}.sgrd", algorithm));

    // SAGA needs its library path set
    let saga_dir = PathBuf::from(SAGA_PATH);
    let saga_lib = saga_dir
        .parent()?
        .parent()?
        .join("saga_core")
        .join("saga_api");

    // Also set the tools path
    let tools_path = saga_dir.parent()?.parent()?.join("tools");

    let (library, module, extra_args): (&str, &str, Vec<String>) = match algorithm {
        "slope" => (
            "ta_morphometry", "0",
            vec![
                format!("-ELEVATION={}", dem_path.to_str()?),
                format!("-SLOPE={}", output.to_str()?),
            ],
        ),
        "aspect" => (
            "ta_morphometry", "0",
            vec![
                format!("-ELEVATION={}", dem_path.to_str()?),
                format!("-ASPECT={}", output.to_str()?),
            ],
        ),
        "hillshade" => (
            "ta_lighting", "0",
            vec![
                format!("-ELEVATION={}", dem_path.to_str()?),
                format!("-SHADE={}", output.to_str()?),
            ],
        ),
        _ => return None,
    };

    let mut cmd = Command::new(SAGA_PATH);
    cmd.arg(library).arg(module);
    for arg in &extra_args {
        cmd.arg(arg);
    }

    // Set library search paths
    let ld_path = format!(
        "{}:{}:{}",
        saga_lib.display(),
        tools_path.display(),
        env::var("LD_LIBRARY_PATH").unwrap_or_default()
    );
    cmd.env("LD_LIBRARY_PATH", &ld_path);
    cmd.env("SAGA_TLB", tools_path.to_str()?);

    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let start = Instant::now();
    let status = cmd.status().ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    status.success().then_some(ms)
}

// ─── WhiteboxTools ──────────────────────────────────────────────────────

fn bench_whitebox(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let output = out_dir.join(format!("wbt_{}.tif", algorithm));
    let _ = fs::remove_file(&output);

    let abs_dem = fs::canonicalize(dem_path).ok()?;
    let abs_out = out_dir.canonicalize().ok()?.join(format!("wbt_{}.tif", algorithm));

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
    status.success().then_some(ms)
}

// ─── R / terra ──────────────────────────────────────────────────────────

fn check_r_terra() -> bool {
    Command::new("Rscript")
        .args(["-e", "library(terra)"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn bench_r_terra(algorithm: &str, dem_path: &Path, out_dir: &Path) -> Option<f64> {
    let abs_dem = fs::canonicalize(dem_path).ok()?;
    let abs_out = out_dir.canonicalize().ok()?.join(format!("r_{}.tif", algorithm));

    let r_code = match algorithm {
        "slope" => format!(
            r#"library(terra); r <- rast("{}"); t0 <- proc.time(); s <- terrain(r, "slope", unit="degrees"); cat((proc.time()-t0)[3]*1000); writeRaster(s, "{}", overwrite=TRUE)"#,
            abs_dem.display(), abs_out.display()
        ),
        "aspect" => format!(
            r#"library(terra); r <- rast("{}"); t0 <- proc.time(); s <- terrain(r, "aspect", unit="degrees"); cat((proc.time()-t0)[3]*1000); writeRaster(s, "{}", overwrite=TRUE)"#,
            abs_dem.display(), abs_out.display()
        ),
        "hillshade" => format!(
            r#"library(terra); r <- rast("{}"); t0 <- proc.time(); sl <- terrain(r, "slope", unit="radians"); asp <- terrain(r, "aspect", unit="radians"); h <- shade(sl, asp, angle=45, direction=315); cat((proc.time()-t0)[3]*1000); writeRaster(h, "{}", overwrite=TRUE)"#,
            abs_dem.display(), abs_out.display()
        ),
        _ => return None,
    };

    let output = Command::new("Rscript")
        .args(["-e", &r_code])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // R prints the elapsed time in ms to stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<f64>().ok()
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn generate_dem(size: usize) -> Raster<f64> {
    let mut dem = Raster::new(size, size);
    dem.set_transform(GeoTransform::new(500_000.0, 4_500_000.0, 10.0, -10.0));

    let center = size as f64 / 2.0;
    for row in 0..size {
        for col in 0..size {
            let dx = col as f64 - center;
            let dy = row as f64 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            let base = (500.0 - dist * 0.5).max(50.0);
            let ridge1 = ((row as f64 * 0.05).sin() * 30.0).abs();
            let ridge2 = ((col as f64 * 0.03).cos() * 20.0).abs();
            let noise = ((row * 7 + col * 13) % 41) as f64 * 0.5;

            dem.set(row, col, base + ridge1 + ridge2 + noise).unwrap();
        }
    }
    dem
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

fn parse_size() -> usize {
    let args: Vec<String> = env::args().collect();
    for i in 1..args.len() {
        if args[i] == "--size" {
            if let Some(s) = args.get(i + 1) {
                return s.parse().unwrap_or(2048);
            }
        }
    }
    2048
}
