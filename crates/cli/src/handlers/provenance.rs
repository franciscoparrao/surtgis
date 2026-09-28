//! `surtgis provenance` and `surtgis verify`: read the record embedded in
//! a SurtGIS output and check it against the file and its inputs.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use surtgis_core::Raster;
use surtgis_core::dispatch_any;
use surtgis_core::io::{raster_data_hash, read_geotiff_any, read_geotiff_bands, read_provenance};
use surtgis_core::provenance::hash_file;

/// `surtgis provenance <file> [--json]`.
pub fn show(input: PathBuf, json: bool) -> Result<()> {
    let Some(p) = read_provenance(&input).context("Failed to read provenance")? else {
        println!("{}: no provenance record", input.display());
        return Ok(());
    };
    if json {
        println!("{}", p.to_json_pretty());
        return Ok(());
    }
    println!("File: {}", input.display());
    println!(
        "Engine: {} {}  ({})",
        p.engine.name, p.engine.version, p.platform
    );
    println!("Created: {}", p.created);
    if let Some(op) = &p.operation {
        println!("Operation: {op}");
    }
    if !p.command.is_empty() {
        println!("Command: {}", p.command.join(" "));
    }
    if let Some(cwd) = &p.cwd {
        println!("Working dir: {cwd}");
    }
    if let Some(t) = p.threads {
        println!("Threads: {t}");
    }
    if let Some(s) = p.seed {
        println!("Seed: {s}");
    }
    if !p.parameters.is_null() {
        println!("Parameters: {}", p.parameters);
    }
    println!("Inputs ({}):", p.inputs.len());
    for i in &p.inputs {
        let digest = i
            .blake3
            .as_deref()
            .map(|h| format!("blake3:{}…", &h[..16.min(h.len())]))
            .unwrap_or_else(|| "(not hashed)".into());
        let size = i.bytes.map(|b| format!(", {b} bytes")).unwrap_or_default();
        let role = i
            .role
            .as_deref()
            .map(|r| format!(" [{r}]"))
            .unwrap_or_default();
        println!("  {}{}  {}{}", i.source, role, digest, size);
    }
    match &p.output {
        Some(o) => println!(
            "Output: {} {}x{}, {} band(s), blake3:{}",
            o.dtype, o.shape[0], o.shape[1], o.bands, o.data_blake3
        ),
        None => println!("Output: (no digest)"),
    }
    Ok(())
}

/// `surtgis verify <file> [--skip-inputs]`. Exit status is non-zero when
/// the output digest or any hashed input does not match.
pub fn verify(input: PathBuf, skip_inputs: bool) -> Result<()> {
    let Some(p) = read_provenance(&input).context("Failed to read provenance")? else {
        bail!(
            "{}: no provenance record to verify against",
            input.display()
        );
    };
    let Some(expected) = &p.output else {
        bail!("{}: record carries no output digest", input.display());
    };

    let actual = output_digest(&input, &expected.dtype, expected.bands)?;
    let output_ok = actual == expected.data_blake3;
    println!(
        "Output  {}  {}",
        if output_ok { "OK      " } else { "MISMATCH" },
        input.display()
    );
    if !output_ok {
        println!("  recorded blake3:{}", expected.data_blake3);
        println!("  actual   blake3:{actual}");
    }

    let mut failed = !output_ok;
    let mut unverifiable = 0usize;
    if !skip_inputs {
        for i in &p.inputs {
            let Some(recorded) = &i.blake3 else {
                println!("Input   skipped   {} (not hashed at write time)", i.source);
                unverifiable += 1;
                continue;
            };
            let Some(path) = locate(&i.source, p.cwd.as_deref()) else {
                println!("Input   missing   {}", i.source);
                unverifiable += 1;
                continue;
            };
            let (digest, _) =
                hash_file(&path).with_context(|| format!("hashing {}", path.display()))?;
            let ok = &digest == recorded;
            println!(
                "Input   {}  {}",
                if ok { "OK      " } else { "MISMATCH" },
                path.display()
            );
            failed |= !ok;
        }
    }

    if failed {
        bail!("verification FAILED");
    }
    if unverifiable > 0 {
        println!("Verified: output matches; {unverifiable} input(s) could not be checked");
    } else {
        println!("Verified: output and all inputs match the record");
    }
    Ok(())
}

/// Recompute the array digest as the writer defined it: single band via
/// the file's own dtype, stacks band-sequentially in the recorded dtype.
fn output_digest(path: &Path, dtype: &str, bands: usize) -> Result<String> {
    if bands <= 1 {
        let any = read_geotiff_any(path, None).context("Failed to read raster")?;
        let digest = dispatch_any!(&any, r => raster_data_hash(&[r]).data_blake3);
        return Ok(digest);
    }
    macro_rules! stack {
        ($t:ty) => {{
            let bands: Vec<Raster<$t>> =
                read_geotiff_bands(path).context("Failed to read bands")?;
            let refs: Vec<&Raster<$t>> = bands.iter().collect();
            Ok(raster_data_hash(&refs).data_blake3)
        }};
    }
    match dtype {
        "u8" => stack!(u8),
        "u16" => stack!(u16),
        "i16" => stack!(i16),
        "u32" => stack!(u32),
        "i32" => stack!(i32),
        "f32" => stack!(f32),
        "f64" => stack!(f64),
        other => bail!("unsupported recorded dtype '{other}'"),
    }
}

/// Resolve a recorded input path: as given, else relative to the recorded
/// working directory.
fn locate(source: &str, cwd: Option<&str>) -> Option<PathBuf> {
    let p = PathBuf::from(source);
    if p.is_file() {
        return Some(p);
    }
    if !p.is_absolute()
        && let Some(cwd) = cwd
    {
        let q = Path::new(cwd).join(&p);
        if q.is_file() {
            return Some(q);
        }
    }
    None
}
