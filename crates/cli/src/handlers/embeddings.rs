//! `surtgis embeddings`: foundation-model embedding stacks (AlphaEarth,
//! TESSERA, your own encoder) as first-class rasters — similarity maps,
//! PCA false colour, norms, and a look at what a file holds.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::embeddings::{
    Dequantize, PcaModel, SimilarityMetric, SimilarityParams, mean_vector, norm, similarity,
    vector_at,
};
use surtgis_core::Raster;
use surtgis_core::io::{read_geotiff_bands, write_geotiff_stack};

use crate::commands::EmbeddingsCommands;
use crate::helpers::{done, spinner, write_opts, write_result};

pub fn handle(cmd: EmbeddingsCommands, compress: bool) -> Result<()> {
    match cmd {
        EmbeddingsCommands::Info {
            input,
            bbox,
            dequantize,
        } => info(&input, parse_dq(&dequantize)?, bbox.as_deref()),
        EmbeddingsCommands::Similarity {
            input,
            bbox,
            output,
            dequantize,
            ref_lonlat,
            ref_cell,
            ref_vec,
            metric,
        } => {
            let bands = read_stack(&input, parse_dq(&dequantize)?, bbox.as_deref())?;
            let refs: Vec<&Raster<f64>> = bands.iter().collect();
            let reference = resolve_reference(&refs, ref_lonlat, ref_cell, ref_vec)?;
            let mut p = SimilarityParams::default();
            p.metric = SimilarityMetric::parse(&metric)
                .with_context(|| format!("unknown metric '{metric}' (cosine|dot|euclidean)"))?;
            let start = Instant::now();
            let out = similarity(&refs, &reference, p).context("similarity failed")?;
            write_result(&out, &output, compress)?;
            done("Similarity", &output, start.elapsed());
            Ok(())
        }
        EmbeddingsCommands::Pca {
            input,
            bbox,
            output,
            dequantize,
            components,
            samples,
            model,
            save_model,
        } => {
            let bands = read_stack(&input, parse_dq(&dequantize)?, bbox.as_deref())?;
            let refs: Vec<&Raster<f64>> = bands.iter().collect();
            let start = Instant::now();
            let fitted = match model {
                Some(path) => {
                    let s = std::fs::read_to_string(&path)
                        .with_context(|| format!("reading model {}", path.display()))?;
                    PcaModel::from_json(&s).context("parsing PCA model")?
                }
                None => {
                    PcaModel::fit(&refs, components, Some(samples)).context("PCA fit failed")?
                }
            };
            println!(
                "PCA: {} bands, {} components, fitted on {} vectors",
                fitted.n_bands(),
                fitted.components.len(),
                fitted.n_samples
            );
            for (i, (e, v)) in fitted
                .eigenvalues
                .iter()
                .zip(&fitted.variance_explained)
                .enumerate()
            {
                println!(
                    "  PC{}: eigenvalue {:.4}, variance {:.1}%",
                    i + 1,
                    e,
                    v * 100.0
                );
            }
            if let Some(path) = save_model {
                std::fs::write(&path, fitted.to_json())
                    .with_context(|| format!("writing model {}", path.display()))?;
                println!("Model saved to: {}", path.display());
            }
            let scores = fitted.project(&refs).context("PCA projection failed")?;
            let names: Vec<String> = (1..=scores.len()).map(|i| format!("PC{i}")).collect();
            let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let score_refs: Vec<&Raster<f64>> = scores.iter().collect();
            let pb = spinner("Writing output...");
            write_geotiff_stack(
                &score_refs,
                Some(&name_refs),
                &output,
                &write_opts(compress),
            )
            .context("Failed to write output")?;
            pb.finish_and_clear();
            done("PCA", &output, start.elapsed());
            Ok(())
        }
        EmbeddingsCommands::Locate {
            index,
            lon,
            lat,
            year,
        } => locate(&index, lon, lat, year),
        EmbeddingsCommands::Norm {
            input,
            bbox,
            output,
            dequantize,
        } => {
            let bands = read_stack(&input, parse_dq(&dequantize)?, bbox.as_deref())?;
            let refs: Vec<&Raster<f64>> = bands.iter().collect();
            let start = Instant::now();
            let out = norm(&refs).context("norm failed")?;
            write_result(&out, &output, compress)?;
            done("Norm", &output, start.elapsed());
            Ok(())
        }
    }
}

fn parse_dq(s: &str) -> Result<Dequantize> {
    Dequantize::parse(s).with_context(|| {
        format!("--dequantize '{s}': use none | alphaearth | linear:SCALE[,OFFSET]")
    })
}

/// Read every band of the stack. The native reader handles the usual
/// GeoTIFF layouts; planar (INTERLEAVE=BAND), ZSTD or bottom-up files —
/// the AlphaEarth tiles as published — go through the COG reader's local
/// backend. Nodata becomes NaN and `dequantize` is applied once here.
fn parse_bbox(s: &str) -> Result<[f64; 4]> {
    let v: Vec<f64> = s
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .context("--bbox must be four comma-separated numbers")?;
    if v.len() != 4 || v[0] >= v[2] || v[1] >= v[3] {
        bail!("--bbox must be min_x,min_y,max_x,max_y with min < max");
    }
    Ok([v[0], v[1], v[2], v[3]])
}

/// Read every band of the stack, or only `bbox` (stack CRS). The native
/// reader handles the usual GeoTIFF layouts; planar (INTERLEAVE=BAND),
/// ZSTD or bottom-up files — the AlphaEarth tiles as published — go
/// through the COG reader's local backend. Nodata becomes NaN and
/// `dequantize` is applied once here.
fn read_stack(
    input: &std::path::Path,
    dequantize: Dequantize,
    bbox: Option<&str>,
) -> Result<Vec<Raster<f64>>> {
    let bbox = bbox.map(parse_bbox).transpose()?;
    let pb = spinner("Reading embedding stack...");
    let native: Result<Vec<Raster<f64>>> = match bbox {
        None => read_geotiff_bands(input).map_err(|e| anyhow::anyhow!("{e}")),
        Some(b) => read_native_window(input, b),
    };
    let bands: Vec<Raster<f64>> = match native {
        Ok(b) => b,
        Err(native_err) => read_stack_cog(input, bbox).map_err(|cog_err| {
            anyhow::anyhow!(
                "reading {}: native reader: {native_err}; COG reader: {cog_err}",
                input.display()
            )
        })?,
    };
    pb.finish_and_clear();
    if bands.is_empty() {
        bail!("{}: no bands", input.display());
    }
    // Declared nodata → NaN so every operator sees one convention.
    let mut bands: Vec<Raster<f64>> = bands
        .into_iter()
        .map(|mut b| {
            if let Some(nd) = b.nodata()
                && !nd.is_nan()
            {
                b.data_mut()
                    .mapv_inplace(|v| if v == nd { f64::NAN } else { v });
                b.set_nodata(Some(f64::NAN));
            }
            b
        })
        .collect();
    dequantize.apply_stack(&mut bands);
    Ok(bands)
}

fn read_native_window(input: &std::path::Path, b: [f64; 4]) -> Result<Vec<Raster<f64>>> {
    use surtgis_core::io::window::{geotiff_info, read_geotiff_window_bands};
    let info = geotiff_info(input).map_err(|e| anyhow::anyhow!("{e}"))?;
    let Some(pw) = info.window_for_bounds(0, b[0], b[1], b[2], b[3]) else {
        bail!("--bbox lies outside the stack");
    };
    read_geotiff_window_bands::<f64, _>(input, &info, 0, &pw).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(feature = "cloud")]
fn read_stack_cog(input: &std::path::Path, bbox: Option<[f64; 4]>) -> Result<Vec<Raster<f64>>> {
    use surtgis_cloud::blocking::CogReaderBlocking;
    use surtgis_cloud::{BBox, CogReaderOptions};
    let mut r = CogReaderBlocking::open(&input.display().to_string(), CogReaderOptions::default())?;
    let m = r.metadata();
    let bb = match bbox {
        Some(b) => BBox::new(b[0], b[1], b[2], b[3]),
        None => {
            let (min_x, min_y, max_x, max_y) =
                m.geo_transform.bounds(m.width as usize, m.height as usize);
            BBox::new(min_x, min_y, max_x, max_y)
        }
    };
    let bands = r.read_bbox_bands::<f64>(&bb, None)?;
    Ok(bands)
}

#[cfg(not(feature = "cloud"))]
fn read_stack_cog(_input: &std::path::Path, _bbox: Option<[f64; 4]>) -> Result<Vec<Raster<f64>>> {
    bail!("this build has no COG reader (feature `cloud`) for planar/ZSTD/bottom-up files")
}

/// Which cell a lon/lat falls in, in the stack's CRS.
fn cell_of_lonlat(r: &Raster<f64>, lon: f64, lat: f64) -> Result<(usize, usize)> {
    let epsg = r
        .crs()
        .and_then(|c| c.epsg())
        .context("the stack has no EPSG code; use --ref-cell")?;
    let (x, y) = if epsg == 4326 {
        (lon, lat)
    } else {
        #[cfg(feature = "projections")]
        {
            surtgis_core::warp::Transformer::new(4326, epsg)
                .with_context(|| format!("no transform EPSG:4326 → EPSG:{epsg}"))?
                .forward(lon, lat)
                .context("lon/lat does not transform into the stack's CRS")?
        }
        #[cfg(not(feature = "projections"))]
        {
            bail!(
                "--ref-lonlat on a projected stack needs the `projections` feature; use --ref-cell"
            )
        }
    };
    let (col, row) = r.transform().geo_to_pixel(x, y);
    let (rows, cols) = r.shape();
    if col < 0.0 || row < 0.0 || col >= cols as f64 || row >= rows as f64 {
        bail!("reference {lon},{lat} falls outside the stack");
    }
    Ok((row as usize, col as usize))
}

fn resolve_reference(
    bands: &[&Raster<f64>],
    lonlat: Option<String>,
    cell: Option<String>,
    vec: Option<String>,
) -> Result<Vec<f64>> {
    let given = [lonlat.is_some(), cell.is_some(), vec.is_some()]
        .iter()
        .filter(|b| **b)
        .count();
    if given != 1 {
        bail!("give exactly one of --ref-lonlat, --ref-cell, --ref-vec");
    }
    if let Some(v) = vec {
        let values: Vec<f64> = v
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .context("--ref-vec must be comma-separated numbers")?;
        if values.len() != bands.len() {
            bail!(
                "--ref-vec has {} values, the stack has {} bands",
                values.len(),
                bands.len()
            );
        }
        return Ok(values);
    }
    let (row, col) = if let Some(ll) = lonlat {
        let (lon, lat) = parse_pair(&ll, "--ref-lonlat")?;
        cell_of_lonlat(bands[0], lon, lat)?
    } else {
        let (r, c) = parse_pair(&cell.unwrap(), "--ref-cell")?;
        (r as usize, c as usize)
    };
    match vector_at(bands, row, col) {
        Some(v) => {
            println!("Reference: cell ({row}, {col})");
            Ok(v)
        }
        None => {
            // Nodata at the exact cell: fall back to the 3×3 mean.
            let cells: Vec<(usize, usize)> = (-1i64..=1)
                .flat_map(|dr| (-1i64..=1).map(move |dc| (dr, dc)))
                .filter_map(|(dr, dc)| {
                    let r = row as i64 + dr;
                    let c = col as i64 + dc;
                    (r >= 0 && c >= 0).then_some((r as usize, c as usize))
                })
                .collect();
            let v = mean_vector(bands, &cells).with_context(|| {
                format!("reference cell ({row}, {col}) and its neighbours are nodata")
            })?;
            println!("Reference: 3×3 mean around cell ({row}, {col}) (centre is nodata)");
            Ok(v)
        }
    }
}

/// Tiles of the AlphaEarth index covering `(lon, lat)`: the index is a
/// GeoParquet with one row per tile and `wgs84_*` / `utm_*` bounds,
/// `year`, `utm_zone` and the `gs://` path.
fn locate(index: &std::path::Path, lon: f64, lat: f64, year: Option<i64>) -> Result<()> {
    use surtgis_core::vector::AttributeValue;
    let fc = surtgis_core::vector::read_geoparquet(index)
        .with_context(|| format!("reading index {}", index.display()))?;
    let num = |f: &surtgis_core::vector::Feature, k: &str| -> Option<f64> {
        match f.properties.get(k)? {
            AttributeValue::Float(v) => Some(*v),
            AttributeValue::Int(v) => Some(*v as f64),
            AttributeValue::String(s) => s.parse().ok(),
            _ => None,
        }
    };
    let text = |f: &surtgis_core::vector::Feature, k: &str| -> String {
        match f.properties.get(k) {
            Some(AttributeValue::String(s)) => s.clone(),
            Some(AttributeValue::Int(v)) => v.to_string(),
            Some(AttributeValue::Float(v)) => v.to_string(),
            _ => String::new(),
        }
    };
    let mut hits = 0usize;
    for f in &fc.features {
        let (Some(w), Some(s), Some(e), Some(n)) = (
            num(f, "wgs84_west"),
            num(f, "wgs84_south"),
            num(f, "wgs84_east"),
            num(f, "wgs84_north"),
        ) else {
            continue;
        };
        if lon < w || lon > e || lat < s || lat > n {
            continue;
        }
        if let Some(y) = year
            && num(f, "year").map(|v| v as i64) != Some(y)
        {
            continue;
        }
        hits += 1;
        let path = text(f, "path");
        let https = path.replacen("gs://", "https://storage.googleapis.com/", 1);
        println!(
            "{}  zone {}  utm [{} {} {} {}]",
            text(f, "year"),
            text(f, "utm_zone"),
            text(f, "utm_west"),
            text(f, "utm_south"),
            text(f, "utm_east"),
            text(f, "utm_north")
        );
        println!("  {https}");
    }
    if hits == 0 {
        bail!(
            "no tile in the index covers {lon},{lat}{}",
            year.map(|y| format!(" for {y}")).unwrap_or_default()
        );
    }
    println!(
        "{hits} tile(s). Read a window with e.g. --bbox <utm_west>,<utm_south>,<utm_east>,<utm_north> \
         and --dequantize alphaearth (int8 code, nodata -128)."
    );
    Ok(())
}

fn parse_pair(s: &str, flag: &str) -> Result<(f64, f64)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        bail!("{flag} must be two comma-separated numbers");
    }
    Ok((
        parts[0]
            .trim()
            .parse()
            .with_context(|| format!("{flag}: bad first value"))?,
        parts[1]
            .trim()
            .parse()
            .with_context(|| format!("{flag}: bad second value"))?,
    ))
}

fn info(input: &std::path::Path, dequantize: Dequantize, bbox: Option<&str>) -> Result<()> {
    let bands = read_stack(input, dequantize, bbox)?;
    let refs: Vec<&Raster<f64>> = bands.iter().collect();
    let (rows, cols) = bands[0].shape();
    println!("File: {}", input.display());
    println!("Bands (dimensions): {}", bands.len());
    println!("Dimensions: {} x {} ({} vectors)", cols, rows, rows * cols);
    if let Some(crs) = bands[0].crs() {
        println!("CRS: {crs}");
    }
    println!("Cell size: {}", bands[0].cell_size());
    let n = norm(&refs)?;
    let mut vals: Vec<f64> = n.data().iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        println!("Norm: no valid vectors");
        return Ok(());
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = vals.len();
    let mean = vals.iter().sum::<f64>() / k as f64;
    println!(
        "Valid vectors: {k} ({:.1}%)",
        100.0 * k as f64 / (rows * cols) as f64
    );
    println!(
        "L2 norm: min {:.4}, median {:.4}, mean {:.4}, max {:.4}",
        vals[0],
        vals[k / 2],
        mean,
        vals[k - 1]
    );
    let spread = (vals[k - 1] - vals[0]) / mean.max(1e-12);
    if spread < 0.05 {
        println!("Unit-normalised (norm ≈ {mean:.3} everywhere): dot product = cosine × {mean:.3}");
    } else {
        println!("Not unit-normalised: use cosine similarity, or scale before dot/euclidean");
    }
    let sample = PcaModel::fit(&refs, 3.min(bands.len()), Some(20_000))?;
    println!(
        "PCA (sample of {}): PC1 {:.1}%, PC2 {:.1}%, PC3 {:.1}% of variance",
        sample.n_samples,
        sample.variance_explained.first().copied().unwrap_or(0.0) * 100.0,
        sample.variance_explained.get(1).copied().unwrap_or(0.0) * 100.0,
        sample.variance_explained.get(2).copied().unwrap_or(0.0) * 100.0,
    );
    Ok(())
}
