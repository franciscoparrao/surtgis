//! Handlers for statistics algorithms (focal, zonal, spatial autocorrelation).

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::statistics::{
    FocalParams, FocalStatistic, ZonalStatistic, focal_statistics, global_morans_i, local_getis_ord,
};
use surtgis_algorithms::statistics::{zonal_statistics_raster_with, zonal_statistics_with};
use surtgis_core::io::read_geotiff;

use crate::commands::StatisticsCommands;
use crate::helpers::{done, read_dem, spinner, write_result};

fn parse_focal_stat(s: &str) -> FocalStatistic {
    match s.to_lowercase().as_str() {
        "mean" | "avg" => FocalStatistic::Mean,
        "std" | "stddev" | "sd" => FocalStatistic::StdDev,
        "min" => FocalStatistic::Min,
        "max" => FocalStatistic::Max,
        "range" => FocalStatistic::Range,
        "sum" => FocalStatistic::Sum,
        "count" => FocalStatistic::Count,
        "median" => FocalStatistic::Median,
        _ => {
            eprintln!("Unknown statistic '{}', using Mean", s);
            FocalStatistic::Mean
        }
    }
}

fn parse_zonal_stat(s: &str) -> ZonalStatistic {
    match s.to_lowercase().as_str() {
        "mean" | "avg" => ZonalStatistic::Mean,
        "std" | "stddev" | "sd" => ZonalStatistic::StdDev,
        "min" => ZonalStatistic::Min,
        "max" => ZonalStatistic::Max,
        "range" => ZonalStatistic::Range,
        "sum" => ZonalStatistic::Sum,
        "count" => ZonalStatistic::Count,
        "median" => ZonalStatistic::Median,
        _ => {
            eprintln!("Unknown statistic '{}', using Mean", s);
            ZonalStatistic::Mean
        }
    }
}

pub fn handle(algorithm: StatisticsCommands, compress: bool) -> Result<()> {
    match algorithm {
        StatisticsCommands::Focal {
            input,
            output,
            stat,
            radius,
            circular,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let statistic = parse_focal_stat(&stat);
            let pb = spinner(&format!("Focal {} (radius={})...", stat, radius));
            let result = focal_statistics(&raster, {
                let mut p = FocalParams::default();
                p.radius = radius;
                p.statistic = statistic;
                p.circular = circular;
                p
            })
            .context("Focal statistics failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done(&format!("Focal {}", stat), &output, start.elapsed());
        }

        StatisticsCommands::Zonal {
            input,
            output,
            zones,
            format,
            include_zero,
        } => {
            let values = read_dem(&input)?;
            let pb = spinner("Reading zones...");
            let zone_raster: surtgis_core::Raster<i32> =
                read_geotiff(&zones, None).context("Failed to read zone raster")?;
            pb.finish_and_clear();

            let start = Instant::now();
            let pb = spinner("Computing zonal statistics...");
            let background = if include_zero { None } else { Some(0) };
            let results = zonal_statistics_with(&values, &zone_raster, background)
                .context("Zonal statistics failed")?;
            pb.finish_and_clear();

            let csv = match format.as_deref() {
                Some(f) => f == "csv",
                None => output
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("csv")),
            };
            let mut zone_ids: Vec<&i32> = results.keys().collect();
            zone_ids.sort();
            let text = if csv {
                let mut csv = String::from("zone_id,count,sum,mean,std_dev,min,max,range,median\n");
                for id in &zone_ids {
                    let z = &results[id];
                    csv.push_str(&format!(
                        "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                        z.zone_id,
                        z.count,
                        z.sum,
                        z.mean,
                        z.std_dev,
                        z.min,
                        z.max,
                        z.range,
                        z.median
                    ));
                }
                csv
            } else {
                let num = |v: f64| {
                    if v.is_finite() {
                        serde_json::json!(v)
                    } else {
                        serde_json::Value::Null
                    }
                };
                let zones_json: Vec<serde_json::Value> = zone_ids
                    .iter()
                    .map(|id| {
                        let z = &results[id];
                        serde_json::json!({
                            "zone_id": z.zone_id,
                            "count": z.count,
                            "sum": num(z.sum),
                            "mean": num(z.mean),
                            "std_dev": num(z.std_dev),
                            "min": num(z.min),
                            "max": num(z.max),
                            "range": num(z.range),
                            "median": num(z.median),
                        })
                    })
                    .collect();
                let doc = serde_json::json!({
                    "input": input.display().to_string(),
                    "zones": zones.display().to_string(),
                    "background_zone": background,
                    "zones_computed": zones_json.len(),
                    "statistics": zones_json,
                });
                serde_json::to_string_pretty(&doc).expect("serialisable") + "\n"
            };
            std::fs::write(&output, &text).context("Failed to write output")?;
            println!(
                "  {} zones computed ({}{})",
                results.len(),
                if csv { "CSV" } else { "JSON" },
                if include_zero {
                    ""
                } else {
                    "; zone 0 treated as background"
                }
            );
            done("Zonal Statistics", &output, start.elapsed());
        }

        StatisticsCommands::ZonalRaster {
            input,
            output,
            zones,
            stat,
            include_zero,
        } => {
            let values = read_dem(&input)?;
            let pb = spinner("Reading zones...");
            let zone_raster: surtgis_core::Raster<i32> =
                read_geotiff(&zones, None).context("Failed to read zone raster")?;
            pb.finish_and_clear();

            let start = Instant::now();
            let statistic = parse_zonal_stat(&stat);
            let pb = spinner(&format!("Zonal {} (raster output)...", stat));
            let background = if include_zero { None } else { Some(0) };
            let result = zonal_statistics_raster_with(&values, &zone_raster, statistic, background)
                .context("Zonal statistics raster failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done(&format!("Zonal {}", stat), &output, start.elapsed());
        }

        StatisticsCommands::MoransI { input } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Computing Moran's I...");
            let result = global_morans_i(&raster).context("Moran's I failed")?;
            pb.finish_and_clear();
            println!("Moran's I: {:.6}", result.i);
            println!("  Expected: {:.6}", result.expected);
            println!("  Z-score:  {:.4}", result.z_score);
            println!("  P-value:  {:.6}", result.p_value);
            println!("  Time:     {:.2?}", start.elapsed());
        }

        StatisticsCommands::GetisOrd {
            input,
            output,
            radius,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner(&format!("Getis-Ord Gi* (radius={})...", radius));
            let result = local_getis_ord(&raster, radius).context("Getis-Ord failed")?;
            pb.finish_and_clear();
            write_result(&result.z_scores, &output, compress)?;
            // Also write p-values
            let p_path = output.with_file_name(format!(
                "{}_pvalues.tif",
                output.file_stem().unwrap_or_default().to_string_lossy()
            ));
            write_result(&result.p_values, &p_path, compress)?;
            println!("  Z-scores: {}", output.display());
            println!("  P-values: {}", p_path.display());
            done("Getis-Ord Gi*", &output, start.elapsed());
        }
    }
    Ok(())
}
