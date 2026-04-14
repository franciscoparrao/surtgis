//! Handlers for statistics algorithms (focal, zonal, spatial autocorrelation).

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::statistics::{
    focal_statistics, global_morans_i, local_getis_ord, zonal_statistics,
    FocalParams, FocalStatistic, ZonalStatistic,
};
use surtgis_algorithms::statistics::zonal::zonal_statistics_raster;
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
            input, output, stat, radius, circular,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let statistic = parse_focal_stat(&stat);
            let pb = spinner(&format!("Focal {} (radius={})...", stat, radius));
            let result = focal_statistics(
                &raster,
                FocalParams {
                    radius,
                    statistic,
                    circular,
                },
            )
            .context("Focal statistics failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done(&format!("Focal {}", stat), &output, start.elapsed());
        }

        StatisticsCommands::Zonal {
            input, output, zones,
        } => {
            let values = read_dem(&input)?;
            let pb = spinner("Reading zones...");
            let zone_raster: surtgis_core::Raster<i32> =
                read_geotiff(&zones, None).context("Failed to read zone raster")?;
            pb.finish_and_clear();

            let start = Instant::now();
            let pb = spinner("Computing zonal statistics...");
            let results = zonal_statistics(&values, &zone_raster)
                .context("Zonal statistics failed")?;
            pb.finish_and_clear();

            // Write as CSV (zone_id,count,mean,std,min,max,median)
            let mut csv = String::from("zone_id,count,sum,mean,std_dev,min,max,range,median\n");
            let mut zone_ids: Vec<&i32> = results.keys().collect();
            zone_ids.sort();
            for id in &zone_ids {
                let z = &results[id];
                csv.push_str(&format!(
                    "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                    z.zone_id, z.count, z.sum, z.mean, z.std_dev,
                    z.min, z.max, z.range, z.median
                ));
            }
            std::fs::write(&output, &csv).context("Failed to write output")?;
            println!("  {} zones computed", results.len());
            done("Zonal Statistics", &output, start.elapsed());
        }

        StatisticsCommands::ZonalRaster {
            input, output, zones, stat,
        } => {
            let values = read_dem(&input)?;
            let pb = spinner("Reading zones...");
            let zone_raster: surtgis_core::Raster<i32> =
                read_geotiff(&zones, None).context("Failed to read zone raster")?;
            pb.finish_and_clear();

            let start = Instant::now();
            let statistic = parse_zonal_stat(&stat);
            let pb = spinner(&format!("Zonal {} (raster output)...", stat));
            let result = zonal_statistics_raster(&values, &zone_raster, statistic)
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
            input, output, radius,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner(&format!("Getis-Ord Gi* (radius={})...", radius));
            let result = local_getis_ord(&raster, radius)
                .context("Getis-Ord failed")?;
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
