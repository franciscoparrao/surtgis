//! Handler for imagery / spectral index subcommands.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::Instant;

use surtgis_algorithms::imagery::{
    band_math_binary, bsi, burn_severity_classify, cloud_mask_scl, dnbr, evi, evi2, gndvi,
    index_builder, median_composite, mndwi, msavi, nbr, ndbi, ndmi, ndre, ndsi, ndvi, ndwi,
    ngrdi, reci, reclassify, savi, EviParams, ReclassifyParams, SaviParams,
};

use crate::commands::ImageryCommands;
use crate::helpers::{
    done, parse_band_assignments, parse_band_math_op, parse_reclass_entry, parse_scl_classes,
    read_dem, write_result,
};

pub fn handle(algorithm: ImageryCommands, compress: bool) -> Result<()> {
    match algorithm {
        ImageryCommands::Ndvi { nir, red, output } => {
            let nir_r = read_dem(&nir)?;
            let red_r = read_dem(&red)?;
            let start = Instant::now();
            let result = ndvi(&nir_r, &red_r).context("Failed to calculate NDVI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDVI", &output, elapsed);
        }

        ImageryCommands::Ndwi { green, nir, output } => {
            let green_r = read_dem(&green)?;
            let nir_r = read_dem(&nir)?;
            let start = Instant::now();
            let result = ndwi(&green_r, &nir_r).context("Failed to calculate NDWI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDWI", &output, elapsed);
        }

        ImageryCommands::Mndwi {
            green,
            swir,
            output,
        } => {
            let green_r = read_dem(&green)?;
            let swir_r = read_dem(&swir)?;
            let start = Instant::now();
            let result =
                mndwi(&green_r, &swir_r).context("Failed to calculate MNDWI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("MNDWI", &output, elapsed);
        }

        ImageryCommands::Nbr { nir, swir, output } => {
            let nir_r = read_dem(&nir)?;
            let swir_r = read_dem(&swir)?;
            let start = Instant::now();
            let result = nbr(&nir_r, &swir_r).context("Failed to calculate NBR")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NBR", &output, elapsed);
        }

        ImageryCommands::Savi {
            nir,
            red,
            output,
            l_factor,
        } => {
            let nir_r = read_dem(&nir)?;
            let red_r = read_dem(&red)?;
            let start = Instant::now();
            let result = savi(&nir_r, &red_r, SaviParams { l_factor })
                .context("Failed to calculate SAVI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("SAVI", &output, elapsed);
        }

        ImageryCommands::Evi {
            nir,
            red,
            blue,
            output,
        } => {
            let nir_r = read_dem(&nir)?;
            let red_r = read_dem(&red)?;
            let blue_r = read_dem(&blue)?;
            let start = Instant::now();
            let result = evi(&nir_r, &red_r, &blue_r, EviParams::default())
                .context("Failed to calculate EVI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("EVI", &output, elapsed);
        }

        ImageryCommands::Bsi {
            swir,
            red,
            nir,
            blue,
            output,
        } => {
            let swir_r = read_dem(&swir)?;
            let red_r = read_dem(&red)?;
            let nir_r = read_dem(&nir)?;
            let blue_r = read_dem(&blue)?;
            let start = Instant::now();
            let result = bsi(&swir_r, &red_r, &nir_r, &blue_r)
                .context("Failed to calculate BSI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("BSI", &output, elapsed);
        }

        ImageryCommands::BandMath { a, b, output, op } => {
            let op = parse_band_math_op(&op)?;
            let a_r = read_dem(&a)?;
            let b_r = read_dem(&b)?;
            let start = Instant::now();
            let result = band_math_binary(&a_r, &b_r, op)
                .context("Failed to perform band math")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Band math", &output, elapsed);
        }

        ImageryCommands::Calc {
            expression,
            band,
            output,
        } => {
            let assignments = parse_band_assignments(&band)?;
            let mut rasters: Vec<(String, surtgis_core::Raster<f64>)> = Vec::new();
            for (name, path) in &assignments {
                let r = read_dem(path)?;
                rasters.push((name.clone(), r));
            }
            let band_refs: HashMap<&str, &surtgis_core::Raster<f64>> = rasters
                .iter()
                .map(|(name, r)| (name.as_str(), r))
                .collect();
            let start = Instant::now();
            let result = index_builder(&expression, &band_refs)
                .context("Failed to evaluate expression")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Calc", &output, elapsed);
        }

        ImageryCommands::Evi2 { nir, red, output } => {
            let nir_r = read_dem(&nir)?;
            let red_r = read_dem(&red)?;
            let start = Instant::now();
            let result = evi2(&nir_r, &red_r).context("Failed to calculate EVI2")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("EVI2", &output, elapsed);
        }

        ImageryCommands::Gndvi { nir, green, output } => {
            let nir_r = read_dem(&nir)?;
            let green_r = read_dem(&green)?;
            let start = Instant::now();
            let result = gndvi(&nir_r, &green_r).context("Failed to calculate GNDVI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("GNDVI", &output, elapsed);
        }

        ImageryCommands::Ngrdi { green, red, output } => {
            let green_r = read_dem(&green)?;
            let red_r = read_dem(&red)?;
            let start = Instant::now();
            let result = ngrdi(&green_r, &red_r).context("Failed to calculate NGRDI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NGRDI", &output, elapsed);
        }

        ImageryCommands::Reci { nir, red_edge, output } => {
            let nir_r = read_dem(&nir)?;
            let re_r = read_dem(&red_edge)?;
            let start = Instant::now();
            let result = reci(&nir_r, &re_r).context("Failed to calculate RECI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("RECI", &output, elapsed);
        }

        ImageryCommands::Ndre { nir, red_edge, output } => {
            let nir_r = read_dem(&nir)?;
            let re_r = read_dem(&red_edge)?;
            let start = Instant::now();
            let result = ndre(&nir_r, &re_r).context("Failed to calculate NDRE")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDRE", &output, elapsed);
        }

        ImageryCommands::Ndsi { green, swir, output } => {
            let green_r = read_dem(&green)?;
            let swir_r = read_dem(&swir)?;
            let start = Instant::now();
            let result = ndsi(&green_r, &swir_r).context("Failed to calculate NDSI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDSI", &output, elapsed);
        }

        ImageryCommands::Ndmi { nir, swir, output } => {
            let nir_r = read_dem(&nir)?;
            let swir_r = read_dem(&swir)?;
            let start = Instant::now();
            let result = ndmi(&nir_r, &swir_r).context("Failed to calculate NDMI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDMI", &output, elapsed);
        }

        ImageryCommands::Ndbi { swir, nir, output } => {
            let swir_r = read_dem(&swir)?;
            let nir_r = read_dem(&nir)?;
            let start = Instant::now();
            let result = ndbi(&swir_r, &nir_r).context("Failed to calculate NDBI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDBI", &output, elapsed);
        }

        ImageryCommands::Msavi { nir, red, output } => {
            let nir_r = read_dem(&nir)?;
            let red_r = read_dem(&red)?;
            let start = Instant::now();
            let result = msavi(&nir_r, &red_r).context("Failed to calculate MSAVI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("MSAVI", &output, elapsed);
        }

        ImageryCommands::Reclassify {
            input,
            output,
            class,
            default,
        } => {
            let classes: Vec<_> = class
                .iter()
                .map(|s| parse_reclass_entry(s))
                .collect::<Result<Vec<_>>>()?;
            let default_value: f64 = if default.to_lowercase() == "nan" {
                f64::NAN
            } else {
                default.parse().context("Invalid default value")?
            };
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = reclassify(
                &raster,
                ReclassifyParams {
                    classes,
                    default_value,
                },
            )
            .context("Failed to reclassify")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Reclassify", &output, elapsed);
        }

        ImageryCommands::MedianComposite { input, output } => {
            if input.len() < 2 {
                anyhow::bail!("median-composite requires at least 2 input rasters");
            }
            let rasters: Vec<surtgis_core::Raster<f64>> = input
                .iter()
                .map(|p| {
                    read_dem(p)
                })
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
            let start = Instant::now();
            let result =
                median_composite(&refs).context("Failed to compute median composite")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            println!("  {} input rasters composited", input.len());
            done("Median composite", &output, elapsed);
        }

        ImageryCommands::Dnbr { pre_nir, pre_swir, post_nir, post_swir, output, severity_output } => {
            let pn = read_dem(&pre_nir)?;
            let ps = read_dem(&pre_swir)?;
            let on = read_dem(&post_nir)?;
            let os = read_dem(&post_swir)?;
            let start = Instant::now();
            let result = dnbr(&pn, &ps, &on, &os).context("Failed to compute dNBR")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("dNBR", &output, elapsed);

            if let Some(sev_path) = severity_output {
                let severity = burn_severity_classify(&result)
                    .context("Failed to classify burn severity")?;
                write_result(&severity, &sev_path, compress)?;
                println!("Burn severity saved to: {}", sev_path.display());
            }
        }

        ImageryCommands::CloudMask {
            input,
            scl,
            output,
            keep,
        } => {
            let data = read_dem(&input)?;
            let scl_r = read_dem(&scl)?;
            let classes = parse_scl_classes(&keep)?;
            let start = Instant::now();
            let result = cloud_mask_scl(&data, &scl_r, &classes)
                .context("Failed to apply cloud mask")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Cloud mask", &output, elapsed);
        }
    }

    Ok(())
}
