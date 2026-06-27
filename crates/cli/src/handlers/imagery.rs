//! Handler for imagery / spectral index subcommands.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::Instant;

use surtgis_algorithms::imagery::{
    Dos1Params, EviParams, IrMadParams, LandsatToaParams, ReclassifyParams, S2ReflectanceParams,
    SaviParams, band_math_binary, bsi, burn_severity_classify, cloud_mask_scl,
    dn_to_reflectance_s2, dn_to_surface_reflectance_landsat_c2, dn_to_toa_landsat, dnbr, dos1,
    dual_pol_water_index, evi, evi2, feather_mosaic, gndvi, histogram_match, index_builder, ir_mad,
    lee_filter, linear_to_db, mad, median_composite, mndwi, moment_match, msavi, nbr, ndbi, ndmi,
    ndre, ndsi, ndvi, ndwi, ngrdi, reci, reclassify, sar_water_mask, savi,
};
use surtgis_algorithms::pansharpening::{brovey, gram_schmidt, pca_pansharpen};

use crate::commands::{
    CalibrateCommands, ChangeDetectionCommands, ColorBalanceCommands, ImageryCommands,
    PansharpenCommands,
};
use crate::helpers::{
    done, parse_band_assignments, parse_band_math_op, parse_reclass_entry, parse_scl_classes,
    read_dem, write_result, write_result_u8,
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
            let result = mndwi(&green_r, &swir_r).context("Failed to calculate MNDWI")?;
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
            let result =
                bsi(&swir_r, &red_r, &nir_r, &blue_r).context("Failed to calculate BSI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("BSI", &output, elapsed);
        }

        ImageryCommands::BandMath { a, b, output, op } => {
            let op = parse_band_math_op(&op)?;
            let a_r = read_dem(&a)?;
            let b_r = read_dem(&b)?;
            let start = Instant::now();
            let result = band_math_binary(&a_r, &b_r, op).context("Failed to perform band math")?;
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
            let band_refs: HashMap<&str, &surtgis_core::Raster<f64>> =
                rasters.iter().map(|(name, r)| (name.as_str(), r)).collect();
            let start = Instant::now();
            let result =
                index_builder(&expression, &band_refs).context("Failed to evaluate expression")?;
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

        ImageryCommands::Reci {
            nir,
            red_edge,
            output,
        } => {
            let nir_r = read_dem(&nir)?;
            let re_r = read_dem(&red_edge)?;
            let start = Instant::now();
            let result = reci(&nir_r, &re_r).context("Failed to calculate RECI")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("RECI", &output, elapsed);
        }

        ImageryCommands::Ndre {
            nir,
            red_edge,
            output,
        } => {
            let nir_r = read_dem(&nir)?;
            let re_r = read_dem(&red_edge)?;
            let start = Instant::now();
            let result = ndre(&nir_r, &re_r).context("Failed to calculate NDRE")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("NDRE", &output, elapsed);
        }

        ImageryCommands::Ndsi {
            green,
            swir,
            output,
        } => {
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
                .map(|p| read_dem(p))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
            let start = Instant::now();
            let result = median_composite(&refs).context("Failed to compute median composite")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            println!("  {} input rasters composited", input.len());
            done("Median composite", &output, elapsed);
        }

        ImageryCommands::Dnbr {
            pre_nir,
            pre_swir,
            post_nir,
            post_swir,
            output,
            severity_output,
        } => {
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
                let severity =
                    burn_severity_classify(&result).context("Failed to classify burn severity")?;
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
            let result =
                cloud_mask_scl(&data, &scl_r, &classes).context("Failed to apply cloud mask")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Cloud mask", &output, elapsed);
        }

        ImageryCommands::Calibrate { action } => handle_calibrate(action, compress)?,

        ImageryCommands::Pansharpen { action } => handle_pansharpen(action, compress)?,

        ImageryCommands::ChangeDetection { action } => {
            handle_change_detection(action, compress)?;
        }

        ImageryCommands::ColorBalance { action } => handle_color_balance(action, compress)?,

        ImageryCommands::MosaicFeather { output, inputs } => {
            if inputs.len() < 2 {
                anyhow::bail!("mosaic-feather: need at least 2 --input rasters");
            }
            let rasters = inputs
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<_> = rasters.iter().collect();
            let start = Instant::now();
            let result = feather_mosaic(&refs).context("feather-mosaic failed")?;
            write_result(&result, &output, compress)?;
            done(
                &format!("Feather mosaic ({} inputs)", inputs.len()),
                &output,
                start.elapsed(),
            );
        }

        ImageryCommands::Stack { output, bands } => {
            if !matches!(bands.len(), 1 | 3 | 4) {
                anyhow::bail!(
                    "imagery stack supports 1, 3 or 4 bands; got {}",
                    bands.len()
                );
            }
            let rasters = bands
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
            let start = Instant::now();
            // NativeGeoTiffOptions, not io::GeoTiffOptions: under the `gdal`
            // feature the latter is the GDAL variant, but the multi-band writer
            // is native-only and needs the native options type.
            let opts = if compress {
                Some(surtgis_core::io::NativeGeoTiffOptions {
                    compression: "DEFLATE".to_string(),
                })
            } else {
                None
            };
            surtgis_core::io::write_geotiff_multiband(&refs, &output, opts)
                .context("Stack write failed")?;
            done(
                &format!("Stack ({} bands)", bands.len()),
                &output,
                start.elapsed(),
            );
        }

        ImageryCommands::SarDb { input, output } => {
            let r = read_dem(&input)?;
            let start = Instant::now();
            let result = linear_to_db(&r).context("SAR linear→dB failed")?;
            write_result(&result, &output, compress)?;
            done("SAR backscatter (dB)", &output, start.elapsed());
        }

        ImageryCommands::SarWaterIndex {
            co_pol,
            cross_pol,
            output,
        } => {
            let co = read_dem(&co_pol)?;
            let cross = read_dem(&cross_pol)?;
            let start = Instant::now();
            let result =
                dual_pol_water_index(&co, &cross).context("SAR dual-pol water index failed")?;
            write_result(&result, &output, compress)?;
            done("SAR dual-pol water index", &output, start.elapsed());
        }

        ImageryCommands::SarWaterMask {
            input,
            output,
            threshold,
            water_above,
        } => {
            let r = read_dem(&input)?;
            let start = Instant::now();
            // water_below is the default; --water-above inverts it.
            let mask =
                sar_water_mask(&r, threshold, !water_above).context("SAR water mask failed")?;
            write_result_u8(&mask, &output, compress)?;
            done("SAR water mask", &output, start.elapsed());
        }

        ImageryCommands::SarLee {
            input,
            output,
            window_size,
            looks,
        } => {
            let r = read_dem(&input)?;
            let start = Instant::now();
            let filtered =
                lee_filter(&r, window_size, looks).context("SAR Lee speckle filter failed")?;
            write_result(&filtered, &output, compress)?;
            done("SAR Lee speckle filter", &output, start.elapsed());
        }
    }

    Ok(())
}

fn handle_pansharpen(action: PansharpenCommands, compress: bool) -> Result<()> {
    enum Method {
        Brovey,
        Pca,
        GramSchmidt,
    }
    let (method, pan, bands, output_dir, prefix) = match action {
        PansharpenCommands::Brovey {
            pan,
            bands,
            output_dir,
            prefix,
        } => (Method::Brovey, pan, bands, output_dir, prefix),
        PansharpenCommands::Pca {
            pan,
            bands,
            output_dir,
            prefix,
        } => (Method::Pca, pan, bands, output_dir, prefix),
        PansharpenCommands::GramSchmidt {
            pan,
            bands,
            output_dir,
            prefix,
        } => (Method::GramSchmidt, pan, bands, output_dir, prefix),
    };

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let pan_r = read_dem(&pan).context("reading pan")?;
    let ms_rasters = bands
        .iter()
        .map(|p| read_dem(p).with_context(|| format!("reading band {}", p.display())))
        .collect::<Result<Vec<_>>>()?;
    let ms_refs: Vec<_> = ms_rasters.iter().collect();

    let start = Instant::now();
    let method_name = match method {
        Method::Brovey => "Brovey",
        Method::Pca => "PCA",
        Method::GramSchmidt => "Gram-Schmidt",
    };
    let results = match method {
        Method::Brovey => brovey(&pan_r, &ms_refs).context("Brovey failed")?,
        Method::Pca => pca_pansharpen(&pan_r, &ms_refs).context("PCA pansharpen failed")?,
        Method::GramSchmidt => gram_schmidt(&pan_r, &ms_refs).context("Gram-Schmidt failed")?,
    };
    for (idx, out) in results.iter().enumerate() {
        let path = output_dir.join(format!("{}_band{:02}.tif", prefix, idx + 1));
        write_result(out, &path, compress)?;
    }
    done(
        &format!("{} pansharpening", method_name),
        &output_dir.join(format!("{}_band*.tif", prefix)),
        start.elapsed(),
    );
    Ok(())
}

fn handle_calibrate(action: CalibrateCommands, compress: bool) -> Result<()> {
    match action {
        CalibrateCommands::LandsatToa {
            input,
            output,
            multiplicative,
            additive,
            sun_elevation,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = dn_to_toa_landsat(
                &raster,
                LandsatToaParams {
                    multiplicative,
                    additive,
                    sun_elevation_deg: sun_elevation,
                },
            )
            .context("Landsat TOA calibration failed")?;
            write_result(&result, &output, compress)?;
            done("Landsat TOA", &output, start.elapsed());
        }
        CalibrateCommands::LandsatSrC2 { input, output } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = dn_to_surface_reflectance_landsat_c2(&raster)
                .context("Landsat C2 SR calibration failed")?;
            write_result(&result, &output, compress)?;
            done("Landsat C2 SR", &output, start.elapsed());
        }
        CalibrateCommands::S2 {
            input,
            output,
            quantification,
            offset,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = dn_to_reflectance_s2(
                &raster,
                S2ReflectanceParams {
                    quantification_value: quantification,
                    offset,
                },
            )
            .context("Sentinel-2 reflectance calibration failed")?;
            write_result(&result, &output, compress)?;
            done("S2 reflectance", &output, start.elapsed());
        }
        CalibrateCommands::Dos1 {
            input,
            output,
            quantile,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = dos1(
                &raster,
                Dos1Params {
                    dark_object_quantile: quantile,
                },
            )
            .context("DOS1 failed")?;
            write_result(&result, &output, compress)?;
            done("DOS1", &output, start.elapsed());
        }
    }
    Ok(())
}

fn handle_color_balance(action: ColorBalanceCommands, compress: bool) -> Result<()> {
    match action {
        ColorBalanceCommands::Histogram {
            source,
            reference,
            output,
        } => {
            let src = read_dem(&source)?;
            let refr = read_dem(&reference)?;
            let start = Instant::now();
            let result = histogram_match(&src, &refr).context("histogram_match failed")?;
            write_result(&result, &output, compress)?;
            done("Histogram match", &output, start.elapsed());
        }
        ColorBalanceCommands::Moments {
            source,
            reference,
            output,
        } => {
            let src = read_dem(&source)?;
            let refr = read_dem(&reference)?;
            let start = Instant::now();
            let result = moment_match(&src, &refr).context("moment_match failed")?;
            write_result(&result, &output, compress)?;
            done("Moment match", &output, start.elapsed());
        }
    }
    Ok(())
}

fn handle_change_detection(action: ChangeDetectionCommands, compress: bool) -> Result<()> {
    match action {
        ChangeDetectionCommands::Mad {
            output_dir,
            t1,
            t2,
            prefix,
        } => {
            if t1.len() != t2.len() {
                anyhow::bail!(
                    "MAD: --t1 has {} bands, --t2 has {} (must match)",
                    t1.len(),
                    t2.len()
                );
            }
            std::fs::create_dir_all(&output_dir)
                .with_context(|| format!("failed to create {}", output_dir.display()))?;
            let t1_r = t1
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading t1 band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let t2_r = t2
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading t2 band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let t1_refs: Vec<_> = t1_r.iter().collect();
            let t2_refs: Vec<_> = t2_r.iter().collect();
            let start = Instant::now();
            let result = mad(&t1_refs, &t2_refs).context("MAD failed")?;
            for (i, m) in result.mad.iter().enumerate() {
                let path = output_dir.join(format!("{}_variate{:02}.tif", prefix, i + 1));
                write_result(m, &path, compress)?;
            }
            // Pretty-print correlations as a summary line.
            let rhos: Vec<String> = result
                .correlations
                .iter()
                .map(|r| format!("{:.4}", r))
                .collect();
            println!(
                "MAD canonical correlations (MAD_1 → MAD_{}): {}",
                result.correlations.len(),
                rhos.join(", ")
            );
            done(
                &format!("MAD ({} variates)", result.mad.len()),
                &output_dir.join(format!("{}_variate*.tif", prefix)),
                start.elapsed(),
            );
        }
        ChangeDetectionCommands::IrMad {
            output_dir,
            t1,
            t2,
            max_iter,
            tol,
            regularisation,
            prefix,
        } => {
            if t1.len() != t2.len() {
                anyhow::bail!(
                    "IR-MAD: --t1 has {} bands, --t2 has {} (must match)",
                    t1.len(),
                    t2.len()
                );
            }
            std::fs::create_dir_all(&output_dir)
                .with_context(|| format!("failed to create {}", output_dir.display()))?;
            let t1_r = t1
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading t1 band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let t2_r = t2
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading t2 band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let t1_refs: Vec<_> = t1_r.iter().collect();
            let t2_refs: Vec<_> = t2_r.iter().collect();
            let start = Instant::now();
            let result = ir_mad(
                &t1_refs,
                &t2_refs,
                IrMadParams {
                    max_iter,
                    tol,
                    regularisation,
                },
            )
            .context("IR-MAD failed")?;
            for (i, m) in result.mad.iter().enumerate() {
                let path = output_dir.join(format!("{}_variate{:02}.tif", prefix, i + 1));
                write_result(m, &path, compress)?;
            }
            let weights_path = output_dir.join(format!("{}_nochange_prob.tif", prefix));
            write_result(&result.weights, &weights_path, compress)?;
            let rhos: Vec<String> = result
                .correlations
                .iter()
                .map(|r| format!("{:.4}", r))
                .collect();
            println!(
                "IR-MAD converged in {} iterations. Correlations: {}",
                result.n_iter,
                rhos.join(", ")
            );
            done(
                &format!("IR-MAD ({} variates + weights)", result.mad.len()),
                &output_dir.join(format!("{}_*.tif", prefix)),
                start.elapsed(),
            );
        }
    }
    Ok(())
}
