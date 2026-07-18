//! Zonal statistics
//!
//! Computes statistics for each zone defined by an integer zone raster.
//! Zones are identified by unique integer values in the zone raster.

use std::collections::HashMap;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Available zonal statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZonalStatistic {
    /// Arithmetic mean of the zone's values.
    Mean,
    /// Population standard deviation of the zone's values.
    StdDev,
    /// Minimum value in the zone.
    Min,
    /// Maximum value in the zone.
    Max,
    /// Difference between the zone's maximum and minimum.
    Range,
    /// Sum of the zone's values.
    Sum,
    /// Number of valid cells in the zone.
    Count,
    /// Median value of the zone.
    Median,
    /// Most frequent value in the zone (mode).
    Majority,
    /// Least frequent value in the zone.
    Minority,
    /// Count of distinct values in the zone.
    Variety,
}

/// Result of zonal statistics for one zone
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ZonalResult {
    /// Integer identifier of the zone.
    pub zone_id: i32,
    /// Number of valid cells in the zone.
    pub count: usize,
    /// Sum of the zone's values.
    pub sum: f64,
    /// Arithmetic mean of the zone's values.
    pub mean: f64,
    /// Population standard deviation of the zone's values.
    pub std_dev: f64,
    /// Minimum value in the zone.
    pub min: f64,
    /// Maximum value in the zone.
    pub max: f64,
    /// Difference between the zone's maximum and minimum.
    pub range: f64,
    /// Median value of the zone.
    pub median: f64,
    /// Most frequent value in the zone (mode). NaN if the zone is empty.
    pub majority: f64,
    /// Least frequent value in the zone (anti-mode). NaN if the zone is empty.
    pub minority: f64,
    /// Number of distinct values in the zone.
    pub variety: usize,
}

/// Check that the value and zone rasters share the same geotransform and
/// (lenient, EPSG-only) CRS.
///
/// `zonal_statistics` combines a `Raster<f64>` (values) with a `Raster<i32>`
/// (zones), so the shared-type `surtgis_core::raster::check_aligned` helper
/// cannot be used directly (it requires a homogeneous `&[&Raster<T>]`
/// slice). This mirrors its tolerance and CRS-lenience semantics for the
/// two-type case. Shape is checked separately by the caller via
/// `Error::SizeMismatch`.
fn check_geo_alignment(values: &Raster<f64>, zones: &Raster<i32>) -> Result<()> {
    const REL_TOL: f64 = 1e-9;
    let gt_v = values.transform().to_gdal();
    let gt_z = zones.transform().to_gdal();
    for (a, b) in gt_v.iter().zip(gt_z.iter()) {
        let tol = REL_TOL * a.abs().max(b.abs()).max(1.0);
        if (a - b).abs() > tol {
            return Err(Error::Misaligned {
                reason: format!(
                    "geotransform mismatch: values raster has {gt_v:?} but zones \
                     raster has {gt_z:?} (GDAL coefficient order)"
                ),
            });
        }
    }

    if let (Some(a), Some(b)) = (
        values.crs().and_then(|c| c.epsg()),
        zones.crs().and_then(|c| c.epsg()),
    ) && a != b
    {
        return Err(Error::Misaligned {
            reason: format!("CRS mismatch: values raster is EPSG:{a} but zones raster is EPSG:{b}"),
        });
    }

    Ok(())
}

/// Compute the mode (majority), anti-mode (minority), and cardinality (variety)
/// of a set of values by grouping on exact bit-pattern equality.
///
/// Values are treated as categorical class codes (e.g. integer class values
/// stored as `f64`), so grouping uses `to_bits()` rather than a float
/// tolerance comparison — this avoids NaN/partial-ordering pitfalls and
/// matches how the rest of the zone raster represents discrete classes.
/// `values` must not contain NaN (callers already filter NaN as nodata).
fn categorical_stats(values: &[f64]) -> (f64, f64, usize) {
    let mut freq: HashMap<u64, (f64, usize)> = HashMap::new();
    for &v in values {
        let entry = freq.entry(v.to_bits()).or_insert((v, 0));
        entry.1 += 1;
    }

    let majority = freq
        .values()
        .max_by_key(|(_, count)| *count)
        .map(|(v, _)| *v)
        .unwrap_or(f64::NAN);
    let minority = freq
        .values()
        .min_by_key(|(_, count)| *count)
        .map(|(v, _)| *v)
        .unwrap_or(f64::NAN);
    let variety = freq.len();

    (majority, minority, variety)
}

/// Compute zonal statistics
///
/// For each unique zone in the zone raster, computes statistics from
/// the corresponding cells in the value raster.
///
/// # Arguments
/// * `values` - Input raster with values to analyze
/// * `zones` - Zone raster (integer identifiers)
///
/// # Returns
/// HashMap mapping zone_id → ZonalResult
pub fn zonal_statistics(
    values: &Raster<f64>,
    zones: &Raster<i32>,
) -> Result<HashMap<i32, ZonalResult>> {
    let (rows_v, cols_v) = values.shape();
    let (rows_z, cols_z) = zones.shape();

    if rows_v != rows_z || cols_v != cols_z {
        return Err(Error::SizeMismatch {
            er: rows_v,
            ec: cols_v,
            ar: rows_z,
            ac: cols_z,
        });
    }

    check_geo_alignment(values, zones)?;

    // Collect values per zone
    let mut zone_values: HashMap<i32, Vec<f64>> = HashMap::new();

    for row in 0..rows_v {
        for col in 0..cols_v {
            let zone = unsafe { zones.get_unchecked(row, col) };
            let val = unsafe { values.get_unchecked(row, col) };

            if zone == 0 || val.is_nan() {
                continue; // Skip nodata zones (0) and NaN values
            }

            zone_values.entry(zone).or_default().push(val);
        }
    }

    // Compute statistics per zone
    let mut results = HashMap::new();

    for (zone_id, mut vals) in zone_values {
        if vals.is_empty() {
            continue;
        }

        let count = vals.len();
        let sum: f64 = vals.iter().sum();
        let mean = sum / count as f64;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count as f64;
        let std_dev = var.sqrt();

        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = vals[0];
        let max = vals[vals.len() - 1];
        let range = max - min;

        let median = if count % 2 == 0 {
            (vals[count / 2 - 1] + vals[count / 2]) / 2.0
        } else {
            vals[count / 2]
        };

        let (majority, minority, variety) = categorical_stats(&vals);

        results.insert(
            zone_id,
            ZonalResult {
                zone_id,
                count,
                sum,
                mean,
                std_dev,
                min,
                max,
                range,
                median,
                majority,
                minority,
                variety,
            },
        );
    }

    Ok(results)
}

/// Create a raster where each cell is replaced by the zonal statistic
/// for its zone.
///
/// # Arguments
/// * `values` - Input value raster
/// * `zones` - Zone raster
/// * `statistic` - Which statistic to map back
///
/// # Returns
/// Raster where each cell contains the zone statistic
pub fn zonal_statistics_raster(
    values: &Raster<f64>,
    zones: &Raster<i32>,
    statistic: ZonalStatistic,
) -> Result<Raster<f64>> {
    let stats = zonal_statistics(values, zones)?;
    let (rows, cols) = values.shape();

    let mut output = values.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));

    for row in 0..rows {
        for col in 0..cols {
            let zone = unsafe { zones.get_unchecked(row, col) };
            if let Some(zr) = stats.get(&zone) {
                let val = match statistic {
                    ZonalStatistic::Mean => zr.mean,
                    ZonalStatistic::StdDev => zr.std_dev,
                    ZonalStatistic::Min => zr.min,
                    ZonalStatistic::Max => zr.max,
                    ZonalStatistic::Range => zr.range,
                    ZonalStatistic::Sum => zr.sum,
                    ZonalStatistic::Count => zr.count as f64,
                    ZonalStatistic::Median => zr.median,
                    ZonalStatistic::Majority => zr.majority,
                    ZonalStatistic::Minority => zr.minority,
                    ZonalStatistic::Variety => zr.variety as f64,
                };
                output.set(row, col, val).unwrap();
            } else {
                output.set(row, col, f64::NAN).unwrap();
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_zonal_basic() {
        let mut values = Raster::new(4, 4);
        values.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        let mut zones: Raster<i32> = Raster::new(4, 4);
        zones.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));

        // Zone 1: top-left 2x2, Zone 2: top-right 2x2
        for row in 0..4 {
            for col in 0..4 {
                values.set(row, col, (row * 4 + col) as f64).unwrap();
                let zone = if col < 2 { 1 } else { 2 };
                zones.set(row, col, zone).unwrap();
            }
        }

        let results = zonal_statistics(&values, &zones).unwrap();
        assert_eq!(results.len(), 2);

        let z1 = results.get(&1).unwrap();
        assert_eq!(z1.count, 8);

        let z2 = results.get(&2).unwrap();
        assert_eq!(z2.count, 8);
    }

    #[test]
    fn test_zonal_uniform_zone() {
        let values = Raster::filled(5, 5, 10.0_f64);
        let zones: Raster<i32> = Raster::filled(5, 5, 1);

        let results = zonal_statistics(&values, &zones).unwrap();
        let z1 = results.get(&1).unwrap();

        assert_eq!(z1.count, 25);
        assert!((z1.mean - 10.0).abs() < 1e-10);
        assert!(z1.std_dev.abs() < 1e-10);
        assert!((z1.min - 10.0).abs() < 1e-10);
        assert!((z1.max - 10.0).abs() < 1e-10);
        assert!(z1.range.abs() < 1e-10);
    }

    #[test]
    fn test_zonal_dimension_mismatch() {
        let values: Raster<f64> = Raster::new(5, 5);
        let zones: Raster<i32> = Raster::new(3, 3);
        let result = zonal_statistics(&values, &zones);
        assert!(result.is_err());
    }

    #[test]
    fn test_zonal_skip_nodata() {
        let mut values = Raster::new(3, 3);
        values.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut zones: Raster<i32> = Raster::filled(3, 3, 1);
        zones.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        for row in 0..3 {
            for col in 0..3 {
                values.set(row, col, 5.0).unwrap();
            }
        }
        values.set(1, 1, f64::NAN).unwrap(); // NaN cell

        let results = zonal_statistics(&values, &zones).unwrap();
        let z1 = results.get(&1).unwrap();
        assert_eq!(z1.count, 8); // 9 - 1 NaN = 8
    }

    #[test]
    fn test_zonal_raster_output() {
        let mut values = Raster::new(4, 4);
        values.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        let mut zones: Raster<i32> = Raster::new(4, 4);
        zones.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));

        for row in 0..4 {
            for col in 0..4 {
                let zone = if row < 2 { 1 } else { 2 };
                zones.set(row, col, zone).unwrap();
                values
                    .set(row, col, if row < 2 { 10.0 } else { 20.0 })
                    .unwrap();
            }
        }

        let result = zonal_statistics_raster(&values, &zones, ZonalStatistic::Mean).unwrap();
        assert!((result.get(0, 0).unwrap() - 10.0).abs() < 1e-10);
        assert!((result.get(3, 3).unwrap() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_categorical_stats_majority_minority_variety() {
        // [1,1,1,2,2,3] -> majority=1 (count 3), minority=3 (count 1), variety=3
        let vals = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let (majority, minority, variety) = categorical_stats(&vals);
        assert_eq!(majority, 1.0);
        assert_eq!(minority, 3.0);
        assert_eq!(variety, 3);
    }

    #[test]
    fn test_zonal_majority_minority_variety_not_mean() {
        // CR-4 regression: Majority/Minority/Variety used to silently fall back
        // to the zone mean. A class raster with values [1,1,1,2,2,3] has
        // mean=1.6666..., but majority/minority/variety must be categorical.
        let mut values = Raster::new(2, 3);
        values.set_transform(GeoTransform::new(0.0, 2.0, 1.0, -1.0));
        let mut zones: Raster<i32> = Raster::filled(2, 3, 1);
        zones.set_transform(GeoTransform::new(0.0, 2.0, 1.0, -1.0));

        let class_vals = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let mut idx = 0;
        for row in 0..2 {
            for col in 0..3 {
                values.set(row, col, class_vals[idx]).unwrap();
                idx += 1;
            }
        }

        let results = zonal_statistics(&values, &zones).unwrap();
        let z1 = results.get(&1).unwrap();
        assert!((z1.mean - 1.6666666666666667).abs() < 1e-10);
        assert_eq!(z1.majority, 1.0);
        assert_eq!(z1.minority, 3.0);
        assert_eq!(z1.variety, 3);

        let majority_raster =
            zonal_statistics_raster(&values, &zones, ZonalStatistic::Majority).unwrap();
        let minority_raster =
            zonal_statistics_raster(&values, &zones, ZonalStatistic::Minority).unwrap();
        let variety_raster =
            zonal_statistics_raster(&values, &zones, ZonalStatistic::Variety).unwrap();

        assert_eq!(majority_raster.get(0, 0).unwrap(), 1.0);
        assert_eq!(minority_raster.get(0, 0).unwrap(), 3.0);
        assert_eq!(variety_raster.get(0, 0).unwrap(), 3.0);
    }
}
