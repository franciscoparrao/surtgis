//! Streaming temporal reducers over a [`CubeSource`] (SPEC_SURTGIS_TEMPORAL_STREAMING.md).
//!
//! `temporal/{trend,phenology,anomaly,statistics}.rs` take full raster
//! stacks in memory. [`reduce_temporal`] instead drives any [`CubeSource`]
//! (an in-memory [`Cube`], or eventually a STAC-backed source) row-chunk by
//! row-chunk, so RAM scales with `chunk_rows` rather than with the number of
//! timestamps or the grid size (the R9 pattern). [`TemporalReducer`] wraps a
//! per-pixel core already used by a stack-based function above, so both
//! paths share the exact same math.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::{CubeSource, Raster, Result};

use super::trend::sens_slope_series;

/// Reduces the time series of ONE pixel to one or more output values.
///
/// The orchestrator ([`reduce_temporal`]) supplies `values` and `times`
/// already aligned by timestamp. Invalid samples arrive as `NaN` — the same
/// convention used throughout `temporal/*.rs` and by [`Cube::pixel_series`]
/// — there is no separate boolean mask.
///
/// [`Cube::pixel_series`]: surtgis_core::Cube::pixel_series
pub trait TemporalReducer: Send + Sync {
    /// Names of the output bands, e.g. `["slope"]` or `["slope", "pvalue"]`.
    fn outputs(&self) -> &[&str];
    /// Reduce one pixel's series. Returns exactly `outputs().len()` values.
    /// Only called when at least [`min_valid`](Self::min_valid) samples are
    /// finite; implementations still filter `NaN` themselves since some
    /// reducers need it in more than one pass.
    fn reduce(&self, values: &[f64], times: &[i64]) -> Vec<f64>;
    /// Minimum number of finite samples required to produce output; pixels
    /// with fewer are left as `NaN` without calling [`reduce`](Self::reduce).
    fn min_valid(&self) -> usize {
        3
    }
}

/// Robust linear-trend reducer: the Theil–Sen slope (median of pairwise
/// slopes). Wraps [`sens_slope_series`], the same per-pixel core used by the
/// in-memory [`sens_slope`](super::sens_slope), so streaming and in-memory
/// results are bit-identical for the same series.
pub struct TheilSenTrend;

impl TemporalReducer for TheilSenTrend {
    fn outputs(&self) -> &[&str] {
        &["slope"]
    }

    fn reduce(&self, values: &[f64], times: &[i64]) -> Vec<f64> {
        let t: Vec<f64> = times.iter().map(|&x| x as f64).collect();
        vec![sens_slope_series(values, &t).unwrap_or(f64::NAN)]
    }

    fn min_valid(&self) -> usize {
        2
    }
}

/// Applies `reducer` to every pixel of `source`, chunk-by-chunk, producing
/// one output [`Raster`] per `reducer.outputs()` entry.
///
/// Peak RAM is `O(chunk_rows · cols · n_times)` plus the output rasters
/// (`O(rows · cols · n_outputs)`, unavoidable since every pixel needs a
/// value) — it does **not** scale with the source's total row count. This
/// is the streaming counterpart to computing a stack-based reducer
/// (`temporal::linear_trend`, etc.) directly on an in-memory `Vec<Raster>`.
pub fn reduce_temporal<S, R>(source: &S, reducer: &R, chunk_rows: usize) -> Result<Vec<Raster<f64>>>
where
    S: CubeSource,
    R: TemporalReducer,
{
    let (rows, cols) = source.shape();
    let times = source.times();
    let n_times = times.len();
    let n_outputs = reducer.outputs().len();
    let min_valid = reducer.min_valid();

    let mut out_flats: Vec<Vec<f64>> = (0..n_outputs)
        .map(|_| vec![f64::NAN; rows * cols])
        .collect();

    let mut row0 = 0;
    while row0 < rows {
        let chunk = source.chunk(row0, chunk_rows)?;
        let chunk_rows_n = chunk.rows;

        let views: Vec<_> = (0..n_times).map(|t| chunk.view(t, 0, 1)).collect();

        let row_results: Vec<Vec<f64>> = (0..chunk_rows_n)
            .into_par_iter()
            .map(|r| {
                let mut row_out = vec![f64::NAN; cols * n_outputs];
                let mut values = vec![0.0f64; n_times];
                for c in 0..cols {
                    for (t, v) in views.iter().enumerate() {
                        values[t] = v[[r, c]];
                    }
                    let valid = values.iter().filter(|v| v.is_finite()).count();
                    if valid >= min_valid {
                        let out = reducer.reduce(&values, times);
                        debug_assert_eq!(
                            out.len(),
                            n_outputs,
                            "reducer returned wrong output count"
                        );
                        row_out[c * n_outputs..(c + 1) * n_outputs].copy_from_slice(&out);
                    }
                }
                row_out
            })
            .collect();

        for (r, row_out) in row_results.into_iter().enumerate() {
            let global_row = row0 + r;
            for c in 0..cols {
                for (o, out_flat) in out_flats.iter_mut().enumerate() {
                    out_flat[global_row * cols + c] = row_out[c * n_outputs + o];
                }
            }
        }

        row0 += chunk_rows_n;
    }

    let transform = source.transform();
    Ok(out_flats
        .into_iter()
        .map(|flat| {
            let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
            let mut r = Raster::from_array(arr);
            r.set_transform(transform);
            r.set_nodata(Some(f64::NAN));
            r
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::{Cube, GeoTransform};

    fn make_linear_cube(
        n: usize,
        rows: usize,
        cols: usize,
        slope: f64,
        intercept: f64,
    ) -> Cube<f64> {
        let mut slices = Vec::with_capacity(n);
        for t in 0..n {
            let mut r: Raster<f64> = Raster::new(rows, cols);
            r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
            let value = intercept + slope * t as f64;
            for row in 0..rows {
                for col in 0..cols {
                    r.set(row, col, value).unwrap();
                }
            }
            slices.push(r);
        }
        let times: Vec<i64> = (0..n as i64).collect();
        Cube::from_slices(times, vec!["v".into()], slices).unwrap()
    }

    #[test]
    fn theil_sen_recovers_exact_slope() {
        let cube = make_linear_cube(8, 4, 4, 2.5, 10.0);
        let outputs = reduce_temporal(&cube, &TheilSenTrend, 2).unwrap();
        assert_eq!(outputs.len(), 1);
        for &v in outputs[0].data().iter() {
            assert!((v - 2.5).abs() < 1e-10, "slope {} != 2.5", v);
        }
    }

    #[test]
    fn streaming_matches_in_memory_bit_identical() {
        let cube = make_linear_cube(10, 6, 5, -1.75, 3.0);

        // Streaming path, one row at a time (forces multiple chunks).
        let streaming = reduce_temporal(&cube, &TheilSenTrend, 1).unwrap();

        // In-memory path via the stack-based `sens_slope` (same per-pixel
        // core, `sens_slope_series`, since the cube uses index times 0..n-1).
        let rasters: Vec<Raster<f64>> = (0..cube.n_times())
            .map(|t| cube.slice(t, 0).unwrap().clone())
            .collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();
        let in_memory = super::super::trend::sens_slope(&refs).unwrap();

        assert_eq!(streaming[0].shape(), in_memory.shape());
        for (a, b) in streaming[0].data().iter().zip(in_memory.data().iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "streaming {} != in-memory {}",
                a,
                b
            );
        }
    }

    #[test]
    fn gaps_respect_min_valid() {
        let mut cube = make_linear_cube(5, 1, 1, 4.0, 0.0);
        // Knock out all but one sample so valid count drops below min_valid=2.
        {
            let slices_mut: Vec<Raster<f64>> = (0..cube.n_times())
                .map(|t| cube.slice(t, 0).unwrap().clone())
                .collect();
            let mut only_first = slices_mut;
            for r in only_first.iter_mut().skip(1) {
                r.set(0, 0, f64::NAN).unwrap();
            }
            let times: Vec<i64> = (0..only_first.len() as i64).collect();
            cube = Cube::from_slices(times, vec!["v".into()], only_first).unwrap();
        }

        let outputs = reduce_temporal(&cube, &TheilSenTrend, 8).unwrap();
        assert!(
            outputs[0].data()[[0, 0]].is_nan(),
            "single sample: no pair to slope"
        );
    }

    #[test]
    fn gaps_still_recover_slope_with_enough_valid_samples() {
        let mut cube = make_linear_cube(6, 1, 1, 3.0, 1.0);
        {
            let mut slices: Vec<Raster<f64>> = (0..cube.n_times())
                .map(|t| cube.slice(t, 0).unwrap().clone())
                .collect();
            // Drop two interior samples (cloud gaps); slope must still be
            // recovered exactly from the remaining points on a perfect line.
            slices[1].set(0, 0, f64::NAN).unwrap();
            slices[3].set(0, 0, f64::NAN).unwrap();
            let times: Vec<i64> = (0..slices.len() as i64).collect();
            cube = Cube::from_slices(times, vec!["v".into()], slices).unwrap();
        }

        let outputs = reduce_temporal(&cube, &TheilSenTrend, 8).unwrap();
        assert!((outputs[0].data()[[0, 0]] - 3.0).abs() < 1e-10);
    }
}
