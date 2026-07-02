//! Parity tests between the batch and streaming (`WindowAlgorithm`)
//! implementations of the terrain focal-window algorithms.
//!
//! Both paths now share a single per-cell kernel, so their outputs must
//! agree exactly on interior cells (away from the image border by the
//! kernel radius, where the border policies legitimately differ: batch
//! leaves NaN, streaming sees NaN halo padding).
//!
//! The streaming path is driven through an emulation of
//! `surtgis_core::streaming::StripProcessor`'s chunking and NaN halo
//! padding, with a small strip size so several chunk seams fall inside
//! the test raster.

use ndarray::Array2;
use surtgis_algorithms::terrain::{
    CircularVarianceParams, CircularVarianceStreaming, ConvergenceParams, ConvergenceStreaming,
    DevParams, DevStreaming, DiffFromMeanParams, DiffFromMeanStreaming, EastnessStreaming,
    NormalDeviationParams, NormalDeviationStreaming, NorthnessStreaming, PercentElevRangeParams,
    PercentElevRangeStreaming, SphericalStdDevParams, SphericalStdDevStreaming, TpiParams,
    TpiStreaming, TriParams, TriStreaming, VrmParams, VrmStreaming, circular_variance_aspect,
    convergence_index, dev, diff_from_mean_elev, eastness, normal_vector_deviation, northness,
    percent_elev_range, spherical_std_dev, tpi, tri, vrm,
};
use surtgis_core::raster::Raster;
use surtgis_core::{GeoTransform, WindowAlgorithm};

const ROWS: usize = 40;
const COLS: usize = 37;
const CHUNK_ROWS: usize = 7; // forces several strip seams inside the raster

/// Deterministic synthetic DEM with varied slopes/aspects, square 5 m cells
/// (cell size != 1 so cell-size-dependent kernels are actually exercised),
/// and a small NaN hole to exercise nodata skipping.
fn synthetic_dem() -> Raster<f64> {
    let mut dem = Raster::new(ROWS, COLS);
    dem.set_transform(GeoTransform::new(0.0, ROWS as f64 * 5.0, 5.0, -5.0));
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = c as f64;
            let y = r as f64;
            let z = 30.0 * (x * 0.31).sin()
                + 22.0 * (y * 0.23).cos()
                + 0.4 * x * y / 10.0
                + ((r * 7 + c * 13) % 17) as f64 * 1.5;
            dem.set(r, c, z).unwrap();
        }
    }
    // NaN hole
    dem.set(20, 18, f64::NAN).unwrap();
    dem
}

/// Emulate `StripProcessor::process`: strip-by-strip with NaN halo padding.
fn run_streaming<A: WindowAlgorithm>(alg: &A, dem: &Raster<f64>) -> Array2<f64> {
    let (rows, cols) = dem.shape();
    let radius = alg.kernel_radius();
    let data = dem.data();
    let cs_x = dem.transform().pixel_width.abs();
    let cs_y = dem.transform().pixel_height.abs();

    let mut full = Array2::from_elem((rows, cols), f64::NAN);
    let mut out_start = 0usize;
    while out_start < rows {
        let out_end = (out_start + CHUNK_ROWS).min(rows);
        let n_out = out_end - out_start;
        let in_start = out_start.saturating_sub(radius);
        let in_end = (out_end + radius).min(rows);

        let padded_rows = n_out + 2 * radius;
        let mut input = Array2::from_elem((padded_rows, cols), f64::NAN);
        let top_pad = radius.saturating_sub(out_start);
        let copy_rows = (in_end - in_start).min(padded_rows - top_pad);
        input
            .slice_mut(ndarray::s![top_pad..top_pad + copy_rows, ..])
            .assign(&data.slice(ndarray::s![in_start..in_start + copy_rows, ..]));

        let mut output = Array2::from_elem((n_out, cols), f64::NAN);
        alg.process_chunk(&input, &mut output, dem.nodata(), cs_x, cs_y);
        full.slice_mut(ndarray::s![out_start..out_end, ..])
            .assign(&output);
        out_start = out_end;
    }
    full
}

/// Compare batch and streaming outputs on interior cells (at least `border`
/// cells away from the image edge).
fn assert_interior_match(batch: &Raster<f64>, streaming: &Array2<f64>, border: usize, tol: f64) {
    let (rows, cols) = batch.shape();
    let mut compared = 0usize;
    for r in border..rows - border {
        for c in border..cols - border {
            let b = batch.get(r, c).unwrap();
            let s = streaming[[r, c]];
            match (b.is_nan(), s.is_nan()) {
                (true, true) => continue,
                (false, false) => {
                    assert!(
                        (b - s).abs() <= tol,
                        "mismatch at ({r},{c}): batch={b}, streaming={s}"
                    );
                    compared += 1;
                }
                _ => panic!("NaN mismatch at ({r},{c}): batch={b}, streaming={s}"),
            }
        }
    }
    assert!(compared > 100, "too few comparable cells: {compared}");
}

#[test]
fn tpi_parity() {
    let dem = synthetic_dem();
    let batch = tpi(&dem, TpiParams { radius: 2 }).unwrap();
    let stream = run_streaming(&TpiStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 2, 0.0);
}

#[test]
fn tri_parity() {
    let dem = synthetic_dem();
    let batch = tri(&dem, TriParams { radius: 2 }).unwrap();
    let stream = run_streaming(&TriStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 2, 0.0);
}

#[test]
fn dev_parity() {
    let dem = synthetic_dem();
    let batch = dev(&dem, DevParams { radius: 3 }).unwrap();
    let stream = run_streaming(&DevStreaming { radius: 3 }, &dem);
    assert_interior_match(&batch, &stream, 3, 0.0);
}

#[test]
fn diff_from_mean_parity() {
    let dem = synthetic_dem();
    let batch = diff_from_mean_elev(&dem, DiffFromMeanParams { radius: 2 }).unwrap();
    let stream = run_streaming(&DiffFromMeanStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 2, 0.0);
}

#[test]
fn percent_elev_range_parity() {
    let dem = synthetic_dem();
    let batch = percent_elev_range(&dem, PercentElevRangeParams { radius: 2 }).unwrap();
    let stream = run_streaming(&PercentElevRangeStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 2, 0.0);
}

#[test]
fn circular_variance_parity() {
    let dem = synthetic_dem();
    let batch = circular_variance_aspect(&dem, CircularVarianceParams { radius: 2 }).unwrap();
    let stream = run_streaming(&CircularVarianceStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 3, 0.0); // kernel_radius = radius + 1
}

#[test]
fn convergence_parity() {
    let dem = synthetic_dem();
    let batch = convergence_index(&dem, ConvergenceParams { radius: 1 }).unwrap();
    let stream = run_streaming(&ConvergenceStreaming { radius: 1 }, &dem);
    assert_interior_match(&batch, &stream, 2, 0.0); // kernel_radius = radius + 1
}

#[test]
fn normal_vector_deviation_parity() {
    // Exercises the cell-size handling: the streaming path previously
    // ignored cell size (assumed 1.0) while batch used dem.cell_size().
    let dem = synthetic_dem();
    let batch = normal_vector_deviation(&dem, NormalDeviationParams { radius: 2 }).unwrap();
    let stream = run_streaming(&NormalDeviationStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 3, 0.0); // kernel_radius = radius + 1
}

#[test]
fn spherical_std_dev_parity() {
    // Exercises the cell-size handling (see normal_vector_deviation_parity).
    let dem = synthetic_dem();
    let batch = spherical_std_dev(&dem, SphericalStdDevParams { radius: 2 }).unwrap();
    let stream = run_streaming(&SphericalStdDevStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 3, 0.0); // kernel_radius = radius + 1
}

#[test]
fn vrm_parity() {
    let dem = synthetic_dem();
    let batch = vrm(&dem, VrmParams { radius: 2 }).unwrap();
    let stream = run_streaming(&VrmStreaming { radius: 2 }, &dem);
    assert_interior_match(&batch, &stream, 3, 0.0); // kernel_radius = radius + 1
}

#[test]
fn northness_parity() {
    // Batch goes through aspect() (normalises the bearing to [0, 2π) before
    // cos), streaming skips the normalisation — identical up to FP rounding.
    let dem = synthetic_dem();
    let batch = northness(&dem).unwrap();
    let stream = run_streaming(&NorthnessStreaming, &dem);
    assert_interior_match(&batch, &stream, 1, 1e-12);
}

#[test]
fn eastness_parity() {
    let dem = synthetic_dem();
    let batch = eastness(&dem).unwrap();
    let stream = run_streaming(&EastnessStreaming, &dem);
    assert_interior_match(&batch, &stream, 1, 1e-12);
}
