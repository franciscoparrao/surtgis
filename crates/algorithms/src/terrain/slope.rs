//! Slope calculation from DEMs
//!
//! Calculates the rate of change of elevation using the Horn (1981) method,
//! which uses a 3x3 neighborhood to compute partial derivatives.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Units for slope output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlopeUnits {
    /// Degrees (0-90)
    #[default]
    Degrees,
    /// Percent (0-infinity, typically 0-100+)
    Percent,
    /// Radians (0-π/2)
    Radians,
}

/// Parameters for slope calculation
#[derive(Debug, Clone)]
pub struct SlopeParams {
    /// Output units
    pub units: SlopeUnits,
    /// Vertical exaggeration applied to elevations before the gradient
    /// (GDAL/ArcGIS convention: `z' = z_factor * z`). Default 1.0.
    ///
    /// Rasters with a geographic CRS get automatic per-row metric cell
    /// sizes — do NOT use the legacy `111320` cell-size hack, which relied
    /// on the pre-0.17 reciprocal semantics.
    pub z_factor: f64,
}

impl Default for SlopeParams {
    fn default() -> Self {
        Self {
            units: SlopeUnits::Degrees,
            z_factor: 1.0,
        }
    }
}

/// Slope algorithm
#[derive(Debug, Clone, Default)]
pub struct Slope;

impl Algorithm for Slope {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = SlopeParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Slope"
    }

    fn description(&self) -> &'static str {
        "Calculate slope (rate of change of elevation) from a DEM using Horn's method"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        slope(&input, params)
    }
}

/// Calculate slope from a DEM
///
/// Uses Horn's (1981) method with a 3x3 neighborhood:
/// ```text
/// a b c
/// d e f
/// g h i
/// ```
///
/// dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize)
/// dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize)
/// slope = atan(sqrt(dz/dx² + dz/dy²))
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Slope calculation parameters
///
/// # Returns
/// Raster with slope values in the specified units
pub fn slope(dem: &Raster<f64>, params: SlopeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // GDAL/ArcGIS semantics: z_factor scales the ELEVATIONS (the gradient
    // numerator), not the cell size. Geographic rasters (CRS in degrees)
    // get automatic per-row metric cell sizes — no z_factor hack needed.
    let zf = params.z_factor;
    let cell_sizes = super::spheroidal_grid::CellSizes::for_dem(dem);

    let units = params.units;
    let data = dem
        .data()
        .as_slice()
        .ok_or_else(|| Error::Other("raster data must be contiguous".into()))?;

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        if row == 0 || row == rows - 1 {
            return; // edge rows stay NaN
        }
        let top = &data[(row - 1) * cols..row * cols];
        let mid = &data[row * cols..(row + 1) * cols];
        let bot = &data[(row + 1) * cols..(row + 2) * cols];

        let (dx, dy) = cell_sizes.at_row(row);
        let (eight_dx, eight_dy) = (8.0 * dx, 8.0 * dy);

        for col in 1..cols - 1 {
            // Get center value
            let e = mid[col];
            if e.is_nan() || nodata.is_some_and(|nd| (e - nd).abs() < f64::EPSILON) {
                continue; // stays NaN
            }

            // 3x3 neighborhood
            let (a, b, c) = (top[col - 1], top[col], top[col + 1]);
            let (d, f) = (mid[col - 1], mid[col + 1]);
            let (g, h, i) = (bot[col - 1], bot[col], bot[col + 1]);

            if a.is_nan()
                || b.is_nan()
                || c.is_nan()
                || d.is_nan()
                || f.is_nan()
                || g.is_nan()
                || h.is_nan()
                || i.is_nan()
            {
                continue; // stays NaN
            }

            // Horn's method (z_factor scales z, per GDAL convention)
            let dz_dx = zf * ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_dx;
            let dz_dy = zf * ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / eight_dy;

            let grad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();

            out_row[col] = match units {
                SlopeUnits::Degrees => grad.atan().to_degrees(),
                // tan(atan(g)) = g — no need for either call
                SlopeUnits::Percent => grad * 100.0,
                SlopeUnits::Radians => grad.atan(),
            };
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming slope calculator implementing `WindowAlgorithm`.
///
/// Processes a DEM strip-by-strip with bounded memory.
/// Uses the same Horn (1981) 3×3 method as `slope()`.
#[derive(Debug, Clone)]
pub struct SlopeStreaming {
    /// Output units for the slope angle (degrees, radians, or percent).
    pub units: SlopeUnits,
    /// Vertical exaggeration applied to elevations before the gradient.
    pub z_factor: f64,
}

impl Default for SlopeStreaming {
    fn default() -> Self {
        Self {
            units: SlopeUnits::Degrees,
            z_factor: 1.0,
        }
    }
}

impl SlopeStreaming {
    /// Process a single output row given its already-resolved cell sizes
    /// (`eight_dx`/`eight_dy`). Shared by the constant-cell-size path
    /// (`process_chunk`) and the per-row geographic-correction path
    /// (`process_chunk_geo`), which differ only in how those two values
    /// are derived.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn process_row(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        r: usize,
        ir: usize,
        in_rows: usize,
        cols: usize,
        eight_dx: f64,
        eight_dy: f64,
    ) {
        if ir == 0 || ir >= in_rows - 1 {
            // Edge row — fill with NaN
            for c in 0..cols {
                output[[r, c]] = f64::NAN;
            }
            return;
        }

        // z_factor scales z (GDAL convention); dx and dy used separately.
        let zf = self.z_factor;

        for c in 0..cols {
            if c == 0 || c >= cols - 1 {
                output[[r, c]] = f64::NAN;
                continue;
            }

            let e = input[[ir, c]];
            if e.is_nan() || nodata.map_or(false, |nd| (e - nd).abs() < f64::EPSILON) {
                output[[r, c]] = f64::NAN;
                continue;
            }

            let a = input[[ir - 1, c - 1]];
            let b = input[[ir - 1, c]];
            let cv = input[[ir - 1, c + 1]];
            let d = input[[ir, c - 1]];
            let f = input[[ir, c + 1]];
            let g = input[[ir + 1, c - 1]];
            let h = input[[ir + 1, c]];
            let i = input[[ir + 1, c + 1]];

            if [a, b, cv, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                output[[r, c]] = f64::NAN;
                continue;
            }

            let dz_dx = zf * ((cv + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_dx;
            let dz_dy = zf * ((g + 2.0 * h + i) - (a + 2.0 * b + cv)) / eight_dy;
            let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();

            output[[r, c]] = match self.units {
                SlopeUnits::Degrees => slope_rad.to_degrees(),
                SlopeUnits::Percent => slope_rad.tan() * 100.0,
                SlopeUnits::Radians => slope_rad,
            };
        }
    }
}

impl surtgis_core::WindowAlgorithm for SlopeStreaming {
    fn kernel_radius(&self) -> usize {
        1 // 3×3 kernel
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        cell_size_y: f64,
    ) {
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        let eight_dx = 8.0 * cell_size_x;
        let eight_dy = 8.0 * cell_size_y.abs();

        for r in 0..out_rows {
            let ir = r + radius; // input row corresponding to output row r
            self.process_row(
                input, output, nodata, r, ir, in_rows, cols, eight_dx, eight_dy,
            );
        }
    }

    /// REG-1 fix: on geographic (lon/lat) rasters, the batch path
    /// (`slope()`) applies a per-row latitude-dependent metric cell size
    /// via `CellSizes::for_dem`/`at_row` (see `spheroidal_grid.rs`).
    /// Before this override, the streaming path silently used the raw
    /// degree spacing as if it were meters — same DEM, correct slope in
    /// batch mode, silently wrong slope in streaming (i.e. via the CLI,
    /// which auto-selects streaming for large files). This mirrors the
    /// batch correction using `geo_ctx` (built by `StripProcessor` from
    /// the raster's CRS/GeoTransform) to recover each row's absolute
    /// latitude and re-derive metric `(dx, dy)` for it.
    fn process_chunk_geo(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        cell_size_y: f64,
        geo_ctx: Option<surtgis_core::GeoRowContext>,
    ) {
        let Some(ctx) = geo_ctx else {
            // Not geographic: fall back to the constant-cell-size path.
            return self.process_chunk(input, output, nodata, cell_size_x, cell_size_y);
        };

        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        for r in 0..out_rows {
            let ir = r + radius;
            let abs_row = ctx.row_offset + r;
            let lat = ctx.origin_y + (abs_row as f64 + 0.5) * ctx.pixel_height;
            let dims = super::spheroidal_grid::cell_dimensions(
                lat,
                cell_size_x,
                cell_size_y,
                &super::spheroidal_grid::SpheroidalParams::default(),
            );
            let eight_dx = 8.0 * dims.dx;
            let eight_dy = 8.0 * dims.dy;
            self.process_row(
                input, output, nodata, r, ir, in_rows, cols, eight_dx, eight_dy,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dem() -> Raster<f64> {
        // Create a simple tilted plane: z = x + y
        let mut dem = Raster::new(10, 10);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }
        dem
    }

    /// z_factor now follows the GDAL/ArcGIS convention: it scales the
    /// elevations (z\' = zf*z), so slope(zf=2) on gradient g == atan(2g).
    /// The pre-0.17 semantics scaled the CELL SIZE (the reciprocal).
    #[test]
    fn test_z_factor_gdal_semantics() {
        let n = 12;
        let mut dem = Raster::new(n, n);
        dem.set_transform(surtgis_core::GeoTransform::new(
            0.0,
            n as f64 * 10.0,
            10.0,
            -10.0,
        ));
        for r in 0..n {
            for c in 0..n {
                dem.set(r, c, c as f64 * 5.0).unwrap(); // dz/dx = 0.5
            }
        }
        let s1 = slope(
            &dem,
            SlopeParams {
                z_factor: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        let expected = (2.0f64 * 0.5).atan().to_degrees(); // 45°
        let got = s1.get(n / 2, n / 2).unwrap();
        assert!(
            (got - expected).abs() < 1e-9,
            "zf=2 on 0.5 gradient must be atan(1.0)=45°, got {}",
            got
        );
    }

    /// Rasters with a geographic CRS get automatic per-row metric cell
    /// sizes: a pure E-W gradient in degrees at 45°S must produce the
    /// slope computed with dx = one_degree_lon(45°) meters, and the same
    /// DEM WITHOUT a CRS must keep the raw (degree-unit) computation.
    #[test]
    fn test_geographic_auto_correction() {
        use super::super::spheroidal_grid::{SpheroidalParams, cell_dimensions};

        let n = 10;
        let px = 0.001; // ~100 m at the equator
        let lat0 = -45.0;
        let mut dem = Raster::new(n, n);
        dem.set_transform(surtgis_core::GeoTransform::new(-71.0, lat0, px, -px));
        for r in 0..n {
            for c in 0..n {
                dem.set(r, c, c as f64 * 20.0).unwrap(); // 20 m per column
            }
        }

        // Without CRS: degrees treated as linear units (documented behavior)
        let raw = slope(&dem, SlopeParams::default()).unwrap();
        let raw_val = raw.get(n / 2, n / 2).unwrap();
        assert!(raw_val > 89.0, "without CRS, dz=20 over dx=0.001 is ~90°");

        // With geographic CRS: per-row metric dx
        dem.set_crs(Some(surtgis_core::crs::CRS::from_epsg(4326)));
        let geo = slope(&dem, SlopeParams::default()).unwrap();
        let row = n / 2;
        let lat = lat0 + (row as f64 + 0.5) * (-px);
        let dims = cell_dimensions(lat, px, px, &SpheroidalParams::default());
        let expected = (20.0f64 / dims.dx).atan().to_degrees();
        let got = geo.get(row, n / 2).unwrap();
        assert!(
            (got - expected).abs() < 1e-6,
            "geographic slope at 45°S: expected {} got {}",
            expected,
            got
        );
        // Sanity: ~20 m over ~79 m => ~14°, nothing like the raw 90°
        assert!(
            got > 10.0 && got < 20.0,
            "plausible metric slope, got {}",
            got
        );
    }

    /// Rectangular (dx != dy) cells: each gradient component uses its own
    /// cell size instead of silently assuming squares.
    #[test]
    fn test_rectangular_cells() {
        let n = 10;
        let mut dem = Raster::new(n, n);
        // dx = 10, dy = 20
        dem.set_transform(surtgis_core::GeoTransform::new(
            0.0,
            n as f64 * 20.0,
            10.0,
            -20.0,
        ));
        for r in 0..n {
            for c in 0..n {
                dem.set(r, c, r as f64 * 10.0).unwrap(); // dz/row = 10
            }
        }
        let s1 = slope(&dem, SlopeParams::default()).unwrap();
        // N-S gradient: 10 m per 20 m cell = 0.5 => atan(0.5)
        let expected = 0.5f64.atan().to_degrees();
        let got = s1.get(n / 2, n / 2).unwrap();
        assert!(
            (got - expected).abs() < 1e-9,
            "dy=20 must divide the N-S gradient: expected {} got {}",
            expected,
            got
        );
    }

    #[test]
    fn test_slope_flat() {
        let mut dem: Raster<f64> = Raster::filled(10, 10, 100.0);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = slope(&dem, SlopeParams::default()).unwrap();

        // Interior cells should have zero slope
        let val = result.get(5, 5).unwrap();
        assert!(
            val.abs() < 0.001,
            "Expected ~0 slope for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_slope_tilted() {
        let dem = create_test_dem();
        let result = slope(&dem, SlopeParams::default()).unwrap();

        // All interior cells should have the same slope (constant gradient)
        let val1 = result.get(3, 3).unwrap();
        let val2 = result.get(5, 5).unwrap();

        assert!(
            (val1 - val2).abs() < 0.001,
            "Expected uniform slope, got {} vs {}",
            val1,
            val2
        );
    }

    #[test]
    fn test_slope_units() {
        let dem = create_test_dem();

        let deg = slope(
            &dem,
            SlopeParams {
                units: SlopeUnits::Degrees,
                z_factor: 1.0,
            },
        )
        .unwrap();
        let rad = slope(
            &dem,
            SlopeParams {
                units: SlopeUnits::Radians,
                z_factor: 1.0,
            },
        )
        .unwrap();
        let pct = slope(
            &dem,
            SlopeParams {
                units: SlopeUnits::Percent,
                z_factor: 1.0,
            },
        )
        .unwrap();

        let deg_val = deg.get(5, 5).unwrap();
        let rad_val = rad.get(5, 5).unwrap();
        let pct_val = pct.get(5, 5).unwrap();

        // Verify unit conversions
        assert!(
            (deg_val - rad_val.to_degrees()).abs() < 0.001,
            "Degree/radian mismatch"
        );
        assert!(
            (pct_val - rad_val.tan() * 100.0).abs() < 0.001,
            "Percent mismatch"
        );
    }
}
