//! Hillshade (shaded relief) calculation
//!
//! Creates a shaded relief visualization from a DEM based on
//! illumination angle and direction.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Parameters for hillshade calculation
#[derive(Debug, Clone)]
pub struct HillshadeParams {
    /// Sun azimuth in degrees (0 = North, clockwise)
    pub azimuth: f64,
    /// Sun altitude in degrees above horizon (0-90)
    pub altitude: f64,
    /// Vertical exaggeration applied to elevations before the gradient
    /// (GDAL/ArcGIS convention: `z' = z_factor * z`). Default 1.0.
    ///
    /// Rasters with a geographic CRS get automatic per-row metric cell
    /// sizes — do NOT use the legacy `111320` cell-size hack, which relied
    /// on the pre-0.17 reciprocal semantics.
    pub z_factor: f64,
    /// Output range: false = 0-255, true = 0.0-1.0
    pub normalized: bool,
}

impl Default for HillshadeParams {
    fn default() -> Self {
        Self {
            azimuth: 315.0, // NW illumination (standard)
            altitude: 45.0, // 45° above horizon
            z_factor: 1.0,
            normalized: false,
        }
    }
}

/// Hillshade algorithm
#[derive(Debug, Clone, Default)]
pub struct Hillshade;

impl Algorithm for Hillshade {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = HillshadeParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Hillshade"
    }

    fn description(&self) -> &'static str {
        "Calculate shaded relief from a DEM"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        hillshade(&input, params)
    }
}

/// Calculate hillshade from a DEM
///
/// Uses the standard algorithm based on slope, aspect, and illumination geometry.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Hillshade parameters (azimuth, altitude, z-factor)
///
/// # Returns
/// Raster with hillshade values (0-255 or 0.0-1.0)
pub fn hillshade(dem: &Raster<f64>, params: HillshadeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // GDAL/ArcGIS semantics: z_factor scales the elevations. Geographic
    // rasters get automatic per-row metric cell sizes.
    let zf = params.z_factor;
    let cell_sizes = super::spheroidal_grid::CellSizes::for_dem(dem);

    // Pre-compute illumination angles in radians
    let azimuth_rad = (360.0 - params.azimuth + 90.0).to_radians();
    let zenith_rad = (90.0 - params.altitude).to_radians();
    let cos_zenith = zenith_rad.cos();
    let sin_zenith = zenith_rad.sin();
    let (sin_az, cos_az) = azimuth_rad.sin_cos();

    let normalized = params.normalized;

    let data = dem
        .data()
        .as_slice()
        .ok_or_else(|| Error::Other("raster data must be contiguous".into()))?;

    // The classic formulation computes atan, cos, sin, atan2 and cos per
    // cell. All of it collapses algebraically. With x = dz_dx, y = dz_dy,
    // g = sqrt(x² + y²):
    //
    //   slope  = atan(g)          →  cos s = 1/√(1+g²),  sin s = g/√(1+g²)
    //   aspect = atan2(y, −x)     →  cos aspect = −x/g,  sin aspect = y/g
    //   cos(az − aspect) = cos az·cos aspect + sin az·sin aspect
    //                    = (−x·cos az + y·sin az) / g
    //
    //   shade = cosθz·cos s + sinθz·sin s·cos(az − aspect)
    //         = (cosθz + sinθz·(y·sin az − x·cos az)) / √(1 + x² + y²)
    //
    // Zero transcendental calls per cell (one sqrt + one division), and the
    // flat case (g = 0) needs no special branch: the formula reduces to
    // cosθz on its own. This matches gdaldem's formulation.
    let output_data = par_map_rows(rows, cols, |row, out_row| {
        if row == 0 || row == rows - 1 {
            return; // edge rows stay NaN
        }
        // Three row slices: the borrow-checked equivalent of data[[r±1, c±1]]
        // without per-access stride arithmetic and bounds checks.
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

            let num = cos_zenith + sin_zenith * (dz_dy * sin_az - dz_dx * cos_az);
            let shade = num / (1.0 + dz_dx * dz_dx + dz_dy * dz_dy).sqrt();

            let shade_clamped = shade.clamp(0.0, 1.0);
            out_row[col] = if normalized {
                shade_clamped
            } else {
                (shade_clamped * 255.0).round()
            };
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    // NaN, not 0.0: a shade of 0 is a valid value (full shadow).
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming hillshade calculator implementing `WindowAlgorithm`.
///
/// Processes a DEM strip-by-strip with bounded memory.
/// Uses the same Horn (1981) 3×3 method as `hillshade()`.
#[derive(Debug, Clone)]
pub struct HillshadeStreaming {
    /// Sun azimuth in degrees (0 = North, clockwise)
    pub azimuth: f64,
    /// Sun altitude in degrees above horizon (0-90)
    pub altitude: f64,
    /// Vertical exaggeration: `z' = z_factor * z` (GDAL convention).
    pub z_factor: f64,
}

impl Default for HillshadeStreaming {
    fn default() -> Self {
        Self {
            azimuth: 315.0,
            altitude: 45.0,
            z_factor: 1.0,
        }
    }
}

impl HillshadeStreaming {
    /// Process a single output row given its already-resolved cell sizes
    /// (`eight_dx`/`eight_dy`) and pre-computed illumination geometry.
    /// Shared by the constant-cell-size path (`process_chunk`) and the
    /// per-row geographic-correction path (`process_chunk_geo`).
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
        cos_zenith: f64,
        sin_zenith: f64,
        sin_az: f64,
        cos_az: f64,
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

            // Horn's method (z_factor scales z, per GDAL convention)
            let dz_dx = zf * ((cv + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_dx;
            let dz_dy = zf * ((g + 2.0 * h + i) - (a + 2.0 * b + cv)) / eight_dy;

            // Algebraic form — see hillshade() for the derivation.
            let num = cos_zenith + sin_zenith * (dz_dy * sin_az - dz_dx * cos_az);
            let shade = num / (1.0 + dz_dx * dz_dx + dz_dy * dz_dy).sqrt();

            output[[r, c]] = (shade.clamp(0.0, 1.0) * 255.0).round();
        }
    }
}

impl surtgis_core::WindowAlgorithm for HillshadeStreaming {
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

        // Pre-compute illumination angles (same convention as hillshade())
        let azimuth_rad = (360.0 - self.azimuth + 90.0).to_radians();
        let zenith_rad = (90.0 - self.altitude).to_radians();
        let cos_zenith = zenith_rad.cos();
        let sin_zenith = zenith_rad.sin();
        let (sin_az, cos_az) = azimuth_rad.sin_cos();

        let eight_dx = 8.0 * cell_size_x;
        let eight_dy = 8.0 * cell_size_y.abs();

        for r in 0..out_rows {
            let ir = r + radius; // input row corresponding to output row r
            self.process_row(
                input, output, nodata, r, ir, in_rows, cols, eight_dx, eight_dy, cos_zenith,
                sin_zenith, sin_az, cos_az,
            );
        }
    }

    /// REG-1 fix: see `SlopeStreaming::process_chunk_geo` — same mechanism.
    /// Hillshade's gradient is just as latitude-sensitive as slope's, so
    /// this was silently wrong on geographic CRSs in streaming mode too.
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
            return self.process_chunk(input, output, nodata, cell_size_x, cell_size_y);
        };

        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        let azimuth_rad = (360.0 - self.azimuth + 90.0).to_radians();
        let zenith_rad = (90.0 - self.altitude).to_radians();
        let cos_zenith = zenith_rad.cos();
        let sin_zenith = zenith_rad.sin();
        let (sin_az, cos_az) = azimuth_rad.sin_cos();

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
                input, output, nodata, r, ir, in_rows, cols, eight_dx, eight_dy, cos_zenith,
                sin_zenith, sin_az, cos_az,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dem() -> Raster<f64> {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64 * 10.0).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_hillshade_range() {
        let dem = create_test_dem();
        let result = hillshade(&dem, HillshadeParams::default()).unwrap();

        // All valid values should be in [0, 255]; edges are NaN (nodata)
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                if row == 0 || row == result.rows() - 1 || col == 0 || col == result.cols() - 1 {
                    assert!(val.is_nan(), "Edge should be NaN at ({}, {})", row, col);
                    continue;
                }
                assert!(
                    val >= 0.0 && val <= 255.0,
                    "Hillshade value {} out of range at ({}, {})",
                    val,
                    row,
                    col
                );
            }
        }
    }

    /// Regression test for the N–S mirrored illumination bug.
    ///
    /// A uniform north-facing slope (elevation increasing southward) lit
    /// from the NW (az=315, alt=45) must be bright. Verified against
    /// `gdaldem hillshade`: expected 218 for slope tan = 0.5; the mirrored
    /// formula produced 104. The south-facing slope must be darker.
    #[test]
    fn test_hillshade_directional_north_vs_south() {
        let n = 20;
        let cell = 10.0;
        // North-facing: z increases with row (rows go south in a north-up raster)
        let mut north = Raster::new(n, n);
        north.set_transform(surtgis_core::GeoTransform::new(
            0.0,
            n as f64 * cell,
            cell,
            -cell,
        ));
        // South-facing: z decreases with row
        let mut south = Raster::new(n, n);
        south.set_transform(surtgis_core::GeoTransform::new(
            0.0,
            n as f64 * cell,
            cell,
            -cell,
        ));
        for row in 0..n {
            for col in 0..n {
                north.set(row, col, row as f64 * cell * 0.5).unwrap();
                south
                    .set(row, col, (n - 1 - row) as f64 * cell * 0.5)
                    .unwrap();
            }
        }

        let params = HillshadeParams::default(); // az 315, alt 45
        let hn = hillshade(&north, params.clone()).unwrap();
        let hs = hillshade(&south, params).unwrap();

        let vn = hn.get(n / 2, n / 2).unwrap();
        let vs = hs.get(n / 2, n / 2).unwrap();

        // GDAL gives 218 for the north-facing slope with these parameters
        assert!(
            (vn - 218.0).abs() <= 1.0,
            "North-facing slope with NW sun should be ~218 (GDAL), got {}",
            vn
        );
        // North-facing must be brighter than south-facing under NW sun
        assert!(
            vn > vs + 50.0,
            "North-facing ({}) must be brighter than south-facing ({}) under NW sun",
            vn,
            vs
        );
    }

    #[test]
    fn test_hillshade_flat() {
        let mut dem: Raster<f64> = Raster::filled(10, 10, 100.0);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let params = HillshadeParams {
            altitude: 45.0,
            ..Default::default()
        };

        let result = hillshade(&dem, params).unwrap();
        let val = result.get(5, 5).unwrap();

        // Flat surface at 45° altitude should have shade ≈ cos(45°) ≈ 0.707 → ~180
        assert!(
            (val - 180.0).abs() < 20.0,
            "Expected ~180 for flat surface, got {}",
            val
        );
    }

    /// The algebraic kernel must be numerically equivalent to the classic
    /// trigonometric formulation (slope/aspect + cos(az − aspect)) — same
    /// math, reassociated. Verified over an irregular synthetic surface.
    #[test]
    fn test_algebraic_matches_trig_formulation() {
        let n = 40;
        let mut dem = Raster::new(n, n);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, n as f64, 1.0, -1.0));
        for r in 0..n {
            for c in 0..n {
                // Irregular surface exercising all aspect quadrants
                let z = (r as f64 * 0.37).sin() * 40.0
                    + (c as f64 * 0.23).cos() * 25.0
                    + r as f64 * 1.5
                    - c as f64 * 0.8;
                dem.set(r, c, z).unwrap();
            }
        }

        for az in [0.0, 90.0, 135.0, 315.0] {
            let params = HillshadeParams {
                azimuth: az,
                normalized: true,
                ..Default::default()
            };
            let result = hillshade(&dem, params).unwrap();

            // Reference: classic trig formulation
            let azimuth_rad = (360.0 - az + 90.0).to_radians();
            let zenith_rad = (90.0f64 - 45.0).to_radians();
            let data = dem.data();
            for r in 1..n - 1 {
                for c in 1..n - 1 {
                    let dz_dx = ((data[[r - 1, c + 1]]
                        + 2.0 * data[[r, c + 1]]
                        + data[[r + 1, c + 1]])
                        - (data[[r - 1, c - 1]] + 2.0 * data[[r, c - 1]] + data[[r + 1, c - 1]]))
                        / 8.0;
                    let dz_dy = ((data[[r + 1, c - 1]]
                        + 2.0 * data[[r + 1, c]]
                        + data[[r + 1, c + 1]])
                        - (data[[r - 1, c - 1]] + 2.0 * data[[r - 1, c]] + data[[r - 1, c + 1]]))
                        / 8.0;
                    let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
                    let aspect_rad = dz_dy.atan2(-dz_dx);
                    let expected = (zenith_rad.cos() * slope_rad.cos()
                        + zenith_rad.sin() * slope_rad.sin() * (azimuth_rad - aspect_rad).cos())
                    .clamp(0.0, 1.0);
                    let got = result.get(r, c).unwrap();
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "az={} at ({},{}): algebraic {} vs trig {}",
                        az,
                        r,
                        c,
                        got,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_hillshade_normalized() {
        let dem = create_test_dem();
        let params = HillshadeParams {
            normalized: true,
            ..Default::default()
        };

        let result = hillshade(&dem, params).unwrap();

        // All valid values should be in [0, 1]; edges are NaN
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                if val.is_nan() {
                    continue;
                }
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Normalized hillshade {} out of range",
                    val
                );
            }
        }
    }
}
