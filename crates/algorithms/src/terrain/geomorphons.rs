//! Geomorphons — pattern recognition approach to landform classification
//!
//! Jasiewicz & Stepinski (2013): classifies terrain into 10 landform elements
//! by comparing each cell's elevation to the surrounding terrain along 8
//! directions via line-of-sight zenith/nadir angles. Each direction yields a
//! ternary value (+1, 0, −1); the counts of pluses and minuses index the
//! canonical form lookup table of the paper (the reduction of the 498
//! distinct ternary patterns to the 10 most common landform elements).
//!
//! This implementation reproduces GRASS `r.geomorphon` (the reference
//! implementation by the paper's authors) cell by cell in its default mode
//! (`comparison=anglev1`, basic correction), including:
//!
//! - circular, distance-based search (`radius`), inner `skip` ring and the
//!   `flatness_distance` threshold relaxation,
//! - the exact zenith/nadir ternary rule with its tie → 0 behaviour,
//! - the 9×9 (num−, num+) form matrix,
//! - the nulled border ring of width `skip + 1` ("on borders forms usually
//!   are innatural").
//!
//! Output class codes match the GRASS `forms` category values (1–10).
//!
//! Reference: Jasiewicz, J., Stepinski, T.F. (2013). Geomorphons — a pattern
//! recognition approach to classification and mapping of landforms.
//! *Geomorphology* 182, 147–156.

use crate::maybe_rayon::*;
use ndarray::Array2;
use std::f64::consts::{FRAC_PI_2, SQRT_2};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Geomorphon landform classes (same codes as the GRASS `forms` categories)
pub mod class {
    pub const FLAT: u8 = 1;
    pub const PEAK: u8 = 2;
    pub const RIDGE: u8 = 3;
    pub const SHOULDER: u8 = 4;
    pub const SPUR: u8 = 5;
    pub const SLOPE: u8 = 6;
    pub const HOLLOW: u8 = 7;
    pub const FOOTSLOPE: u8 = 8;
    pub const VALLEY: u8 = 9;
    pub const PIT: u8 = 10;
}

/// Parameters for geomorphon computation.
///
/// Mirrors GRASS `r.geomorphon`: `radius` = `search`, `skip` = `skip`,
/// `flatness_threshold` = `flat`, `flatness_distance` = `dist` (all
/// cell-based; GRASS defaults are `search=3`, `skip=0`, `flat=1`, `dist=0`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GeomorphonParams {
    /// Search radius in cells (default 10; GRASS default is 3). The search
    /// is circular: a cell is visited while its centre-to-centre distance is
    /// strictly less than `radius × cell_size`.
    pub radius: usize,
    /// Inner skip radius in cells (default 0): ignore the first `skip` cells
    /// of every profile to suppress the influence of small nearby forms.
    /// Must be smaller than `radius`.
    pub skip: usize,
    /// Flatness threshold in degrees (default 1.0)
    pub flatness_threshold: f64,
    /// Flatness distance in cells (default 0 = disabled): beyond this
    /// distance the flatness threshold is relaxed to the angle subtended by
    /// `tan(flatness_threshold) × flatness_distance` at the actual distance,
    /// reducing the impact of small irregularities on large-radius searches.
    /// Ignored (like GRASS, which warns) unless `skip < dist < radius`.
    pub flatness_distance: f64,
}

impl Default for GeomorphonParams {
    fn default() -> Self {
        Self {
            radius: 10,
            skip: 0,
            flatness_threshold: 1.0,
            flatness_distance: 0.0,
        }
    }
}

/// 8 direction vectors in GRASS profile order (NE, N, NW, W, SW, S, SE, E).
const DIRECTIONS: [(isize, isize); 8] = [
    (-1, 1),  // NE
    (-1, 0),  // N
    (-1, -1), // NW
    (0, -1),  // W
    (1, -1),  // SW
    (1, 0),   // S
    (1, 1),   // SE
    (0, 1),   // E
];

/// Canonical geomorphon form matrix (Jasiewicz & Stepinski 2013, as in GRASS
/// `determine_form`), indexed `[num_minus][num_plus]`. `0` marks impossible
/// combinations (`num_minus + num_plus > 8`).
#[rustfmt::skip]
const FORM_MATRIX: [[u8; 9]; 9] = {
    use class::*;
    /* minus ↓            plus →
              0         1         2         3        4       5       6       7      8 */
    [
    /* 0 */ [FLAT,      FLAT,     FLAT,     FOOTSLOPE, FOOTSLOPE, VALLEY, VALLEY, VALLEY, PIT],
    /* 1 */ [FLAT,      FLAT,     FOOTSLOPE, FOOTSLOPE, FOOTSLOPE, VALLEY, VALLEY, VALLEY, 0],
    /* 2 */ [FLAT,      SHOULDER, SLOPE,    SLOPE,    HOLLOW, HOLLOW, VALLEY, 0,     0],
    /* 3 */ [SHOULDER,  SHOULDER, SLOPE,    SLOPE,    SLOPE,  HOLLOW, 0,      0,     0],
    /* 4 */ [SHOULDER,  SHOULDER, SPUR,     SLOPE,    SLOPE,  0,      0,      0,     0],
    /* 5 */ [RIDGE,     RIDGE,    SPUR,     SPUR,     0,      0,      0,      0,     0],
    /* 6 */ [RIDGE,     RIDGE,    RIDGE,    0,        0,      0,      0,      0,     0],
    /* 7 */ [RIDGE,     RIDGE,    0,        0,        0,      0,      0,      0,     0],
    /* 8 */ [PEAK,      0,        0,        0,        0,      0,      0,      0,     0],
    ]
};

/// Compute geomorphons landform classification.
///
/// For each cell, walks the 8 directional elevation profiles out to the
/// circular search limit, computes the line-of-sight zenith and nadir angles
/// of each profile (ignoring the first `skip` cells), and converts them into
/// a ternary value with the flatness threshold: higher terrain dominates
/// (+1), lower terrain dominates (−1), or neither exceeds the threshold /
/// exact tie (0). The counts of pluses and minuses index the canonical
/// Jasiewicz & Stepinski form matrix.
///
/// Reproduces GRASS `r.geomorphon` (default `comparison=anglev1`, basic
/// mode) cell by cell, including its nulled border ring of width
/// `skip + 1`: border cells are set to nodata (0), because their truncated
/// profiles produce unnatural forms.
///
/// # Arguments
/// * `dem` - Input DEM (projected CRS; distances derive from the cell size)
/// * `params` - Search/skip radii and flatness threshold/distance
///
/// # Returns
/// `Raster<u8>` with landform class codes (1-10, GRASS category values;
/// 0 = nodata / border)
pub fn geomorphons(dem: &Raster<f64>, params: GeomorphonParams) -> Result<Raster<u8>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }
    if params.skip >= params.radius {
        return Err(Error::Algorithm(
            "Skip radius must be at least 1 cell smaller than the search radius".into(),
        ));
    }
    if params.flatness_threshold <= 0.0 {
        return Err(Error::Algorithm(
            "Flatness threshold must be greater than 0".into(),
        ));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();

    let search_distance = params.radius as f64 * cell_size;
    let skip_distance = params.skip as f64 * cell_size;
    let flat_threshold = params.flatness_threshold.to_radians();

    // GRASS `dist`: beyond `flatness_distance` the threshold angle is the one
    // subtended by a fixed height at the actual distance. Out-of-range values
    // are ignored, exactly like GRASS (which warns and sets it to 0).
    let mut flat_distance = params.flatness_distance * cell_size;
    let flat_threshold_height = flat_threshold.tan() * flat_distance;
    if (flat_distance > 0.0 && flat_distance <= skip_distance) || flat_distance >= search_distance {
        flat_distance = 0.0;
    }

    // GRASS nulls this outer ring ("on borders forms usually are innatural").
    let border = params.skip + 1;
    let skip = params.skip;

    // `par_map_rows` (crate::maybe_rayon) is hard-coded to `Array2<f64>`, but
    // this algorithm's output is `u8` landform codes, so it can't be reused
    // directly. This mirrors that helper's mechanism (preallocate once, hand
    // each thread a `&mut [u8]` row via `par_chunks_mut`) instead of the
    // slower `flat_map(|row| vec![...]).collect()` pattern.
    let mut output_data = Array2::from_elem((rows, cols), 0u8);
    output_data
        .as_slice_mut()
        .expect("freshly created Array2 is contiguous")
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            if row < border || row + border >= rows {
                return; // nulled border ring
            }
            for (col, row_data_col) in out_row.iter_mut().enumerate() {
                if col < border || col + border >= cols {
                    continue; // nulled border ring
                }
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if dem.is_nodata(z0) {
                    continue;
                }

                let mut num_plus = 0usize;
                let mut num_minus = 0usize;

                for &(dr, dc) in DIRECTIONS.iter() {
                    // Line-of-sight existence, exactly as GRASS: bounds are
                    // probed at the first profile cell (skip + 1 steps out),
                    // nodata at the immediate neighbor.
                    let mut j = skip + 1;
                    let (fr, fc) = (
                        row as isize + dr * j as isize,
                        col as isize + dc * j as isize,
                    );
                    if fr < 0 || fc < 0 || fr as usize >= rows || fc as usize >= cols {
                        continue; // profile leaves the DEM: direction stays 0
                    }
                    let immediate = unsafe {
                        dem.get_unchecked(
                            (row as isize + dr) as usize,
                            (col as isize + dc) as usize,
                        )
                    };
                    if dem.is_nodata(immediate) {
                        continue; // no line of sight: direction stays 0
                    }

                    let step = cell_size * if dr != 0 && dc != 0 { SQRT_2 } else { 1.0 };
                    let mut zenith_angle = -FRAC_PI_2;
                    let mut nadir_angle = FRAC_PI_2;
                    let mut zenith_distance = 0.0f64;
                    let mut nadir_distance = 0.0f64;

                    // Circular search: visit cells strictly inside the search
                    // distance (GRASS: `while (cur_distance < search_distance)`).
                    let mut cur_distance = j as f64 * step;
                    while cur_distance < search_distance {
                        let nr = row as isize + dr * j as isize;
                        let nc = col as isize + dc * j as isize;
                        if nr < 0 || nc < 0 || nr as usize >= rows || nc as usize >= cols {
                            break; // reached the DEM edge
                        }
                        let z = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
                        if !dem.is_nodata(z) {
                            let angle = (z - z0).atan2(cur_distance);
                            if angle > zenith_angle {
                                zenith_angle = angle;
                                zenith_distance = cur_distance;
                            }
                            if angle < nadir_angle {
                                nadir_angle = angle;
                                nadir_distance = cur_distance;
                            }
                        }
                        j += 1;
                        cur_distance = j as f64 * step;
                    }

                    // Distance-relaxed flatness thresholds (GRASS `dist`).
                    let zenith_threshold = if flat_distance > 0.0 && flat_distance < zenith_distance
                    {
                        flat_threshold_height.atan2(zenith_distance)
                    } else {
                        flat_threshold
                    };
                    let nadir_threshold = if flat_distance > 0.0 && flat_distance < nadir_distance {
                        flat_threshold_height.atan2(nadir_distance)
                    } else {
                        flat_threshold
                    };

                    // ANGLEV1 ternary rule: the larger absolute angle wins;
                    // an exact tie (common on integer DEMs) stays 0.
                    if zenith_angle.abs() > zenith_threshold || nadir_angle.abs() > nadir_threshold
                    {
                        if nadir_angle.abs() < zenith_angle.abs() {
                            num_plus += 1;
                        } else if nadir_angle.abs() > zenith_angle.abs() {
                            num_minus += 1;
                        }
                    }
                }

                *row_data_col = FORM_MATRIX[num_minus][num_plus];
            }
        });

    let mut output = dem.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = output_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_dem(size: usize) -> Raster<f64> {
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        dem
    }

    fn params(radius: usize) -> GeomorphonParams {
        GeomorphonParams {
            radius,
            ..Default::default()
        }
    }

    #[test]
    fn test_geomorphons_peak() {
        // High center, low surroundings → peak. The profile must be
        // curved: on a constant-angle cone zenith == nadir exactly and the
        // ANGLEV1 rule keeps the tie at 0 (GRASS behaves identically).
        let mut dem = make_dem(21);
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, 100.0 - (dx * dx + dy * dy) * 0.5)
                    .unwrap();
            }
        }

        let result = geomorphons(&dem, params(8)).unwrap();
        let center = result.get(10, 10).unwrap();
        assert_eq!(center, class::PEAK, "Center of hill should be peak");
    }

    #[test]
    fn test_geomorphons_pit() {
        // Low center, high surroundings → pit (parabolic bowl; see the
        // peak test for why the profile must be curved)
        let mut dem = make_dem(21);
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy) * 0.5).unwrap();
            }
        }

        let result = geomorphons(&dem, params(8)).unwrap();
        let center = result.get(10, 10).unwrap();
        assert_eq!(center, class::PIT, "Center of depression should be pit");
    }

    #[test]
    fn test_geomorphons_flat_and_border() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        let result = geomorphons(&dem, params(5)).unwrap();
        assert_eq!(result.get(10, 10).unwrap(), class::FLAT);
        // GRASS nulls the border ring of width skip + 1 (here 1).
        for i in 0..21 {
            assert_eq!(result.get(0, i).unwrap(), 0, "border row must be nodata");
            assert_eq!(result.get(i, 20).unwrap(), 0, "border col must be nodata");
        }
        assert_ne!(
            result.get(1, 1).unwrap(),
            0,
            "first interior cell classified"
        );
    }

    #[test]
    fn test_geomorphons_ridge_and_valley() {
        // Ridge crest along row 10: lower on ~6 directions, flat along the
        // crest → (6−, 0+) = ridge. The mirrored DEM gives a valley.
        let mut ridge = make_dem(21);
        let mut valley = make_dem(21);
        for row in 0..21 {
            for col in 0..21 {
                let dy = row as f64 - 10.0;
                ridge.set(row, col, 100.0 - dy * dy * 0.5).unwrap();
                valley.set(row, col, dy * dy * 0.5).unwrap();
            }
        }
        let r = geomorphons(&ridge, params(8)).unwrap();
        let v = geomorphons(&valley, params(8)).unwrap();
        assert_eq!(r.get(10, 10).unwrap(), class::RIDGE);
        assert_eq!(v.get(10, 10).unwrap(), class::VALLEY);
    }

    #[test]
    fn test_geomorphons_slope() {
        // Steep slope dipping east (slightly convex so profile angles
        // vary — an exact plane ties in every direction under ANGLEV1):
        // 3 directions down, 3 up, 2 flat → (3−, 3+) = slope.
        let mut dem = make_dem(21);
        for row in 0..21 {
            for col in 0..21 {
                let c = col as f64;
                dem.set(row, col, 100.0 - c * c * 0.2).unwrap();
            }
        }
        let result = geomorphons(&dem, params(8)).unwrap();
        assert_eq!(result.get(10, 10).unwrap(), class::SLOPE);
    }

    #[test]
    fn test_parameter_validation() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(geomorphons(&dem, params(0)).is_err());
        assert!(
            geomorphons(
                &dem,
                GeomorphonParams {
                    radius: 3,
                    skip: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            geomorphons(
                &dem,
                GeomorphonParams {
                    radius: 3,
                    flatness_threshold: 0.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn test_form_matrix_is_canonical() {
        // Spot checks against GRASS determine_form (indexed [minus][plus]).
        use class::*;
        assert_eq!(FORM_MATRIX[0][0], FLAT);
        assert_eq!(FORM_MATRIX[8][0], PEAK);
        assert_eq!(FORM_MATRIX[0][8], PIT);
        assert_eq!(FORM_MATRIX[6][0], RIDGE);
        assert_eq!(FORM_MATRIX[0][6], VALLEY);
        assert_eq!(FORM_MATRIX[3][0], SHOULDER);
        assert_eq!(FORM_MATRIX[0][3], FOOTSLOPE);
        assert_eq!(FORM_MATRIX[5][2], SPUR);
        assert_eq!(FORM_MATRIX[2][5], HOLLOW);
        assert_eq!(FORM_MATRIX[3][3], SLOPE);
        assert_eq!(FORM_MATRIX[2][4], HOLLOW);
        assert_eq!(FORM_MATRIX[4][2], SPUR);
        assert_eq!(FORM_MATRIX[1][2], FOOTSLOPE);
        assert_eq!(FORM_MATRIX[2][1], SHOULDER);
        // Impossible combinations are 0.
        assert_eq!(FORM_MATRIX[8][1], 0);
        assert_eq!(FORM_MATRIX[5][4], 0);
    }

    #[test]
    fn test_exact_tie_is_flat() {
        // In every direction: +5° to the ring at j=1, −5° to the ring at
        // j=2 (heights scaled by the profile distance, so both angles are
        // exactly 5°). ANGLEV1 keeps ties at 0 → the cell must be FLAT, not
        // PIT (+ wins) nor PEAK (− wins).
        let t = 5.0_f64.to_radians().tan();
        let mut dem = make_dem(11);
        for &(dr, dc) in DIRECTIONS.iter() {
            let d = if dr != 0 && dc != 0 { SQRT_2 } else { 1.0 };
            dem.set((5 + dr) as usize, (5 + dc) as usize, t * d)
                .unwrap();
            dem.set((5 + 2 * dr) as usize, (5 + 2 * dc) as usize, -t * 2.0 * d)
                .unwrap();
        }
        let result = geomorphons(&dem, params(3)).unwrap();
        assert_eq!(result.get(5, 5).unwrap(), class::FLAT);
    }

    #[test]
    fn test_search_is_circular_and_strict() {
        // Spikes ON the search-distance circle are excluded (GRASS visits
        // cells with distance strictly below radius × cell); one ring closer
        // they are all seen.
        let radius = 5usize;
        let mut outside = make_dem(21);
        let mut inside = make_dem(21);
        for &(dr, dc) in DIRECTIONS.iter() {
            let diag = dr != 0 && dc != 0;
            // outside: cardinal at j=radius (dist = 5.0), diagonal at j=4
            // (dist = 5.66) — both ≥ search distance
            let j = if diag { 4 } else { radius as isize };
            outside
                .set((10 + dr * j) as usize, (10 + dc * j) as usize, 50.0)
                .unwrap();
            // inside: cardinal at j=4 (4.0), diagonal at j=3 (4.24) — both
            // strictly inside
            let j = if diag { 3 } else { 4 };
            inside
                .set((10 + dr * j) as usize, (10 + dc * j) as usize, 50.0)
                .unwrap();
        }
        let out = geomorphons(&outside, params(radius)).unwrap();
        let inn = geomorphons(&inside, params(radius)).unwrap();
        assert_eq!(
            out.get(10, 10).unwrap(),
            class::FLAT,
            "ring on the circle is unseen"
        );
        assert_eq!(
            inn.get(10, 10).unwrap(),
            class::PIT,
            "ring inside the circle is seen"
        );
    }

    #[test]
    fn test_skip_ignores_near_terrain() {
        // A crater rim at j=1 around the center: with skip=0 the profiles
        // see the rim (PIT); with skip=1 they start beyond it (FLAT). The
        // border widens with skip, so keep the center well inside.
        let mut dem = make_dem(21);
        for &(dr, dc) in DIRECTIONS.iter() {
            dem.set((10 + dr) as usize, (10 + dc) as usize, 50.0)
                .unwrap();
        }
        let no_skip = geomorphons(&dem, params(5)).unwrap();
        let skip1 = geomorphons(
            &dem,
            GeomorphonParams {
                radius: 5,
                skip: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(no_skip.get(10, 10).unwrap(), class::PIT);
        assert_eq!(skip1.get(10, 10).unwrap(), class::FLAT);
    }

    #[test]
    fn test_flatness_distance_relaxes_threshold() {
        // Radially symmetric parabolic bowl whose rim angle (~0.8°) stays
        // under the 1° flatness threshold → FLAT. With dist=4 the threshold
        // at the far rim relaxes to ~0.5° → the bowl becomes a PIT.
        let a = 0.8_f64.to_radians().tan() / 8.0; // angle 0.8° at r = 8 cells
        let mut dem = make_dem(21);
        // parabola z = a·r²: the profile angle atan(a·r) grows with distance
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, a * (dx * dx + dy * dy)).unwrap();
            }
        }

        let strict = geomorphons(&dem, params(9)).unwrap();
        let relaxed = geomorphons(
            &dem,
            GeomorphonParams {
                radius: 9,
                flatness_distance: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(strict.get(10, 10).unwrap(), class::FLAT);
        assert_eq!(relaxed.get(10, 10).unwrap(), class::PIT);
    }

    #[test]
    fn test_nodata_blocks_line_of_sight() {
        // Immediate nodata neighbor: that direction contributes 0 (no line
        // of sight), like GRASS. A crater rim with a nodata gap on the east
        // side loses one + → (0−, 7+) = valley instead of pit.
        let mut dem = make_dem(21);
        dem.set_nodata(Some(-9999.0));
        for &(dr, dc) in DIRECTIONS.iter() {
            dem.set((10 + 3 * dr) as usize, (10 + 3 * dc) as usize, 50.0)
                .unwrap();
        }
        dem.set(10, 11, -9999.0).unwrap(); // east immediate neighbor
        let result = geomorphons(&dem, params(5)).unwrap();
        assert_eq!(result.get(10, 10).unwrap(), class::VALLEY);
    }
}
