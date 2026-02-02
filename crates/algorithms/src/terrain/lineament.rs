//! Lineament Detection Pipeline
//!
//! Detects and classifies geological lineaments from DEM curvature maps.
//! Lineaments are linear features (faults, fractures, fold axes) expressed
//! as ridges/valleys in the terrain surface.
//!
//! Pipeline:
//! 1. Compute plan curvature (kh) and profile curvature (kv)
//! 2. Binarize at zero-crossing
//! 3. Skeletonize using Zhang-Suen thinning
//! 4. Classify by curvature map origin
//!
//! Reference:
//! Florinsky, I.V. (2025). Digital terrain analysis in soil science and geology,
//! Chapter 14: "Lineaments and faults".

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Lineament classification based on which curvature zero-crossings are used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineamentType {
    /// Zero-crossings of plan curvature (kh) only → strike-slip faults
    StrikeSlip = 1,
    /// Zero-crossings of profile curvature (kv) only → dip-slip faults
    DipSlip = 2,
    /// Zero-crossings of both kh and kv → oblique-slip faults
    Oblique = 3,
}

/// Parameters for lineament detection
#[derive(Debug, Clone)]
pub struct LineamentParams {
    /// Minimum skeleton segment length in cells (default: 5).
    /// Shorter segments are filtered out as noise.
    pub min_length: usize,
}

impl Default for LineamentParams {
    fn default() -> Self {
        Self { min_length: 5 }
    }
}

/// Result of lineament detection
#[derive(Debug)]
pub struct LineamentResult {
    /// Skeleton from plan curvature (kh) zero-crossings.
    /// 1 = skeleton pixel, 0 = background.
    pub kh_skeleton: Raster<u8>,
    /// Skeleton from profile curvature (kv) zero-crossings.
    pub kv_skeleton: Raster<u8>,
    /// Classified lineament map:
    /// 0 = no lineament, 1 = strike-slip (kh only),
    /// 2 = dip-slip (kv only), 3 = oblique (both)
    pub classified: Raster<u8>,
}

/// Binarize a curvature raster at zero-crossing.
/// Returns 1 where the value changes sign across neighbors (zero-crossing),
/// 0 elsewhere.
fn binarize_zero_crossing(data: &Array2<f64>) -> Array2<u8> {
    let (rows, cols) = data.dim();
    let mut binary = Array2::<u8>::zeros((rows, cols));

    for r in 1..rows.saturating_sub(1) {
        for c in 1..cols.saturating_sub(1) {
            let v = data[[r, c]];
            if v.is_nan() {
                continue;
            }

            // Check if any 4-connected neighbor has opposite sign
            let neighbors = [
                data[[r - 1, c]],
                data[[r + 1, c]],
                data[[r, c - 1]],
                data[[r, c + 1]],
            ];

            for &n in &neighbors {
                if !n.is_nan() && v * n < 0.0 {
                    binary[[r, c]] = 1;
                    break;
                }
            }
        }
    }

    binary
}

/// Zhang-Suen thinning algorithm for binary image skeletonization.
/// Iteratively removes boundary pixels until skeleton is achieved.
fn zhang_suen_thin(image: &Array2<u8>) -> Array2<u8> {
    let (rows, cols) = image.dim();
    let mut current = image.clone();

    loop {
        let mut changed = false;

        // Sub-iteration 1
        let mut to_remove = Vec::new();
        for r in 1..rows.saturating_sub(1) {
            for c in 1..cols.saturating_sub(1) {
                if current[[r, c]] == 0 {
                    continue;
                }

                let p = neighbors_8(&current, r, c);
                let b = p.iter().filter(|&&v| v == 1).count();
                let a = transitions_01(&p);

                // Conditions for sub-iteration 1
                if (2..=6).contains(&b)
                    && a == 1
                    && (p[0] * p[2] * p[4]) == 0  // P2 * P4 * P6 = 0
                    && (p[2] * p[4] * p[6]) == 0  // P4 * P6 * P8 = 0
                {
                    to_remove.push((r, c));
                }
            }
        }
        for (r, c) in &to_remove {
            current[[*r, *c]] = 0;
            changed = true;
        }

        // Sub-iteration 2
        let mut to_remove = Vec::new();
        for r in 1..rows.saturating_sub(1) {
            for c in 1..cols.saturating_sub(1) {
                if current[[r, c]] == 0 {
                    continue;
                }

                let p = neighbors_8(&current, r, c);
                let b = p.iter().filter(|&&v| v == 1).count();
                let a = transitions_01(&p);

                // Conditions for sub-iteration 2
                if (2..=6).contains(&b)
                    && a == 1
                    && (p[0] * p[2] * p[6]) == 0  // P2 * P4 * P8 = 0
                    && (p[0] * p[4] * p[6]) == 0  // P2 * P6 * P8 = 0
                {
                    to_remove.push((r, c));
                }
            }
        }
        for (r, c) in &to_remove {
            current[[*r, *c]] = 0;
            changed = true;
        }

        if !changed {
            break;
        }
    }

    current
}

/// Get 8 neighbors in clockwise order: P2(N), P3(NE), P4(E), P5(SE), P6(S), P7(SW), P8(W), P9(NW)
fn neighbors_8(img: &Array2<u8>, r: usize, c: usize) -> [u8; 8] {
    [
        img[[r - 1, c]],     // P2 - North
        img[[r - 1, c + 1]], // P3 - NE
        img[[r, c + 1]],     // P4 - East
        img[[r + 1, c + 1]], // P5 - SE
        img[[r + 1, c]],     // P6 - South
        img[[r + 1, c - 1]], // P7 - SW
        img[[r, c - 1]],     // P8 - West
        img[[r - 1, c - 1]], // P9 - NW
    ]
}

/// Count 0→1 transitions in the ordered neighbor sequence
fn transitions_01(p: &[u8; 8]) -> usize {
    let mut count = 0;
    for i in 0..8 {
        if p[i] == 0 && p[(i + 1) % 8] == 1 {
            count += 1;
        }
    }
    count
}

/// Remove short skeleton segments (noise filtering).
/// Uses connected-component analysis with 8-connectivity.
fn filter_short_segments(skeleton: &mut Array2<u8>, min_length: usize) {
    let (rows, cols) = skeleton.dim();
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);

    for r in 0..rows {
        for c in 0..cols {
            if skeleton[[r, c]] == 0 || visited[[r, c]] {
                continue;
            }

            // BFS to find connected component
            let mut component = Vec::new();
            let mut queue = vec![(r, c)];
            visited[[r, c]] = true;

            while let Some((cr, cc)) = queue.pop() {
                component.push((cr, cc));

                for dr in -1_isize..=1 {
                    for dc in -1_isize..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = cr as isize + dr;
                        let nc = cc as isize + dc;
                        if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            if skeleton[[nr, nc]] == 1 && !visited[[nr, nc]] {
                                visited[[nr, nc]] = true;
                                queue.push((nr, nc));
                            }
                        }
                    }
                }
            }

            // Remove if too short
            if component.len() < min_length {
                for (cr, cc) in component {
                    skeleton[[cr, cc]] = 0;
                }
            }
        }
    }
}

/// Detect and classify lineaments from plan and profile curvature rasters.
///
/// # Arguments
/// * `plan_curvature` — Plan curvature (kh) raster
/// * `profile_curvature` — Profile curvature (kv) raster
/// * `params` — Lineament detection parameters
///
/// # Returns
/// `LineamentResult` with kh/kv skeletons and classified map
pub fn lineament_detection(
    plan_curvature: &Raster<f64>,
    profile_curvature: &Raster<f64>,
    params: LineamentParams,
) -> Result<LineamentResult> {
    let (rows_h, cols_h) = plan_curvature.shape();
    let (rows_v, cols_v) = profile_curvature.shape();

    if rows_h != rows_v || cols_h != cols_v {
        return Err(Error::SizeMismatch {
            er: rows_h, ec: cols_h,
            ar: rows_v, ac: cols_v,
        });
    }

    let rows = rows_h;
    let cols = cols_h;

    // Step 1: Binarize at zero-crossings
    let kh_binary = binarize_zero_crossing(plan_curvature.data());
    let kv_binary = binarize_zero_crossing(profile_curvature.data());

    // Step 2: Skeletonize
    let mut kh_skel = zhang_suen_thin(&kh_binary);
    let mut kv_skel = zhang_suen_thin(&kv_binary);

    // Step 3: Filter short segments
    filter_short_segments(&mut kh_skel, params.min_length);
    filter_short_segments(&mut kv_skel, params.min_length);

    // Step 4: Classify
    let mut classified_data = Array2::<u8>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let has_kh = kh_skel[[r, c]] > 0;
            let has_kv = kv_skel[[r, c]] > 0;
            classified_data[[r, c]] = match (has_kh, has_kv) {
                (true, true) => LineamentType::Oblique as u8,
                (true, false) => LineamentType::StrikeSlip as u8,
                (false, true) => LineamentType::DipSlip as u8,
                (false, false) => 0,
            };
        }
    }

    // Build output rasters
    let mut kh_out = plan_curvature.with_same_meta::<u8>(rows, cols);
    kh_out.set_nodata(Some(0));
    *kh_out.data_mut() = kh_skel;

    let mut kv_out = profile_curvature.with_same_meta::<u8>(rows, cols);
    kv_out.set_nodata(Some(0));
    *kv_out.data_mut() = kv_skel;

    let mut class_out = plan_curvature.with_same_meta::<u8>(rows, cols);
    class_out.set_nodata(Some(0));
    *class_out.data_mut() = classified_data;

    Ok(LineamentResult {
        kh_skeleton: kh_out,
        kv_skeleton: kv_out,
        classified: class_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    fn make_curvature_pair(rows: usize, cols: usize) -> (Raster<f64>, Raster<f64>) {
        // Create curvatures with a clear zero-crossing line
        let mut kh_data = Array2::zeros((rows, cols));
        let mut kv_data = Array2::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                // Vertical zero-crossing in kh at center column
                kh_data[[r, c]] = c as f64 - cols as f64 / 2.0;
                // Horizontal zero-crossing in kv at center row
                kv_data[[r, c]] = r as f64 - rows as f64 / 2.0;
            }
        }

        let mut kh = Raster::new(rows, cols);
        kh.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        *kh.data_mut() = kh_data;

        let mut kv = Raster::new(rows, cols);
        kv.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        *kv.data_mut() = kv_data;

        (kh, kv)
    }

    #[test]
    fn test_lineament_basic() {
        let (kh, kv) = make_curvature_pair(21, 21);
        let result = lineament_detection(&kh, &kv, LineamentParams { min_length: 3 }).unwrap();

        // Should have some skeleton pixels in kh
        let kh_count: usize = result.kh_skeleton.data().iter().map(|&v| v as usize).sum();
        assert!(kh_count > 0, "kh skeleton should have pixels");

        // Should have some skeleton pixels in kv
        let kv_count: usize = result.kv_skeleton.data().iter().map(|&v| v as usize).sum();
        assert!(kv_count > 0, "kv skeleton should have pixels");
    }

    #[test]
    fn test_lineament_classification() {
        let (kh, kv) = make_curvature_pair(21, 21);
        let result = lineament_detection(&kh, &kv, LineamentParams { min_length: 3 }).unwrap();

        // Count classified types
        let mut strike = 0;
        let mut dip = 0;
        let mut oblique = 0;
        for &v in result.classified.data().iter() {
            match v {
                1 => strike += 1,
                2 => dip += 1,
                3 => oblique += 1,
                _ => {}
            }
        }

        // Should have at least some strike-slip and dip-slip
        assert!(strike + dip + oblique > 0, "Should have classified lineaments");
    }

    #[test]
    fn test_lineament_size_mismatch() {
        let mut kh = Raster::new(10, 10);
        kh.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let mut kv = Raster::new(10, 20);
        kv.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        assert!(lineament_detection(&kh, &kv, LineamentParams::default()).is_err());
    }

    #[test]
    fn test_zero_crossing_detection() {
        let mut data = Array2::zeros((5, 5));
        // Create a clear zero-crossing between columns 1 and 2
        for r in 0..5 {
            for c in 0..5 {
                data[[r, c]] = c as f64 - 1.5; // -1.5, -0.5, 0.5, 1.5, 2.5
            }
        }
        let binary = binarize_zero_crossing(&data);

        // Zero-crossing between cols 1 (-0.5) and 2 (0.5)
        let total: usize = binary.iter().map(|&v| v as usize).sum();
        assert!(total > 0, "Should detect zero-crossings");
        // Interior cells at col 1 and 2 should be marked
        assert_eq!(binary[[2, 1]], 1, "Col 1 should be at zero-crossing");
        assert_eq!(binary[[2, 2]], 1, "Col 2 should be at zero-crossing");
    }

    #[test]
    fn test_zhang_suen_thinning() {
        // Create a thick line (3 pixels wide) and verify it thins to 1 pixel
        let mut img = Array2::<u8>::zeros((11, 11));
        for r in 3..8 {
            for c in 4..7 {
                img[[r, c]] = 1; // 3-wide vertical line
            }
        }

        let skeleton = zhang_suen_thin(&img);
        let skel_count: usize = skeleton.iter().map(|&v| v as usize).sum();

        // Should be thinner than original
        let orig_count: usize = img.iter().map(|&v| v as usize).sum();
        assert!(
            skel_count < orig_count,
            "Skeleton should be thinner: orig={}, skel={}",
            orig_count, skel_count
        );
        assert!(skel_count > 0, "Skeleton should not be empty");
    }

    #[test]
    fn test_filter_short_segments() {
        let mut skel = Array2::<u8>::zeros((10, 10));
        // Long segment (length 7)
        for c in 2..9 {
            skel[[5, c]] = 1;
        }
        // Short segment (length 2)
        skel[[2, 2]] = 1;
        skel[[2, 3]] = 1;

        filter_short_segments(&mut skel, 5);

        // Long segment should survive
        assert_eq!(skel[[5, 5]], 1, "Long segment should survive");
        // Short segment should be removed
        assert_eq!(skel[[2, 2]], 0, "Short segment should be removed");
    }
}
