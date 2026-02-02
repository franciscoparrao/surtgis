//! Shared partial derivative computation from 3×3 DEM windows
//!
//! Provides reusable functions for estimating first- and second-order
//! partial derivatives from a 3×3 elevation neighborhood.
//!
//! ```text
//! 3×3 window indexing:
//!
//!   z1 z2 z3      (NW) (N) (NE)
//!   z4 z5 z6  →   (W)  (C) (E)
//!   z7 z8 z9      (SW) (S) (SE)
//! ```
//!
//! Three methods are supported:
//!
//! - **Evans-Young (1979/1978)**: Weighted least-squares on all 9 cells.
//!   Recommended by Florinsky (2025) as the standard 3×3 approach.
//!   Computes: p, q, r, s, t (5 derivatives).
//!
//! - **Zevenbergen-Thorne (1987)**: Central differences on cardinal/diagonal.
//!   Legacy method. Computes: p, q, r, s, t.
//!
//! - **Horn (1981)**: 3rd-order finite difference weighted scheme.
//!   Only computes first-order derivatives (p, q). Used by slope/hillshade.
//!
//! Reference:
//! Evans, I.S. (1979). An integrated system of terrain analysis.
//! Young, M. (1978). Statistical characterization of altitude matrices.
//! Zevenbergen, L.W. & Thorne, C.R. (1987). Quantitative analysis of
//!   land surface topography. ESPL.
//! Horn, B.K.P. (1981). Hill shading and the reflectance map. IEEE.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, §4.1.

/// First- and second-order partial derivatives of a surface.
///
/// Notation follows Florinsky (2025):
/// - p = ∂z/∂x  (east-west gradient)
/// - q = ∂z/∂y  (north-south gradient)
/// - r = ∂²z/∂x²  (second derivative in x)
/// - s = ∂²z/∂x∂y  (mixed derivative)
/// - t = ∂²z/∂y²  (second derivative in y)
#[derive(Debug, Clone, Copy)]
pub struct Derivatives {
    /// ∂z/∂x — east-west gradient
    pub p: f64,
    /// ∂z/∂y — north-south gradient
    pub q: f64,
    /// ∂²z/∂x² — second derivative in x
    pub r: f64,
    /// ∂²z/∂x∂y — mixed partial derivative
    pub s: f64,
    /// ∂²z/∂y² — second derivative in y
    pub t: f64,
}

impl Derivatives {
    /// Slope magnitude: √(p² + q²)
    #[inline]
    pub fn slope_magnitude(&self) -> f64 {
        (self.p * self.p + self.q * self.q).sqrt()
    }

    /// Slope angle in radians: atan(√(p² + q²))
    #[inline]
    pub fn slope_angle(&self) -> f64 {
        self.slope_magnitude().atan()
    }

    /// p² + q² — used repeatedly in curvature formulas
    #[inline]
    pub fn p2_plus_q2(&self) -> f64 {
        self.p * self.p + self.q * self.q
    }

    /// General (mean) curvature using full formula
    /// K_G = -((1+q²)r - 2pqs + (1+p²)t) / (2(1+p²+q²)^(3/2))
    #[inline]
    pub fn general_curvature(&self) -> f64 {
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let w = 1.0 + p2 + q2;
        -((1.0 + q2) * self.r - 2.0 * self.p * self.q * self.s + (1.0 + p2) * self.t)
            / (2.0 * w * w.sqrt())
    }

    /// Profile curvature (along slope direction)
    /// K_P = -(p²r + 2pqs + q²t) / ((p²+q²)(1+p²+q²)^(3/2))
    #[inline]
    pub fn profile_curvature(&self) -> f64 {
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let p2q2 = p2 + q2;
        if p2q2 < 1e-20 {
            return 0.0;
        }
        let w = 1.0 + p2q2;
        -(p2 * self.r + 2.0 * self.p * self.q * self.s + q2 * self.t)
            / (p2q2 * w * w.sqrt())
    }

    /// Plan (tangential) curvature (perpendicular to slope)
    /// K_T = -(q²r - 2pqs + p²t) / ((p²+q²)√(1+p²+q²))
    #[inline]
    pub fn plan_curvature(&self) -> f64 {
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let p2q2 = p2 + q2;
        if p2q2 < 1e-20 {
            return 0.0;
        }
        let w = 1.0 + p2q2;
        -(q2 * self.r - 2.0 * self.p * self.q * self.s + p2 * self.t)
            / (p2q2 * w.sqrt())
    }

    /// Gaussian curvature
    /// K = (r·t - s²) / (1+p²+q²)²
    #[inline]
    pub fn gaussian_curvature(&self) -> f64 {
        let w = 1.0 + self.p * self.p + self.q * self.q;
        (self.r * self.t - self.s * self.s) / (w * w)
    }

    /// Simplified general curvature (valid for gentle slopes <10°)
    /// K_G ≈ -(r + t) / 2
    #[inline]
    pub fn general_curvature_simplified(&self) -> f64 {
        -(self.r + self.t) / 2.0
    }
}

/// Compute partial derivatives using Evans-Young (1979) weighted least-squares.
///
/// This is the recommended method (Florinsky 2025). Uses all 9 cells in the
/// 3×3 window for a robust estimate. Computes all 5 derivatives (p,q,r,s,t).
///
/// # Arguments
/// * `z` — 9 elevation values: [z1, z2, z3, z4, z5, z6, z7, z8, z9]
///   in row-major order (NW, N, NE, W, C, E, SW, S, SE)
/// * `cellsize` — Grid cell size (assumes square cells)
#[inline]
pub fn evans_young(z: [f64; 9], cellsize: f64) -> Derivatives {
    let [z1, z2, z3, z4, _z5, z6, z7, z8, z9] = z;

    let cs6 = 6.0 * cellsize;
    let cs2_3 = 3.0 * cellsize * cellsize;
    let cs2_4 = 4.0 * cellsize * cellsize;

    Derivatives {
        p: (z3 + z6 + z9 - z1 - z4 - z7) / cs6,
        q: (z1 + z2 + z3 - z7 - z8 - z9) / cs6,
        r: (z1 + z3 + z4 + z6 + z7 + z9 - 2.0 * (z2 + _z5 + z8)) / cs2_3,
        s: (z3 + z7 - z1 - z9) / cs2_4,
        t: (z1 + z2 + z3 + z7 + z8 + z9 - 2.0 * (z4 + _z5 + z6)) / cs2_3,
    }
}

/// Compute partial derivatives using Zevenbergen-Thorne (1987) central differences.
///
/// Legacy method, less robust than Evans-Young. Only uses cardinal and
/// diagonal neighbors, not all 9 cells for second-order terms.
///
/// # Arguments
/// * `z` — 9 elevation values: [z1, z2, z3, z4, z5, z6, z7, z8, z9]
/// * `cellsize` — Grid cell size
#[inline]
pub fn zevenbergen_thorne(z: [f64; 9], cellsize: f64) -> Derivatives {
    let [z1, z2, z3, z4, z5, z6, z7, z8, z9] = z;

    let two_cs = 2.0 * cellsize;
    let cs2 = cellsize * cellsize;

    Derivatives {
        p: (z6 - z4) / two_cs,
        q: (z2 - z8) / two_cs,
        r: (z4 - 2.0 * z5 + z6) / cs2,
        s: (z3 - z1 - z9 + z7) / (4.0 * cs2),
        t: (z2 - 2.0 * z5 + z8) / cs2,
    }
}

/// Compute first-order derivatives using Horn (1981) 3rd-order finite difference.
///
/// Weighted scheme that gives more weight to cells adjacent to the center.
/// Only computes p and q (r, s, t are set to 0.0).
///
/// # Arguments
/// * `z` — 9 elevation values: [z1, z2, z3, z4, z5, z6, z7, z8, z9]
/// * `cellsize` — Grid cell size
#[inline]
pub fn horn(z: [f64; 9], cellsize: f64) -> Derivatives {
    let [z1, z2, z3, z4, _z5, z6, z7, z8, z9] = z;

    let eight_cs = 8.0 * cellsize;

    Derivatives {
        p: ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / eight_cs,
        q: ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / eight_cs,
        r: 0.0,
        s: 0.0,
        t: 0.0,
    }
}

/// Extract 3×3 window from a raster, returning None if any cell is NaN.
///
/// Returns values in row-major order: [NW, N, NE, W, C, E, SW, S, SE].
#[inline]
pub fn extract_window(data: &ndarray::Array2<f64>, row: usize, col: usize) -> Option<[f64; 9]> {
    let z1 = data[[row - 1, col - 1]];
    let z2 = data[[row - 1, col]];
    let z3 = data[[row - 1, col + 1]];
    let z4 = data[[row, col - 1]];
    let z5 = data[[row, col]];
    let z6 = data[[row, col + 1]];
    let z7 = data[[row + 1, col - 1]];
    let z8 = data[[row + 1, col]];
    let z9 = data[[row + 1, col + 1]];

    if z1.is_nan() || z2.is_nan() || z3.is_nan()
        || z4.is_nan() || z5.is_nan() || z6.is_nan()
        || z7.is_nan() || z8.is_nan() || z9.is_nan()
    {
        return None;
    }

    Some([z1, z2, z3, z4, z5, z6, z7, z8, z9])
}

#[cfg(test)]
mod tests {
    use super::*;

    const CS: f64 = 10.0;

    // Flat surface: all elevations = 100
    fn flat_window() -> [f64; 9] {
        [100.0; 9]
    }

    // Planar surface tilted in x: z = 2x
    // With cellsize=10, going left to right adds 20m per cell
    fn tilted_x_window() -> [f64; 9] {
        // col offsets: -1, 0, +1  →  elevations: -20, 0, +20 from center
        [
            80.0, 100.0, 120.0,  // row-1
            80.0, 100.0, 120.0,  // row
            80.0, 100.0, 120.0,  // row+1
        ]
    }

    // Planar surface tilted in y: z = 3y (north is +y)
    // row-1 is north (+y), so has higher elevation
    fn tilted_y_window() -> [f64; 9] {
        [
            130.0, 130.0, 130.0,  // north (row-1)
            100.0, 100.0, 100.0,  // center
            70.0,  70.0,  70.0,   // south (row+1)
        ]
    }

    // Parabolic bowl: z = x² + y² (minimum at center)
    // For cellsize=10: z(±1,0)=100, z(0,±1)=100, z(±1,±1)=200
    fn bowl_window() -> [f64; 9] {
        [
            200.0, 100.0, 200.0,
            100.0,   0.0, 100.0,
            200.0, 100.0, 200.0,
        ]
    }

    #[test]
    fn test_evans_young_flat() {
        let d = evans_young(flat_window(), CS);
        assert!(d.p.abs() < 1e-10, "p should be 0 on flat, got {}", d.p);
        assert!(d.q.abs() < 1e-10, "q should be 0 on flat, got {}", d.q);
        assert!(d.r.abs() < 1e-10, "r should be 0 on flat, got {}", d.r);
        assert!(d.s.abs() < 1e-10, "s should be 0 on flat, got {}", d.s);
        assert!(d.t.abs() < 1e-10, "t should be 0 on flat, got {}", d.t);
    }

    #[test]
    fn test_evans_young_tilted_x() {
        let d = evans_young(tilted_x_window(), CS);
        // z = 2x, so dz/dx = 2, dz/dy = 0
        assert!(
            (d.p - 2.0).abs() < 0.01,
            "p should be ~2.0, got {:.4}",
            d.p
        );
        assert!(d.q.abs() < 0.01, "q should be ~0, got {:.4}", d.q);
        assert!(d.r.abs() < 0.01, "r should be ~0, got {:.4}", d.r);
        assert!(d.t.abs() < 0.01, "t should be ~0, got {:.4}", d.t);
    }

    #[test]
    fn test_evans_young_tilted_y() {
        let d = evans_young(tilted_y_window(), CS);
        assert!(d.p.abs() < 0.01, "p should be ~0, got {:.4}", d.p);
        assert!(
            (d.q - 3.0).abs() < 0.01,
            "q should be ~3.0, got {:.4}",
            d.q
        );
    }

    #[test]
    fn test_evans_young_bowl() {
        let d = evans_young(bowl_window(), CS);
        // z = x² + y² → r = 2, t = 2, s = 0
        // Evans-Young: r = (z1+z3+z4+z6+z7+z9 - 2(z2+z5+z8)) / (3*cs²)
        // = (200+200+100+100+200+200 - 2*(100+0+100)) / (3*100)
        // = (1000 - 400) / 300 = 2.0
        assert!(
            (d.r - 2.0).abs() < 0.01,
            "r should be ~2.0, got {:.4}",
            d.r
        );
        assert!(
            (d.t - 2.0).abs() < 0.01,
            "t should be ~2.0, got {:.4}",
            d.t
        );
        assert!(d.s.abs() < 0.01, "s should be ~0, got {:.4}", d.s);
    }

    #[test]
    fn test_zevenbergen_thorne_flat() {
        let d = zevenbergen_thorne(flat_window(), CS);
        assert!(d.p.abs() < 1e-10);
        assert!(d.q.abs() < 1e-10);
        assert!(d.r.abs() < 1e-10);
        assert!(d.s.abs() < 1e-10);
        assert!(d.t.abs() < 1e-10);
    }

    #[test]
    fn test_zevenbergen_thorne_tilted_x() {
        let d = zevenbergen_thorne(tilted_x_window(), CS);
        // ZT: p = (z6 - z4) / (2*cs) = (120-80)/(20) = 2.0
        assert!(
            (d.p - 2.0).abs() < 0.01,
            "p should be ~2.0, got {:.4}",
            d.p
        );
        assert!(d.q.abs() < 0.01, "q should be ~0, got {:.4}", d.q);
    }

    #[test]
    fn test_zevenbergen_thorne_bowl() {
        let d = zevenbergen_thorne(bowl_window(), CS);
        // ZT: r = (z4 - 2z5 + z6) / cs² = (100 - 0 + 100) / 100 = 2.0
        assert!(
            (d.r - 2.0).abs() < 0.01,
            "r should be ~2.0, got {:.4}",
            d.r
        );
        assert!(
            (d.t - 2.0).abs() < 0.01,
            "t should be ~2.0, got {:.4}",
            d.t
        );
    }

    #[test]
    fn test_horn_flat() {
        let d = horn(flat_window(), CS);
        assert!(d.p.abs() < 1e-10);
        assert!(d.q.abs() < 1e-10);
    }

    #[test]
    fn test_horn_tilted_x() {
        let d = horn(tilted_x_window(), CS);
        // Horn: p = ((z3+2z6+z9) - (z1+2z4+z7)) / (8*cs)
        // = ((120+240+120) - (80+160+80)) / 80 = (480-320)/80 = 2.0
        assert!(
            (d.p - 2.0).abs() < 0.01,
            "p should be ~2.0, got {:.4}",
            d.p
        );
        assert!(d.q.abs() < 0.01, "q should be ~0, got {:.4}", d.q);
    }

    #[test]
    fn test_horn_tilted_y() {
        let d = horn(tilted_y_window(), CS);
        // Horn: q = ((z7+2z8+z9) - (z1+2z2+z3)) / (8*cs)
        // = ((70+140+70) - (130+260+130)) / 80 = (280-520)/80 = -3.0
        // Note: Horn's q is reversed because row+1 = south = -y
        assert!(
            (d.q - (-3.0)).abs() < 0.01,
            "q should be ~-3.0, got {:.4}",
            d.q
        );
    }

    #[test]
    fn test_curvature_methods() {
        // Bowl with Evans-Young
        let d = evans_young(bowl_window(), CS);

        // General curvature should be negative (concave up = valley)
        let kg = d.general_curvature();
        assert!(
            kg < -0.01,
            "General curvature of bowl should be negative (concave), got {:.6}",
            kg
        );

        // Simplified should agree roughly
        let kg_simple = d.general_curvature_simplified();
        assert!(
            kg_simple < -0.01,
            "Simplified curvature should also be negative, got {:.6}",
            kg_simple
        );

        // Gaussian curvature positive for bowl
        let gauss = d.gaussian_curvature();
        assert!(
            gauss > 0.0,
            "Gaussian curvature of bowl should be positive, got {:.6}",
            gauss
        );
    }

    #[test]
    fn test_slope_magnitude() {
        let d = evans_young(tilted_x_window(), CS);
        let s = d.slope_magnitude();
        assert!((s - 2.0).abs() < 0.01, "Slope magnitude should be ~2.0, got {:.4}", s);

        let d_flat = evans_young(flat_window(), CS);
        assert!(d_flat.slope_magnitude() < 1e-10, "Flat slope should be ~0");
    }

    #[test]
    fn test_extract_window_with_nan() {
        let mut data = ndarray::Array2::from_elem((3, 3), 100.0);
        assert!(extract_window(&data, 1, 1).is_some());

        data[[0, 0]] = f64::NAN;
        assert!(extract_window(&data, 1, 1).is_none());
    }

    #[test]
    fn test_methods_agree_on_first_order() {
        // All three methods should give similar p, q for a simple tilted surface
        let w = tilted_x_window();
        let ey = evans_young(w, CS);
        let zt = zevenbergen_thorne(w, CS);
        let h = horn(w, CS);

        assert!(
            (ey.p - zt.p).abs() < 0.5,
            "EY and ZT p should agree: {:.4} vs {:.4}",
            ey.p, zt.p
        );
        assert!(
            (ey.p - h.p).abs() < 0.5,
            "EY and Horn p should agree: {:.4} vs {:.4}",
            ey.p, h.p
        );
    }
}
