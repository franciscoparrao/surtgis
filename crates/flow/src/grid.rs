//! Simulation grid: DEM, slope cosines and the solid (`NoData`) mask.

use surtgis_core::{GeoTransform, Raster};

use crate::FlowError;

/// Relative tolerance for the square-cell check and transform comparison.
const REL_TOL: f64 = 1e-9;

/// Static simulation domain derived from the input DEM (spec §4).
///
/// Holds the bed elevation, the precomputed slope cosines (spec §2.2: computed
/// once at init, never per step), and the solid mask (`NoData` cells act as
/// reflective walls, spec §2.3). Row-major, row 0 = north.
pub struct SimGrid {
    rows: usize,
    cols: usize,
    /// Uniform cell size in metres (square cells enforced at construction).
    cellsize: f64,
    transform: GeoTransform,
    /// Bed elevation per cell; solid cells hold an arbitrary finite value.
    /// With entrainment active this is always the derived `z₀ − e` (see
    /// [`SimGrid::erode`]), never a running subtraction.
    dem: Vec<f32>,
    /// Pristine bed snapshot `z₀`, taken when entrainment activates (one
    /// extra `Vec<f32>` on top of the amended §6 budget's `e_max`/`e`).
    /// `None` until then — the erosion path is the only user.
    dem0: Option<Vec<f32>>,
    /// cosθ of the local slope per cell (spec §2.2), 1.0 on solid cells.
    cos_theta: Vec<f32>,
    /// `NoData` mask: `true` cells are impenetrable walls.
    solid: Vec<bool>,
}

impl SimGrid {
    /// Build the domain from a DEM raster, validating the grid geometry.
    pub(crate) fn from_raster(dem: &Raster<f32>) -> Result<Self, FlowError> {
        let (rows, cols) = dem.shape();
        if rows == 0 || cols == 0 {
            return Err(FlowError::EmptyGrid);
        }
        let t = *dem.transform();
        if t.row_rotation != 0.0 || t.col_rotation != 0.0 {
            return Err(FlowError::RotatedGrid {
                row_rotation: t.row_rotation,
                col_rotation: t.col_rotation,
            });
        }
        let pw = t.pixel_width;
        let ph = t.pixel_height.abs();
        if !pw.is_finite() || !ph.is_finite() || pw <= 0.0 || ph <= 0.0 {
            return Err(FlowError::NonSquareCells {
                pixel_width: t.pixel_width,
                pixel_height: t.pixel_height,
            });
        }
        if (pw - ph).abs() > REL_TOL * pw.max(ph) {
            return Err(FlowError::NonSquareCells {
                pixel_width: t.pixel_width,
                pixel_height: t.pixel_height,
            });
        }

        let n = rows * cols;
        let mut z = vec![0.0f32; n];
        let mut solid = vec![false; n];
        for (i, &v) in dem.data().iter().enumerate() {
            if v.is_finite() && !dem.is_nodata(v) {
                z[i] = v;
            } else {
                solid[i] = true;
            }
        }

        let mut grid = Self {
            rows,
            cols,
            cellsize: pw,
            transform: t,
            dem: z,
            dem0: None,
            cos_theta: vec![1.0; n],
            solid,
        };
        grid.derive_cos_theta();
        Ok(grid)
    }

    /// Replace the bed elevation in place (spec §4 `update_dem`): re-reads
    /// elevations and the `NoData` mask and re-derives cosθ. The caller has
    /// already validated dimensions and transform.
    pub(crate) fn replace_dem(&mut self, dem: &Raster<f32>) {
        for (i, &v) in dem.data().iter().enumerate() {
            if v.is_finite() && !dem.is_nodata(v) {
                self.dem[i] = v;
                self.solid[i] = false;
            } else {
                self.dem[i] = 0.0;
                self.solid[i] = true;
            }
        }
        self.derive_cos_theta();
    }

    /// Check that `other` shares this grid's dimensions and geotransform.
    pub(crate) fn check_compatible(&self, other: &Raster<f32>) -> Result<(), FlowError> {
        let (rows, cols) = other.shape();
        if rows != self.rows || cols != self.cols {
            return Err(FlowError::GridMismatch {
                expected_rows: self.rows,
                expected_cols: self.cols,
                got_rows: rows,
                got_cols: cols,
            });
        }
        let a = &self.transform;
        let b = other.transform();
        let scale = self.cellsize.abs().max(1.0);
        let close = |x: f64, y: f64, s: f64| (x - y).abs() <= REL_TOL * s;
        let origin_scale = a.origin_x.abs().max(a.origin_y.abs()).max(scale);
        if !(close(a.origin_x, b.origin_x, origin_scale)
            && close(a.origin_y, b.origin_y, origin_scale)
            && close(a.pixel_width, b.pixel_width, scale)
            && close(a.pixel_height, b.pixel_height, scale)
            && b.row_rotation == 0.0
            && b.col_rotation == 0.0)
        {
            return Err(FlowError::TransformMismatch);
        }
        Ok(())
    }

    /// cosθ = 1/√(1 + p² + q²) with p = ∂z/∂x, q = ∂z/∂y from central
    /// differences (one-sided next to walls and domain edges).
    fn derive_cos_theta(&mut self) {
        let dx = self.cellsize;
        for r in 0..self.rows {
            for c in 0..self.cols {
                let i = r * self.cols + c;
                if self.solid[i] {
                    self.cos_theta[i] = 1.0;
                    continue;
                }
                let p = self.gradient_1d(r, c, 0, 1, dx);
                let q = self.gradient_1d(r, c, 1, 0, dx);
                self.cos_theta[i] = (1.0 / (1.0 + p * p + q * q).sqrt()) as f32;
            }
        }
    }

    /// Directional bed gradient at (r, c) using central differences where both
    /// neighbours are valid, one-sided otherwise, 0 if isolated.
    fn gradient_1d(&self, r: usize, c: usize, dr: usize, dc: usize, dx: f64) -> f64 {
        let sample = |r: isize, c: isize| -> Option<f64> {
            if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
                return None;
            }
            let i = r as usize * self.cols + c as usize;
            (!self.solid[i]).then(|| f64::from(self.dem[i]))
        };
        let (r, c) = (r as isize, c as isize);
        let (dr, dc) = (dr as isize, dc as isize);
        let here = sample(r, c).unwrap_or(0.0);
        let prev = sample(r - dr, c - dc);
        let next = sample(r + dr, c + dc);
        match (prev, next) {
            (Some(a), Some(b)) => (b - a) / (2.0 * dx),
            (Some(a), None) => (here - a) / dx,
            (None, Some(b)) => (b - here) / dx,
            (None, None) => 0.0,
        }
    }

    /// Number of rows (north-south extent in cells).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (west-east extent in cells).
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Total number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// `true` if the grid holds no cells (never true post-construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Uniform cell size in metres.
    #[must_use]
    pub fn cellsize(&self) -> f64 {
        self.cellsize
    }

    /// Geotransform of the domain (north-up, square cells).
    #[must_use]
    pub fn transform(&self) -> &GeoTransform {
        &self.transform
    }

    /// `true` if the cell is a `NoData` wall.
    #[must_use]
    pub fn is_solid(&self, row: usize, col: usize) -> bool {
        self.solid[row * self.cols + col]
    }

    /// Current bed elevation per cell in metres, row-major (row 0 = north).
    /// With entrainment active this is the eroded bed `z₀ − e`; solid cells
    /// hold an arbitrary finite value (check [`SimGrid::is_solid`]).
    #[must_use]
    pub fn bed(&self) -> &[f32] {
        &self.dem
    }

    /// Snapshot the current bed as the erosion baseline `z₀`. Called when
    /// entrainment activates and again after [`SimGrid::replace_dem`] so
    /// live barriers become the new baseline (spec §4 + v1.1 §2.2).
    pub(crate) fn snapshot_bed(&mut self) {
        self.dem0 = Some(self.dem.clone());
    }

    /// Re-derive the bed from the invariant `z = z₀ − e` (spec v1.1 §2.2),
    /// where `e` is the CUMULATIVE eroded depth per cell.
    ///
    /// A running `z -= Δe` in f32 silently absorbs sub-ulp increments —
    /// with `z` ~ 500-900 m the ulp is 3-6e-5 m, so small Δe were credited
    /// to `h`/`e`/`eroded_volume` but never left the bed (audit R4). The
    /// derived form rounds `z₀ − e` exactly once, so the bed always agrees
    /// with `e` to within half an ulp regardless of how many substeps ran.
    /// `cos θ` stays frozen at its init value — spec v1.1 §2.2 MAY,
    /// documented limitation: metre-scale erosion on ≥10 m cells changes
    /// the Coulomb slope factor only at second order.
    pub(crate) fn erode(&mut self, e_total: &[f32]) {
        use rayon::prelude::*;
        let dem0 = self
            .dem0
            .as_ref()
            .expect("snapshot_bed must run before erode");
        self.dem
            .par_iter_mut()
            .zip_eq(dem0.par_iter())
            .zip_eq(e_total.par_iter())
            .for_each(|((z, &z0), &e)| {
                if e > 0.0 {
                    *z = (f64::from(z0) - f64::from(e)) as f32;
                }
            });
    }

    #[inline]
    pub(crate) fn solid_at(&self, idx: usize) -> bool {
        self.solid[idx]
    }

    #[inline]
    pub(crate) fn z_at(&self, idx: usize) -> f64 {
        f64::from(self.dem[idx])
    }

    #[inline]
    pub(crate) fn cos_theta_at(&self, idx: usize) -> f64 {
        f64::from(self.cos_theta[idx])
    }
}
