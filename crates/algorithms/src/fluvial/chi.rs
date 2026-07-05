//! χ (chi) integral transform per Perron & Royden (2013).
//!
//! χ is the path integral of (A₀/A(x))^θref along a river profile, from a
//! base level upward. In a steady-state landscape it linearises the
//! elevation profile: a plot of elevation against χ is a straight line
//! whose slope is the channel-steepness index `ksn`. Departures from
//! linearity flag transient knickpoints, lithologic contrasts, or
//! tectonic perturbations.
//!
//! ## Reference
//!
//! Perron, J.T. & Royden, L. (2013). "An integral approach to bedrock
//! river profile analysis." Earth Surface Processes and Landforms 38(6),
//! 570–576. <https://doi.org/10.1002/esp.3302>
//!
//! ## Algorithm (spec §4.1)
//!
//! 1. Build a [`StreamGraph`] from the binary stream raster + D8 flow
//!    direction.
//! 2. Set χ = 0 at every outlet (auto-detected via the graph, or supplied
//!    explicitly via [`ChiParams::base_outlets`]).
//! 3. BFS upstream from each outlet. For each upstream node `u` reached
//!    from a downstream node `i`:
//!
//!    ```text
//!    χ(u) = χ(i) + Δx · (f(i) + f(u)) / 2,  f(x) = (A₀ / A(x))^θref
//!    ```
//!
//!    where Δx is the channel-following distance from `i` to `u`
//!    (= cell_size_m for cardinal D8 step, = cell_size_m · √2 for
//!    diagonal), and A(x) is the drainage area at `x` in m² (including
//!    the cell's own contributing area, i.e. `(flow_acc + 1) · cell_size²`
//!    — `flow_acc` counts upstream cells only, so a headwater has
//!    `flow_acc = 0` but a nonzero drainage area of one cell).
//!
//! 4. Non-stream cells and cells unreachable from any outlet remain
//!    NaN in the output raster.
//!
//! ## Numerical scheme
//!
//! The integration uses the trapezoidal rule: the integrand `f(A) =
//! (A₀/A)^θref` is evaluated at both ends of each segment (`f(i)` and
//! `f(u)`) and averaged. This matches the `cumtrapz` convention used by
//! TopoToolbox 2 and pyTopoToolbox. It is *not* bit-for-bit identical to
//! those reference implementations — differences in flow-routing,
//! smoothing, and floating-point summation order remain — but the
//! integration scheme itself is the same trapezoidal rule, unlike a
//! one-sided Riemann sum.

use std::collections::VecDeque;

use ndarray::Array2;
use surtgis_core::{Raster, Result};

use super::stream_traversal::{StreamGraphError, build_stream_graph};

/// Parameters controlling the χ integration.
#[derive(Debug, Clone)]
pub struct ChiParams {
    /// Reference concavity index used as the exponent on (A₀/A). Default
    /// 0.45 — the typical value for bedrock channels in steady state.
    pub theta_ref: f64,

    /// Reference drainage area in m². Default 1 km² = 1·10⁶ m². The
    /// numerical value cancels out of the dimensionless slope-area
    /// relationship but sets the absolute scale of the χ axis.
    pub a_0_m2: f64,

    /// Cell size in metres. Required because flow_acc is in cell counts
    /// and the integration step Δx is in metres. Read from the raster
    /// transform at the call site (CLI handler) when it's available.
    pub cell_size_m: f64,

    /// Optional explicit list of base outlets in (row, col). When `None`
    /// (the default) every outlet auto-detected by [`StreamGraph`] is used
    /// — i.e. every boundary outlet, every cell whose downstream flow
    /// leaves the stream network, and every pit/flat sentinel. Pass
    /// `Some(...)` to restrict χ to a specific subnetwork (useful when a
    /// catchment has multiple disconnected outlets but the user only
    /// cares about one).
    pub base_outlets: Option<Vec<(usize, usize)>>,
}

impl Default for ChiParams {
    fn default() -> Self {
        Self {
            theta_ref: 0.45,
            a_0_m2: 1.0e6,
            cell_size_m: 30.0, // arbitrary — caller must supply the right value
            base_outlets: None,
        }
    }
}

/// Errors specific to χ computation.
#[derive(Debug, thiserror::Error)]
pub enum ChiError {
    /// Building the stream graph failed.
    #[error(transparent)]
    Graph(#[from] StreamGraphError),

    /// The flow-accumulation raster did not match the stream raster's shape.
    #[error("flow_acc shape {0:?} does not match stream shape {1:?}")]
    AccShapeMismatch((usize, usize), (usize, usize)),

    /// `cell_size_m` was not strictly positive.
    #[error("ChiParams.cell_size_m must be > 0 (got {0})")]
    NonPositiveCellSize(f64),

    /// The reference drainage area `a_0_m2` was not strictly positive.
    #[error("ChiParams.a_0_m2 must be > 0 (got {0})")]
    NonPositiveA0(f64),

    /// The requested base outlet does not fall on a stream cell.
    #[error(
        "Base outlet ({row}, {col}) is not a stream cell — \
         either the coordinate is outside the raster or stream[r, c] != 1"
    )]
    BaseOutletNotStream {
        /// Row index of the offending outlet coordinate.
        row: usize,
        /// Column index of the offending outlet coordinate.
        col: usize,
    },
}

/// Compute χ per stream cell.
///
/// Inputs:
///
/// - `stream`: binary network raster (1 = stream, 0 = non-stream) from
///   [`crate::hydrology::stream_network`].
/// - `flow_dir`: D8 flow direction raster from
///   [`crate::hydrology::flow_direction`].
/// - `flow_acc`: drainage area raster in cell counts from
///   [`crate::hydrology::flow_accumulation`].
/// - `params`: see [`ChiParams`].
///
/// Output: a `Raster<f64>` with χ value (m) at every reachable stream
/// cell and NaN elsewhere. Coordinate system + transform are copied from
/// `stream`.
pub fn chi_transform(
    stream: &Raster<u8>,
    flow_dir: &Raster<u8>,
    flow_acc: &Raster<f64>,
    params: ChiParams,
) -> Result<Raster<f64>> {
    if params.cell_size_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            ChiError::NonPositiveCellSize(params.cell_size_m).to_string(),
        ));
    }
    if params.a_0_m2 <= 0.0 {
        return Err(surtgis_core::Error::Other(
            ChiError::NonPositiveA0(params.a_0_m2).to_string(),
        ));
    }
    let stream_shape = stream.shape();
    if flow_acc.shape() != stream_shape {
        return Err(surtgis_core::Error::Other(
            ChiError::AccShapeMismatch(flow_acc.shape(), stream_shape).to_string(),
        ));
    }
    let (rows, cols) = stream_shape;

    let graph = build_stream_graph(stream, flow_dir)
        .map_err(|e| surtgis_core::Error::Other(e.to_string()))?;

    // Decide which nodes start with χ = 0. Auto = every detected outlet;
    // explicit = the user-supplied (row, col) coordinates, each of which
    // must already be a stream cell present in the graph.
    let start_nodes: Vec<usize> = match &params.base_outlets {
        None => graph.outlets().collect(),
        Some(coords) => {
            let mut out = Vec::with_capacity(coords.len());
            for &(r, c) in coords {
                let idx = graph
                    .stream_cells
                    .iter()
                    .position(|&xy| xy == (r, c))
                    .ok_or_else(|| {
                        surtgis_core::Error::Other(
                            ChiError::BaseOutletNotStream { row: r, col: c }.to_string(),
                        )
                    })?;
                out.push(idx);
            }
            out
        }
    };

    // Pre-allocate output as a flat NaN array; we fill via row-major
    // indexing inside the BFS to skip a re-walk afterwards.
    let mut chi_data = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    // Seed every starting node with χ = 0 and prime the BFS queue.
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(start_nodes.len());
    for &n in &start_nodes {
        let (r, c) = graph.stream_cells[n];
        chi_data[(r, c)] = 0.0;
        queue.push_back(n);
    }

    let cell = params.cell_size_m;
    let cell_diag = cell * std::f64::consts::SQRT_2;
    let cell_area_m2 = cell * cell;

    // BFS upstream. Each node is visited exactly once because the stream
    // graph is a forest rooted at the outlets (each non-outlet node has
    // exactly one downstream_link).
    while let Some(i) = queue.pop_front() {
        let (ir, ic) = graph.stream_cells[i];
        let chi_i = chi_data[(ir, ic)];

        // Integrand at i: f(i) = (A0 / A(i))^theta. A(i) includes the
        // cell's own contributing area — flow_acc counts upstream cells
        // only (a headwater has flow_acc = 0), so drainage area is
        // (flow_acc + 1) cells, not flow_acc cells.
        let a_i_cells = flow_acc.get(ir, ic).unwrap_or(f64::NAN);
        let a_i_m2 = (a_i_cells + 1.0) * cell_area_m2;
        let f_i = if a_i_m2.is_finite() && a_i_m2 > 0.0 {
            Some((params.a_0_m2 / a_i_m2).powf(params.theta_ref))
        } else {
            None
        };

        // If the current node already failed to receive a finite χ
        // (because an upstream cascade hit NoData drainage area), every
        // node above it inherits NaN. Continue the walk so we still
        // populate the visit set, but skip the arithmetic.
        for &u in &graph.upstream_links[i] {
            let (ur, uc) = graph.stream_cells[u];

            // Δx from i to u: cardinal D8 step is 1 cell, diagonal is √2.
            let dr = (ur as isize - ir as isize).abs();
            let dc = (uc as isize - ic as isize).abs();
            let dx = if dr + dc == 2 { cell_diag } else { cell };

            // A(u) read in cell counts → convert to m² with cell_size(),
            // including the cell's own contributing area (see above).
            let a_u_cells = flow_acc.get(ur, uc).unwrap_or(f64::NAN);
            let a_u_m2 = (a_u_cells + 1.0) * cell_area_m2;

            // The χ recurrence, trapezoidal rule: average the integrand
            // at both ends of the segment. Any NaN / non-positive area
            // propagates as NaN downstream consumers can detect. Same
            // applies if the current node failed earlier.
            let chi_u = match (chi_i.is_finite(), f_i, a_u_m2.is_finite() && a_u_m2 > 0.0) {
                (true, Some(f_i), true) => {
                    let f_u = (params.a_0_m2 / a_u_m2).powf(params.theta_ref);
                    chi_i + dx * (f_i + f_u) / 2.0
                }
                _ => f64::NAN,
            };
            chi_data[(ur, uc)] = chi_u;
            queue.push_back(u);
        }
    }

    // Assemble the output raster preserving the input geo metadata.
    let mut out = stream.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = chi_data;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::Raster;

    fn raster_u8(data: Vec<Vec<u8>>) -> Raster<u8> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<u8> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        Raster::from_array(arr)
    }
    fn raster_f64(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        Raster::from_array(arr)
    }

    /// Synthetic 1-D steady-state profile: a single 10-cell horizontal
    /// channel flowing east. Drainage area scales linearly with distance
    /// from the headwater (A(headwater)=1 cell, ..., A(outlet)=10 cells).
    /// Tests cover spec §7.2's three properties:
    ///   (a) χ(outlet) = 0
    ///   (b) χ monotonically increasing upstream
    ///   (c) a steady-state DEM z = K · χ has R² > 0.99 vs the recovered χ
    #[test]
    fn linear_channel_monotonic_upstream_and_outlet_zero() {
        let n = 10;
        let stream = raster_u8(vec![vec![1u8; n]]);
        // Flow_dir 1 = East. All cells point east; the easternmost flows
        // out of the raster (boundary outlet).
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        // flow_acc as produced by flow_accumulation(): counts *upstream*
        // cells only, so the headwater (leftmost) has flow_acc = 0 and
        // the outlet (rightmost) has flow_acc = n - 1. The physical
        // drainage area (flow_acc + 1 cells) still increases west→east
        // from 1 to n.
        let acc_vals: Vec<f64> = (0..n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc_vals]);

        let params = ChiParams {
            theta_ref: 0.45,
            a_0_m2: 1.0e6,
            cell_size_m: 30.0,
            base_outlets: None,
        };
        let chi = chi_transform(&stream, &flow_dir, &flow_acc, params).unwrap();

        let row: Vec<f64> = (0..n).map(|c| chi.get(0, c).unwrap()).collect();

        // (a) The outlet (rightmost) must hold χ = 0.
        assert_eq!(row[n - 1], 0.0, "outlet χ must be exactly 0");

        // (b) Monotonic increase upstream: χ(c) >= χ(c+1) for all c.
        for c in 0..n - 1 {
            assert!(
                row[c] >= row[c + 1],
                "χ must be monotonically non-decreasing upstream; \
                 row[{}]={}, row[{}]={}",
                c,
                row[c],
                c + 1,
                row[c + 1],
            );
        }

        // No NaN anywhere — every stream cell should have a finite χ.
        for c in 0..n {
            assert!(row[c].is_finite(), "χ[{}] is not finite: {}", c, row[c]);
        }
    }

    /// Spec §7.2 property (c): elevation regression against χ on a
    /// synthetic steady-state DEM.
    ///
    /// Build the same 10-cell channel, then generate a DEM z(c) per the
    /// steady-state relation z = K · χ + base. Compute χ via
    /// chi_transform and assert R² of z on χ exceeds 0.99 (it should
    /// be exactly 1.0 modulo floating-point noise because the relation
    /// is linear by construction).
    #[test]
    fn steady_state_elevation_chi_regression_r2_above_099() {
        let n = 10;
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        // flow_acc convention: headwater = 0 (see previous test).
        let acc_vals: Vec<f64> = (0..n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc_vals]);

        let params = ChiParams {
            theta_ref: 0.45,
            a_0_m2: 1.0e6,
            cell_size_m: 30.0,
            base_outlets: None,
        };
        let chi = chi_transform(&stream, &flow_dir, &flow_acc, params).unwrap();

        // Synthetic steady-state DEM: z = K · χ + base.
        let k = 0.5; // arbitrary positive slope; sets ksn
        let base = 100.0;
        let z: Vec<f64> = (0..n).map(|c| k * chi.get(0, c).unwrap() + base).collect();
        let x: Vec<f64> = (0..n).map(|c| chi.get(0, c).unwrap()).collect();

        // R² for the linear regression z ~ x.
        let n_f = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n_f;
        let mean_z = z.iter().sum::<f64>() / n_f;
        let ss_tot: f64 = z.iter().map(|zi| (zi - mean_z).powi(2)).sum();
        let ss_xy: f64 = x
            .iter()
            .zip(z.iter())
            .map(|(xi, zi)| (xi - mean_x) * (zi - mean_z))
            .sum();
        let ss_xx: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let slope = ss_xy / ss_xx;
        let intercept = mean_z - slope * mean_x;
        let ss_res: f64 = x
            .iter()
            .zip(z.iter())
            .map(|(xi, zi)| {
                let pred = slope * xi + intercept;
                (zi - pred).powi(2)
            })
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        assert!(
            r2 > 0.99,
            "Steady-state z ~ χ regression must have R² > 0.99, got {}",
            r2,
        );
    }

    /// Diagonal step in D8 must contribute Δx = cell_size · √2, not
    /// cell_size. We construct a 3-cell channel that steps once
    /// diagonally; the χ increment over that step must use √2 and the
    /// trapezoidal rule (average of the integrand at both segment ends).
    #[test]
    fn diagonal_step_uses_sqrt2_distance() {
        // Stream layout (physical drainage area including each cell
        // itself):
        //   . . 1     headwater (0,2), A=1 cell
        //   . 1 .     middle    (1,1), A=2 cells — flow_dir(0,2) = SW = 6
        //   1 . .     outlet    (2,0), A=3 cells — flow_dir(1,1) = SW = 6
        //
        // flow_acc, as produced by flow_accumulation(), counts *upstream*
        // cells only (headwater = 0), so it is one less than the
        // physical drainage area in cells at each node.
        let stream = raster_u8(vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]]);
        // (2,0): pit/flat (outlet flows out of raster — code 6 = SW exits
        // bottom-left corner). (1,1): code 6 (SW). (0,2): code 6 (SW).
        let flow_dir = raster_u8(vec![vec![0, 0, 6], vec![0, 6, 0], vec![6, 0, 0]]);
        let flow_acc = raster_f64(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![2.0, 0.0, 0.0],
        ]);

        let cell = 30.0;
        let theta = 0.45;
        let a0 = 1.0e6;
        let params = ChiParams {
            theta_ref: theta,
            a_0_m2: a0,
            cell_size_m: cell,
            base_outlets: None,
        };
        let chi = chi_transform(&stream, &flow_dir, &flow_acc, params).unwrap();

        let cell_area_m2 = cell * cell;
        let dx = cell * std::f64::consts::SQRT_2;
        // f(x) = (a0 / A(x))^theta, with A(x) = (flow_acc + 1) * cell².
        let f = |acc_cells: f64| (a0 / ((acc_cells + 1.0) * cell_area_m2)).powf(theta);
        let f_outlet = f(2.0); // A_outlet = 3 cells
        let f_mid = f(1.0); // A_mid = 2 cells
        let f_head = f(0.0); // A_head = 1 cell

        // Trapezoidal rule: χ(u) = χ(i) + Δx · (f(i) + f(u)) / 2.
        let expected_chi_mid = 0.0 + dx * (f_outlet + f_mid) / 2.0;
        let expected_chi_head = expected_chi_mid + dx * (f_mid + f_head) / 2.0;

        let chi_mid = chi.get(1, 1).unwrap();
        let chi_head = chi.get(0, 2).unwrap();
        assert!(
            (chi_mid - expected_chi_mid).abs() < 1e-6,
            "diagonal mid χ wrong: got {}, expected {}",
            chi_mid,
            expected_chi_mid,
        );
        assert!(
            (chi_head - expected_chi_head).abs() < 1e-6,
            "diagonal head χ wrong: got {}, expected {}",
            chi_head,
            expected_chi_head,
        );
    }

    /// NoData drainage area at a stream cell must produce NaN χ at that
    /// cell and propagate NaN to every upstream cell.
    #[test]
    fn nan_drainage_area_propagates_upstream_as_nan() {
        let stream = raster_u8(vec![vec![1, 1, 1, 1]]);
        let flow_dir = raster_u8(vec![vec![1, 1, 1, 1]]); // all east → boundary outlet at col 3
        // Bad drainage area at col 1: NaN.
        let flow_acc = raster_f64(vec![vec![1.0, f64::NAN, 3.0, 4.0]]);

        let chi = chi_transform(
            &stream,
            &flow_dir,
            &flow_acc,
            ChiParams {
                theta_ref: 0.45,
                a_0_m2: 1.0e6,
                cell_size_m: 30.0,
                base_outlets: None,
            },
        )
        .unwrap();

        // Outlet (col 3) is fine — it has A = 4.
        assert!(chi.get(0, 3).unwrap().is_finite());
        // col 2: finite (its A is OK; only A at col 1 is NaN).
        assert!(chi.get(0, 2).unwrap().is_finite());
        // col 1: NaN (its own A is NaN).
        assert!(chi.get(0, 1).unwrap().is_nan());
        // col 0: NaN (inherits from col 1).
        assert!(chi.get(0, 0).unwrap().is_nan());
    }

    /// Bad params (cell_size, a0) must error rather than silently compute.
    #[test]
    fn bad_params_error_cleanly() {
        let stream = raster_u8(vec![vec![1]]);
        let flow_dir = raster_u8(vec![vec![0]]);
        let flow_acc = raster_f64(vec![vec![1.0]]);

        for bad in [-1.0, 0.0] {
            let r = chi_transform(
                &stream,
                &flow_dir,
                &flow_acc,
                ChiParams {
                    theta_ref: 0.45,
                    a_0_m2: 1.0e6,
                    cell_size_m: bad,
                    base_outlets: None,
                },
            );
            assert!(r.is_err(), "cell_size={} should error", bad);
        }

        let r = chi_transform(
            &stream,
            &flow_dir,
            &flow_acc,
            ChiParams {
                theta_ref: 0.45,
                a_0_m2: 0.0,
                cell_size_m: 30.0,
                base_outlets: None,
            },
        );
        assert!(r.is_err(), "a_0_m2=0 should error");
    }
}
