//! Flow accumulation algorithm
//!
//! Calculates the number of upstream cells flowing into each cell
//! based on D8 flow direction. This represents the upstream
//! contributing area (in cell counts).

use ndarray::Array2;
use surtgis_core::raster::{Raster, RasterElement};
use surtgis_core::{Error, Result};

use super::d8::D8_OFFSETS;

/// Calculate flow accumulation from a D8 flow direction raster.
///
/// Each cell receives a count of all upstream cells that flow into it.
/// Headwater cells (no upstream neighbors) have accumulation = 0.
///
/// # Algorithm
/// 1. Count incoming flows for each cell (in-degree)
/// 2. Start from cells with in-degree 0 (headwaters)
/// 3. Propagate downstream, accumulating counts
///
/// # Arguments
/// * `flow_dir` - D8 flow direction raster (output from `flow_direction`)
///
/// # Returns
/// `Raster<f64>` with flow accumulation values
pub fn flow_accumulation(flow_dir: &Raster<u8>) -> Result<Raster<f64>> {
    accumulate(flow_dir, |_| 1.0, false)
}

/// Weighted D8 flow accumulation: each cell receives the sum of the
/// weights of every cell upstream of it.
///
/// This is the "source → propagation" operator: with a susceptibility
/// raster as weights, a cell gets the susceptible source area draining
/// through it; with runoff, sediment or pollutant loads, the load
/// routed to it. Cells whose weight is NaN or the weight raster's
/// nodata contribute nothing but still pass upstream flow through.
///
/// With `include_self = false` a cell's own weight is not added, so
/// weights of 1 everywhere reproduce [`flow_accumulation`] exactly (the
/// count of upstream cells). With `include_self = true` it is added,
/// which is the convention of `terra::flowAccumulation(weight = …)`,
/// ArcGIS *Flow Accumulation* plus the input weight, and WhiteboxTools'
/// `D8FlowAccumulation`.
///
/// Memory: one `f64` per cell for the result plus the weights as given
/// (read them as `f32` to halve that), one byte per cell of in-degree,
/// and a queue of 4-byte cell indices.
///
/// # Errors
/// When `weights` and `flow_dir` differ in shape.
pub fn flow_accumulation_weighted<T: RasterElement>(
    flow_dir: &Raster<u8>,
    weights: &Raster<T>,
    include_self: bool,
) -> Result<Raster<f64>> {
    if weights.shape() != flow_dir.shape() {
        return Err(Error::Other(format!(
            "weights raster is {:?}, flow direction is {:?}; they must share one grid",
            weights.shape(),
            flow_dir.shape()
        )));
    }
    let cols = flow_dir.shape().1;
    let nodata = weights.nodata();
    let data = weights.data();
    accumulate(
        flow_dir,
        |idx| {
            let v = data[(idx / cols, idx % cols)];
            if v.is_nodata(nodata) {
                return 0.0;
            }
            match v.to_f64() {
                Some(w) if w.is_finite() => w,
                _ => 0.0,
            }
        },
        include_self,
    )
}

/// Shared D8 accumulation core: topological sweep from headwaters
/// (Kahn), each cell passing `acc + weight` to its downstream neighbour.
/// Direction 0 (pit, flat outlet or nodata) and codes outside 1..=8 stop
/// the flow; so does leaving the grid.
fn accumulate<F: Fn(usize) -> f64>(
    flow_dir: &Raster<u8>,
    weight: F,
    include_self: bool,
) -> Result<Raster<f64>> {
    let (rows, cols) = flow_dir.shape();
    let n = rows * cols;
    if n > u32::MAX as usize {
        return Err(Error::Other(format!(
            "flow accumulation supports up to {} cells; this grid has {n}",
            u32::MAX
        )));
    }

    let downstream = |idx: usize| -> Option<usize> {
        let (row, col) = (idx / cols, idx % cols);
        let dir = unsafe { flow_dir.get_unchecked(row, col) };
        if dir == 0 || dir as usize > D8_OFFSETS.len() {
            return None;
        }
        let (dr, dc) = D8_OFFSETS[(dir - 1) as usize];
        let nr = row as isize + dr;
        let nc = col as isize + dc;
        if nr < 0 || nc < 0 || nr as usize >= rows || nc as usize >= cols {
            return None;
        }
        Some(nr as usize * cols + nc as usize)
    };

    // Step 1: in-degree (a D8 cell has at most 8 donors, so one byte).
    let mut in_degree = vec![0u8; n];
    for idx in 0..n {
        if let Some(d) = downstream(idx) {
            in_degree[d] += 1;
        }
    }

    // Step 2: headwaters (in-degree 0) seed the queue, in row-major order.
    let mut queue: Vec<u32> = (0..n)
        .filter(|&idx| in_degree[idx] == 0)
        .map(|idx| idx as u32)
        .collect();
    let mut accumulation = vec![0.0f64; n];

    // Step 3: topological sweep; each cell passes its accumulation plus
    // its own weight downstream once all its donors are done.
    while let Some(idx) = queue.pop() {
        let idx = idx as usize;
        let Some(d) = downstream(idx) else {
            continue;
        };
        accumulation[d] += accumulation[idx] + weight(idx);
        in_degree[d] -= 1;
        if in_degree[d] == 0 {
            queue.push(d as u32);
        }
    }
    drop(queue);
    drop(in_degree);

    if include_self {
        for (idx, a) in accumulation.iter_mut().enumerate() {
            *a += weight(idx);
        }
    }

    let mut output = flow_dir.with_same_meta::<f64>(rows, cols);
    *output.data_mut() = Array2::from_shape_vec((rows, cols), accumulation)
        .map_err(|e| Error::Other(format!("flow accumulation: {e}")))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrology::flow_direction::flow_direction;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_flow_accumulation_linear() {
        // 1x5 strip sloping east: all flow goes E
        // Cell 0 → Cell 1 → Cell 2 → Cell 3 → Cell 4
        // Acc:  0      1      2       3       4
        let mut dem = Raster::new(1, 5);
        dem.set_transform(GeoTransform::new(0.0, 1.0, 1.0, -1.0));

        for col in 0..5 {
            dem.set(0, col, (5 - col) as f64).unwrap();
        }

        let fdir = flow_direction(&dem).unwrap();
        let acc = flow_accumulation(&fdir).unwrap();

        assert_eq!(acc.get(0, 0).unwrap(), 0.0); // Headwater
        assert_eq!(acc.get(0, 1).unwrap(), 1.0);
        assert_eq!(acc.get(0, 2).unwrap(), 2.0);
        assert_eq!(acc.get(0, 3).unwrap(), 3.0);
        assert_eq!(acc.get(0, 4).unwrap(), 4.0); // Outlet
    }

    #[test]
    fn test_flow_accumulation_convergent() {
        // 3x3 DEM with center lowest - all flow converges to center
        //  5 5 5
        //  5 1 5
        //  5 5 5
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap();

        let fdir = flow_direction(&dem).unwrap();
        let acc = flow_accumulation(&fdir).unwrap();

        // Center should receive flow from all 8 neighbors
        let center = acc.get(1, 1).unwrap();
        assert_eq!(
            center, 8.0,
            "Center should accumulate all 8 neighbors, got {}",
            center
        );
    }

    #[test]
    fn test_flow_accumulation_plane() {
        // 5x5 plane sloping south: each row accumulates from rows above
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let acc = flow_accumulation(&fdir).unwrap();

        // Top row cells should have accumulation = 0
        for col in 0..5 {
            assert_eq!(
                acc.get(0, col).unwrap(),
                0.0,
                "Top row should have 0 accumulation"
            );
        }

        // Bottom row should have highest accumulation
        let bottom_center = acc.get(4, 2).unwrap();
        assert!(
            bottom_center >= 4.0,
            "Bottom center should have high accumulation, got {}",
            bottom_center
        );
    }
    fn sloped_dem(rows: usize, cols: usize) -> Raster<f64> {
        // Tilted, rough surface so directions vary and flow converges.
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for r in 0..rows {
            for c in 0..cols {
                let z =
                    1000.0 - 3.0 * r as f64 - 2.0 * c as f64 + ((r * 7 + c * 13) % 11) as f64 * 0.7;
                dem.set(r, c, z).unwrap();
            }
        }
        dem
    }

    #[test]
    fn weighted_with_unit_weights_equals_count() {
        let fdir = flow_direction(&sloped_dem(40, 30)).unwrap();
        let ones: Raster<f32> = {
            let mut w = Raster::new(40, 30);
            w.data_mut().fill(1.0);
            w
        };
        let count = flow_accumulation(&fdir).unwrap();
        let weighted = flow_accumulation_weighted(&fdir, &ones, false).unwrap();
        assert_eq!(count.data(), weighted.data());
        let with_self = flow_accumulation_weighted(&fdir, &ones, true).unwrap();
        for (a, b) in count.data().iter().zip(with_self.data()) {
            assert_eq!(a + 1.0, *b);
        }
    }

    #[test]
    fn weighted_matches_brute_force_upstream_sum() {
        let (rows, cols) = (25, 20);
        let fdir = flow_direction(&sloped_dem(rows, cols)).unwrap();
        let mut w: Raster<f64> = Raster::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                w.set(r, c, ((r * 31 + c * 17) % 7) as f64 * 0.25).unwrap();
            }
        }
        // Weight nodata (NaN) contributes nothing but flow passes through.
        w.set(3, 4, f64::NAN).unwrap();
        w.set_nodata(Some(f64::NAN));
        let acc = flow_accumulation_weighted(&fdir, &w, true).unwrap();

        // Brute force: follow every cell's D8 path downstream and add its
        // weight to every cell on the path (itself included).
        let mut want = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let wv = w.get(r, c).unwrap();
                let wv = if wv.is_nan() { 0.0 } else { wv };
                let (mut rr, mut cc) = (r as isize, c as isize);
                let mut steps = 0;
                loop {
                    want[rr as usize * cols + cc as usize] += wv;
                    let dir = fdir.get(rr as usize, cc as usize).unwrap();
                    if dir == 0 || dir > 8 {
                        break;
                    }
                    let (dr, dc) = D8_OFFSETS[(dir - 1) as usize];
                    let (nr, nc) = (rr + dr, cc + dc);
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        break;
                    }
                    rr = nr;
                    cc = nc;
                    steps += 1;
                    assert!(steps <= rows * cols, "cycle in D8 directions");
                }
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                let got = acc.get(r, c).unwrap();
                let exp = want[r * cols + c];
                assert!((got - exp).abs() < 1e-9, "({r},{c}): {got} vs {exp}");
            }
        }
    }

    #[test]
    fn weighted_rejects_mismatched_grid() {
        let fdir = flow_direction(&sloped_dem(10, 10)).unwrap();
        let w: Raster<f32> = Raster::new(10, 11);
        assert!(flow_accumulation_weighted(&fdir, &w, false).is_err());
    }
}
