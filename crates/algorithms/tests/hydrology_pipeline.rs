//! Integration test for the D8 hydrology pipeline contract.
//!
//! Exercises the full chain fill → flow direction → flow accumulation on a
//! synthetic DEM with an interior sink, and verifies the invariants that
//! tie the modules together through the shared D8 encoding
//! (`surtgis_algorithms::hydrology::d8`):
//!
//! - after Priority-Flood with epsilon > 0, every cell except the single
//!   outlet has a valid D8 direction (1–8);
//! - the outlet accumulates every other cell of the grid;
//! - accumulation is strictly monotonic downstream;
//! - filled elevation is strictly decreasing downstream (positive drop).

use surtgis_algorithms::hydrology::{
    PriorityFloodParams, d8, flow_accumulation, flow_direction, priority_flood,
};
use surtgis_core::raster::{GeoTransform, Raster};

const ROWS: usize = 7;
const COLS: usize = 7;

/// Synthetic DEM: an inclined plane draining towards the SE corner
/// (the outlet, global minimum), with an artificial sink at (2, 2).
fn synthetic_dem_with_sink() -> Raster<f64> {
    let mut dem = Raster::new(ROWS, COLS);
    dem.set_transform(GeoTransform::new(0.0, ROWS as f64, 1.0, -1.0));

    for row in 0..ROWS {
        for col in 0..COLS {
            // Decreases towards (ROWS-1, COLS-1): unique global minimum there.
            let z = ((ROWS - 1 - row) + (COLS - 1 - col)) as f64 * 10.0;
            dem.set(row, col, z).unwrap();
        }
    }

    // Interior depression: much lower than all its neighbors.
    dem.set(2, 2, -50.0).unwrap();
    dem
}

#[test]
fn d8_pipeline_fill_flowdir_accumulation() {
    let dem = synthetic_dem_with_sink();

    // Sanity: on the raw DEM the sink is a pit (direction 0).
    let raw_fdir = flow_direction(&dem).unwrap();
    assert_eq!(
        raw_fdir.get(2, 2).unwrap(),
        0,
        "the unfilled sink must be a pit"
    );

    // 1. Fill with epsilon > 0 so the depression drains.
    let filled = priority_flood(&dem, {
        let mut p = PriorityFloodParams::default();
        p.epsilon = 1e-5;
        p
    })
    .unwrap();

    // The sink must have been raised.
    assert!(
        filled.get(2, 2).unwrap() > dem.get(2, 2).unwrap(),
        "priority_flood must raise the sink cell"
    );
    // Cells outside the depression are untouched.
    assert_eq!(filled.get(0, 0).unwrap(), dem.get(0, 0).unwrap());
    assert_eq!(
        filled.get(ROWS - 1, COLS - 1).unwrap(),
        dem.get(ROWS - 1, COLS - 1).unwrap()
    );

    // 2. Flow direction on the filled DEM.
    let fdir = flow_direction(&filled).unwrap();

    // Exactly one cell without direction: the outlet at the global minimum.
    let mut zero_dir_cells = Vec::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            let dir = fdir.get(row, col).unwrap();
            assert!(dir <= 8, "invalid D8 code {dir} at ({row}, {col})");
            if dir == 0 {
                zero_dir_cells.push((row, col));
            } else {
                // Every non-zero code must decode to a real offset and
                // point to an in-bounds neighbor.
                assert!(d8::decode(dir).is_some());
                assert!(
                    d8::downstream(row, col, dir, ROWS, COLS).is_some(),
                    "direction {dir} at ({row}, {col}) points off-grid"
                );
            }
        }
    }
    assert_eq!(
        zero_dir_cells,
        vec![(ROWS - 1, COLS - 1)],
        "after fill with epsilon > 0 only the outlet may lack a direction"
    );

    // 3. Flow accumulation.
    let facc = flow_accumulation(&fdir).unwrap();

    // The outlet receives every other cell: acc counts *upstream* cells
    // (a headwater cell has acc 0), so outlet == ROWS*COLS - 1.
    let outlet_acc = facc.get(ROWS - 1, COLS - 1).unwrap();
    assert_eq!(
        outlet_acc,
        (ROWS * COLS - 1) as f64,
        "outlet must accumulate all {} upstream cells",
        ROWS * COLS - 1
    );

    // 4. Downstream invariants, walked through the shared d8 module:
    //    accumulation strictly increases and filled elevation strictly
    //    decreases along every flow step.
    for row in 0..ROWS {
        for col in 0..COLS {
            let dir = fdir.get(row, col).unwrap();
            if dir == 0 {
                continue;
            }
            let (nr, nc) = d8::downstream(row, col, dir, ROWS, COLS).unwrap();

            let acc_here = facc.get(row, col).unwrap();
            let acc_down = facc.get(nr, nc).unwrap();
            assert!(
                acc_down >= acc_here + 1.0,
                "accumulation not monotonic: ({row},{col})={acc_here} -> ({nr},{nc})={acc_down}"
            );

            let z_here = filled.get(row, col).unwrap();
            let z_down = filled.get(nr, nc).unwrap();
            assert!(
                z_down < z_here,
                "filled DEM not decreasing downstream: ({row},{col})={z_here} -> ({nr},{nc})={z_down}"
            );
        }
    }

    // 5. Every cell drains to the outlet in a bounded number of steps
    //    (no cycles in the D8 graph).
    for row in 0..ROWS {
        for col in 0..COLS {
            let (mut r, mut c) = (row, col);
            let mut steps = 0;
            loop {
                let dir = fdir.get(r, c).unwrap();
                if dir == 0 {
                    break;
                }
                let (nr, nc) = d8::downstream(r, c, dir, ROWS, COLS).unwrap();
                r = nr;
                c = nc;
                steps += 1;
                assert!(
                    steps <= ROWS * COLS,
                    "cycle detected in flow graph starting at ({row}, {col})"
                );
            }
            assert_eq!(
                (r, c),
                (ROWS - 1, COLS - 1),
                "flow path from ({row}, {col}) must end at the outlet"
            );
        }
    }
}
