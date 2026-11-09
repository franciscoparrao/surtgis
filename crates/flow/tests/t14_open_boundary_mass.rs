//! T14 (audit R4): mass accounting through transmissive (open) borders.
//!
//! The transmissive ghost mirrors the edge cell's state *including its
//! velocity*. When the edge velocity points inward — a release sitting
//! against the edge of a cropped DEM, accelerating away from it — the ghost
//! feeds mass into the domain every substep. That exchange is legitimate
//! modelling (the flow field continues past the crop), but before the R4
//! fix it was invisible to the mass books: without entrainment it was
//! silent mass creation (T4 only covers closed borders); with entrainment
//! it tripped `MassBudgetViolated` almost immediately, with an error
//! docstring blaming the solver.
//!
//! Scenario: 20° slope descending EAST, release flush against the WEST
//! edge (open, no NoData ring). The flow accelerates east; the west edge
//! cells keep an inward (eastward) velocity while wet, so the west ghost
//! is a mass source. This is exactly the "hurried Macul run" geometry the
//! audit warned about.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{EntrainmentParams, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 60;
const COLS: usize = 120;
const DX: f64 = 10.0;

/// Slope descending east (away from the open west edge), no closed ring.
fn open_dem() -> Raster<f32> {
    let down = 20.0f64.to_radians().tan();
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            data[r * COLS + c] = ((COLS as f64 * DX - x) * down) as f32;
        }
    }
    let mut dem = Raster::from_vec(data, ROWS, COLS).unwrap();
    dem.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    dem
}

/// Release flush against the west edge (col 0), mid-height.
fn edge_release() -> Raster<f32> {
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 20..40 {
        for c in 0..15 {
            data[r * COLS + c] = 3.0;
        }
    }
    let mut rel = Raster::from_vec(data, ROWS, COLS).unwrap();
    rel.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    rel
}

fn config() -> SolverConfig {
    SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    }
}

#[test]
fn t14_boundary_volume_closes_the_mass_balance() {
    // Without entrainment: total_mass(t) - release must equal the net
    // volume exchanged through the open edges, up to f32 rounding — i.e.
    // `boundary_volume()` turns the T4 conservation statement into one
    // that also holds on open domains.
    let mut sim = Simulation::new(
        &open_dem(),
        &edge_release(),
        VoellmyParams::default(),
        config(),
    )
    .unwrap();
    let mass0 = sim.total_mass();
    assert!(mass0 > 0.0);
    assert_eq!(sim.boundary_volume(), 0.0);

    for _ in 0..15 {
        sim.step(2.0).unwrap();
    }

    let residual = (sim.total_mass() - mass0 - sim.boundary_volume()).abs() / mass0;
    eprintln!(
        "T14a: mass {mass0:.1} -> {:.1} m³, boundary_net = {:+.1} m³, balance residual = {residual:.3e}",
        sim.total_mass(),
        sim.boundary_volume()
    );
    assert!(
        residual < 1e-4,
        "open-domain mass balance violated: residual {residual:e} (mass {:.1}, boundary {:+.1})",
        sim.total_mass(),
        sim.boundary_volume()
    );

    // The west ghost must have fed mass IN while the edge stayed wet with
    // inward velocity — the audit's silent-mass-creation signature. If this
    // stops holding, the boundary contract changed and T14b's premise dies
    // with it, so assert it explicitly.
    assert!(
        sim.boundary_volume() > 0.0,
        "expected net inflow from the west ghost, got {:+.1} m³",
        sim.boundary_volume()
    );
}

#[test]
fn t14_entrainment_run_survives_an_open_boundary() {
    // With entrainment: the same geometry used to freeze the run with a
    // spurious `MassBudgetViolated` (flow volume > release + eroded,
    // because the ghost's inflow was not on the budget side). With the
    // boundary term accounted, the run must simply proceed.
    let dem = open_dem();
    let release = edge_release();
    let mut emax_data = vec![1.0f32; ROWS * COLS];
    // Keep a rim of non-erodible cells irrelevant to the point of the test.
    for c in 0..COLS {
        emax_data[c] = 0.0;
        emax_data[(ROWS - 1) * COLS + c] = 0.0;
    }
    let mut emax = Raster::from_vec(emax_data, ROWS, COLS).unwrap();
    emax.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));

    let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config()).unwrap();
    sim.set_erodible(&emax, EntrainmentParams::default())
        .unwrap();
    let mass0 = sim.total_mass();

    for step in 0..15 {
        sim.step(2.0)
            .unwrap_or_else(|e| panic!("step {step} failed on an open boundary: {e}"));
    }

    // And the extended balance closes: flow = release + eroded + boundary.
    let residual =
        (sim.total_mass() - mass0 - sim.total_eroded() - sim.boundary_volume()).abs() / mass0;
    eprintln!(
        "T14b: mass {mass0:.1} -> {:.1} m³, eroded = {:.1} m³, boundary_net = {:+.1} m³, residual = {residual:.3e}",
        sim.total_mass(),
        sim.total_eroded(),
        sim.boundary_volume()
    );
    assert!(
        residual < 1e-3,
        "entrainment open-domain balance violated: residual {residual:e}"
    );
}
