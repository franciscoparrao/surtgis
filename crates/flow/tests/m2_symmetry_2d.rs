//! M2 supplementary test: 2D symmetry of the full scheme.
//!
//! A circular water column collapsing on a flat frictionless bed must stay
//! symmetric under x-mirror, y-mirror and transpose. Any x/y mix-up in the
//! flux rotation, face orientation or momentum accumulation breaks these
//! symmetries at O(0.1 m); FP reassociation only allows ~1e-6 m. Not a spec
//! §7 gate, but it pins down exactly the bug class a 2D solver invites.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const N: usize = 121; // odd: a well-defined centre cell
const DX: f64 = 1.0;

#[test]
fn m2_circular_dam_break_is_symmetric() {
    let centre = (N as f64) / 2.0; // in cell units, grid midpoint
    let mut rel_data = vec![0.0f32; N * N];
    for r in 0..N {
        for c in 0..N {
            let dr = (r as f64 + 0.5) - centre;
            let dc = (c as f64 + 0.5) - centre;
            if (dr * dr + dc * dc).sqrt() * DX <= 15.0 {
                rel_data[r * N + c] = 2.0;
            }
        }
    }
    let transform = GeoTransform::new(0.0, N as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(vec![0.0f32; N * N], N, N).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, N, N).unwrap();
    release.set_transform(transform);

    let params = VoellmyParams {
        mu: 0.0,
        xi: f32::INFINITY,
        v_stop: 0.0,
    };
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 10_000,
    };
    let mut sim = Simulation::new(&dem, &release, params, config).unwrap();
    sim.step(5.0).unwrap();

    let h = &sim.state().h;
    let get = |r: usize, c: usize| f64::from(h[r * N + c]);
    let mut max_mirror_x = 0.0f64;
    let mut max_mirror_y = 0.0f64;
    let mut max_transpose = 0.0f64;
    for r in 0..N {
        for c in 0..N {
            max_mirror_x = max_mirror_x.max((get(r, c) - get(r, N - 1 - c)).abs());
            max_mirror_y = max_mirror_y.max((get(r, c) - get(N - 1 - r, c)).abs());
            max_transpose = max_transpose.max((get(r, c) - get(c, r)).abs());
        }
    }
    eprintln!(
        "M2 symmetry: mirror-x {max_mirror_x:.2e}, mirror-y {max_mirror_y:.2e}, transpose {max_transpose:.2e} m"
    );
    assert!(max_mirror_x < 1e-4, "x-mirror asymmetry {max_mirror_x:e} m");
    assert!(max_mirror_y < 1e-4, "y-mirror asymmetry {max_mirror_y:e} m");
    assert!(
        max_transpose < 1e-4,
        "transpose asymmetry {max_transpose:e} m"
    );

    // The column must actually have collapsed and spread.
    let centre_h = get(N / 2, N / 2);
    assert!(
        centre_h < 1.0,
        "centre depth {centre_h} m: the column did not collapse"
    );
}
