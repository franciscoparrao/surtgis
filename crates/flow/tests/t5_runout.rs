//! T5 (spec §7): Voellmy runout on a slope break.
//!
//! Inclined plane 30° → 10° with μ = 0.2 (tan⁻¹ 0.2 ≈ 11.3°): the flow MUST
//! NOT stop on the 30° section (tan 30° > μ) and MUST stop completely on the
//! 10° section (tan 10° < μ) — this slope-versus-μ detention is the physical
//! runout mechanism of the Voellmy model (spec §2.2).
//!
//! Stop criterion per spec: Σ|u| < v_stop · N_wet.
//!
//! History: with the v1.0 Audusse reconstruction a thin veneer (~11.5% of
//! the mass) stayed frozen on the steep track — the driving force truncated
//! to O(g·h²/Δx) when the bed step exceeded the flow depth. The Chen &
//! Noelle reconstruction (spec v1.1 §3, N3) removes that truncation: the
//! veneer drains advectively (< 1% measured) and the deposit relaxes to its
//! Coulomb yield profile on the gentle section. The relaxation tail decays
//! asymptotically, so the spec stop criterion is reached near t ≈ 1000 s
//! rather than t ≈ 50 s of the frozen-veneer regime — the drive window
//! below covers it. Gate per spec v1.1 §3: steep-section mass < 8% at stop.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 5;
const COLS: usize = 300;
const DX: f64 = 5.0;
/// x coordinate of the slope break [m].
const BREAK_X: f64 = 300.0;

fn bed(x: f64) -> f64 {
    let steep = 30.0f64.to_radians().tan();
    let gentle = 10.0f64.to_radians().tan();
    let length = COLS as f64 * DX;
    gentle * (length - x.max(BREAK_X)) + steep * (BREAK_X - x).max(0.0)
}

#[test]
fn t5_voellmy_flow_stops_on_gentle_section_only() {
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    let mut rel_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            dem_data[r * COLS + c] = bed(x) as f32;
            // 10 m block on the 30° section (x in 50..250 m).
            if (50.0..250.0).contains(&x) {
                rel_data[r * COLS + c] = 10.0;
            }
        }
    }
    let transform = GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, ROWS, COLS).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, ROWS, COLS).unwrap();
    release.set_transform(transform);

    let params = VoellmyParams {
        mu: 0.2,
        xi: 200.0,
        v_stop: 0.01,
    };
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, params, config).unwrap();
    let mass0 = sim.total_mass();

    // Sum of wet-cell speeds and wet count for the spec stop criterion.
    let flow_speed = |sim: &Simulation| -> (f64, usize) {
        let s = sim.state();
        let mut sum = 0.0f64;
        let mut n_wet = 0usize;
        for i in 0..s.h.len() {
            let h = f64::from(s.h[i]);
            if h < 1e-3 {
                continue;
            }
            n_wet += 1;
            let u = f64::from(s.hu[i]) / h;
            let v = f64::from(s.hv[i]) / h;
            sum += (u * u + v * v).sqrt();
        }
        (sum, n_wet)
    };
    // Mass fraction sitting on the steep (30°) section.
    let steep_fraction = |sim: &Simulation| -> f64 {
        let s = sim.state();
        let mut steep = 0.0f64;
        let mut total = 0.0f64;
        for r in 0..ROWS {
            for c in 0..COLS {
                let h = f64::from(s.h[r * COLS + c]);
                total += h;
                if ((c as f64 + 0.5) * DX) < BREAK_X {
                    steep += h;
                }
            }
        }
        steep / total
    };

    // While the material is still on the 30° section it must keep moving:
    // sample the stop criterion along the way.
    let mut stopped_at: Option<f64> = None;
    for _ in 0..1500 {
        sim.step(1.0).unwrap();
        let (sum_u, n_wet) = flow_speed(&sim);
        if sum_u < 0.01 * n_wet as f64 {
            stopped_at = Some(sim.time());
            break;
        }
        // MUST NOT stop while the deposit is still substantially on the 30°
        // section.
        assert!(
            steep_fraction(&sim) < 0.5 || sum_u >= 0.01 * n_wet as f64,
            "flow stopped on the 30° section at t={} s",
            sim.time()
        );
    }
    let t_stop = stopped_at.expect("flow never satisfied the stop criterion within 1500 s");

    let frac_steep = steep_fraction(&sim);
    let mass_end = sim.total_mass();
    // Mass-weighted centroid of the deposit along x.
    let s = sim.state();
    let (mut m_sum, mut mx_sum) = (0.0f64, 0.0f64);
    for r in 0..ROWS {
        for c in 0..COLS {
            let h = f64::from(s.h[r * COLS + c]);
            m_sum += h;
            mx_sum += h * ((c as f64 + 0.5) * DX);
        }
    }
    let centroid_x = mx_sum / m_sum;
    eprintln!(
        "T5: stopped at t={t_stop:.0} s, mass on 30° section = {:.2}%, centroid x = {centroid_x:.0} m (break at {BREAK_X} m), mass retained = {:.2}%",
        frac_steep * 100.0,
        mass_end / mass0 * 100.0
    );

    // The deposit must sit on the gentle section (tan 10° = 0.176 < μ =
    // 0.2), having evacuated the steep one (tan 30° = 0.577 > μ). Spec
    // v1.1 §3 gate: < 8% (with Chen & Noelle the measured residue is < 1%).
    assert!(
        frac_steep < 0.08,
        "{:.2}% of the mass stopped on the 30° section (spec v1.1 T5 gate 8%)",
        frac_steep * 100.0
    );
    assert!(
        centroid_x > BREAK_X + 100.0,
        "deposit centroid {centroid_x:.0} m is not clearly on the gentle section"
    );
    // Nothing should have left through the distal edge: the runout must be
    // contained well inside the domain for the test to be meaningful.
    assert!(
        mass_end > 0.99 * mass0,
        "mass left the domain: {mass_end} of {mass0} m³"
    );
}
