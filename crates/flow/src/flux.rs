//! HLL numerical flux with hydrostatic reconstruction (spec §3.3–§3.4).
//!
//! All face computations run in f64 from the f32 state; the update is stored
//! back to f32 by the stepper. The flux function is pure and deterministic:
//! recomputing a face from either adjacent cell yields bit-identical results,
//! which is what makes the recompute-at-band-boundaries strategy of spec §6
//! (and the per-cell face iteration of this first-order solver) sound.

use crate::G;

/// One side of a face: depth, face-normal velocity, transverse velocity and
/// bed elevation, already rotated so `un` is normal to the face.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Side {
    pub h: f64,
    pub un: f64,
    pub ut: f64,
    pub z: f64,
}

/// Numerical flux across one face, oriented along the face normal
/// (left → right is the positive direction).
///
/// `mom_l`/`mom_r` are the normal-momentum fluxes as seen by the left and
/// right cells respectively: they differ by the per-cell hydrostatic pressure
/// correction of Audusse et al. (2004), which is what makes the scheme
/// well-balanced. The mass flux is single-valued, so mass is conserved
/// exactly (up to f32 rounding of the state).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NumFlux {
    pub mass: f64,
    pub mom_l: f64,
    pub mom_r: f64,
    pub mom_t: f64,
}

/// HLL flux with Einfeldt (HLLE) wave-speed estimates over interface depths
/// reconstructed with the Chen & Noelle (2017) subcell scheme (spec v1.1 §3,
/// replacing the Audusse reconstruction of spec v1.0 §3.4).
///
/// Reconstruction: with surfaces `η_s = h_s + z_s`,
/// `z* = min(max(zL, zR), min(ηL, ηR))` and
/// `h*ₛ = max(0, min(η_s − z*, h_s))`. In the fully submerged regime
/// `z* = max(zL, zR)` and this reduces exactly to Audusse; when the bed step
/// exceeds the flow depth, `z*` is capped by the lower surface so the
/// per-cell source correction below carries the FULL bed-slope driving
/// force — the thin-layer-over-step truncation of plain Audusse (documented
/// in T5/EXP1/EXP2) is gone (test T13).
///
/// Wave speeds (spec §3.3): `sL = min(uL−√(g·hL*), û−ĉ)`,
/// `sR = max(uR+√(g·hR*), û+ĉ)` with Roe-averaged `û` and `ĉ = √(g·(hL*+hR*)/2)`,
/// specialised for dry sides (Toro 2001, ch. 10).
pub(crate) fn face_flux(l: Side, r: Side) -> NumFlux {
    let eta_l = l.h + l.z;
    let eta_r = r.h + r.z;
    let zf = l.z.max(r.z).min(eta_l.min(eta_r));
    let hl = (eta_l - zf).min(l.h).max(0.0);
    let hr = (eta_r - zf).min(r.h).max(0.0);

    let (mass, momn) = hll(hl, l.un, hr, r.un);

    // Chen & Noelle per-cell source corrections: cell s sees the face flux
    // plus g/2·(h_s + h*_s)·(z* − z_s) on the normal momentum. Submerged:
    // z* − z_s = h_s − h*_s, so this equals the Audusse correction
    // g/2·(h_s² − h*_s²) and the lake at rest balances exactly (T1/T12).
    // Over a step deeper than the flow: z* − z_s ≈ −Δz and the correction
    // becomes −g·h·Δz — the full driving force (T13).
    let p_l = 0.5 * G * (l.h + hl) * (zf - l.z);
    let p_r = 0.5 * G * (r.h + hr) * (zf - r.z);

    // Transverse momentum: passive upwinding by the sign of the mass flux.
    let mom_t = if mass >= 0.0 {
        mass * l.ut
    } else {
        mass * r.ut
    };

    NumFlux {
        mass,
        mom_l: momn + p_l,
        mom_r: momn + p_r,
        mom_t,
    }
}

/// 1D HLL solver on reconstructed depths: returns (mass flux, normal-momentum
/// flux) across the face.
fn hll(hl: f64, ul: f64, hr: f64, ur: f64) -> (f64, f64) {
    if hl <= 0.0 && hr <= 0.0 {
        return (0.0, 0.0);
    }
    let cl = (G * hl).sqrt();
    let cr = (G * hr).sqrt();

    let (sl, sr) = if hl <= 0.0 {
        // Dry left state: left edge of the right rarefaction.
        (ur - 2.0 * cr, ur + cr)
    } else if hr <= 0.0 {
        // Dry right state: right edge of the left rarefaction.
        (ul - cl, ul + 2.0 * cl)
    } else {
        let sql = hl.sqrt();
        let sqr = hr.sqrt();
        let u_hat = (sql * ul + sqr * ur) / (sql + sqr);
        let c_hat = (0.5 * G * (hl + hr)).sqrt();
        ((ul - cl).min(u_hat - c_hat), (ur + cr).max(u_hat + c_hat))
    };

    let fl = (hl * ul, hl * ul * ul + 0.5 * G * hl * hl);
    let fr = (hr * ur, hr * ur * ur + 0.5 * G * hr * hr);

    if sl >= 0.0 {
        fl
    } else if sr <= 0.0 {
        fr
    } else {
        let inv = 1.0 / (sr - sl);
        (
            (sr * fl.0 - sl * fr.0 + sl * sr * (hr - hl)) * inv,
            (sr * fl.1 - sl * fr.1 + sl * sr * (hr * ur - hl * ul)) * inv,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(h: f64, un: f64, ut: f64, z: f64) -> Side {
        Side { h, un, ut, z }
    }

    #[test]
    fn flat_equal_states_reduce_to_physical_flux() {
        let s = side(2.0, 1.5, 0.3, 0.0);
        let f = face_flux(s, s);
        let mass = 2.0 * 1.5;
        let mom = 2.0 * 1.5 * 1.5 + 0.5 * G * 4.0;
        assert!((f.mass - mass).abs() < 1e-12);
        assert!((f.mom_l - mom).abs() < 1e-12);
        assert!((f.mom_r - mom).abs() < 1e-12);
        assert!((f.mom_t - mass * 0.3).abs() < 1e-12);
    }

    #[test]
    fn wall_mirror_yields_zero_mass_flux() {
        // A reflective wall is modelled as a mirrored state with negated
        // normal velocity; the resulting mass flux must vanish exactly.
        let l = side(1.2, 0.8, 0.1, 5.0);
        let r = side(1.2, -0.8, 0.1, 5.0);
        let f = face_flux(l, r);
        assert_eq!(f.mass, 0.0);
    }

    #[test]
    fn lake_at_rest_is_exactly_balanced() {
        // Surface at 10 m over uneven bed: reconstructed depths are equal on
        // both sides, so the corrected momentum flux seen by each cell is
        // exactly g/2·h_i² — the same as its opposite face, giving zero net
        // update (well-balancedness, test T1 property at the face level).
        let l = side(10.0 - 3.0, 0.0, 0.0, 3.0);
        let r = side(10.0 - 4.5, 0.0, 0.0, 4.5);
        let f = face_flux(l, r);
        assert_eq!(f.mass, 0.0);
        assert!((f.mom_l - 0.5 * G * (10.0 - 3.0f64).powi(2)).abs() < 1e-9);
        assert!((f.mom_r - 0.5 * G * (10.0 - 4.5f64).powi(2)).abs() < 1e-9);
    }

    #[test]
    fn thin_layer_over_step_feels_full_driving_force() {
        // 0.05 m film on a 30° slope at 30 m cells: the bed step (~17.3 m)
        // dwarfs the depth. Plain Audusse truncates the driving force to
        // O(g·h²); Chen & Noelle must deliver ~ −g·h·Δz on the upslope
        // cell's momentum balance (spec v1.1 §3, T13 at face level).
        let dz = 30.0 * 30.0f64.to_radians().tan(); // ≈ 17.32 m
        let h = 0.05;
        let l = side(h, 0.0, 0.0, dz); // upslope cell
        let r = side(h, 0.0, 0.0, 0.0); // downslope cell
        let f = face_flux(l, r);
        // Net x-momentum force on the L cell from this face pair is
        // dominated by p_l ≈ ½g(2h)(z*−zL) with z* = ηR = h:
        let expected = 0.5 * G * (2.0 * h) * (h - dz);
        assert!(
            (f.mom_l - (0.5 * G * h * h + expected)).abs() < 0.05 * expected.abs(),
            "mom_l = {}, expected ≈ {}",
            f.mom_l,
            0.5 * G * h * h + expected
        );
        // And the R side must NOT receive a huge spurious force: its z is
        // the face level reference, correction ≈ ½g(h+h*R)(z*−zR) ≥ 0 small.
        assert!(
            f.mom_r.abs() < 1.0,
            "mom_r = {} unexpectedly large",
            f.mom_r
        );
    }

    #[test]
    fn dry_cell_never_loses_mass() {
        // Face between a dry left cell and a wet right cell moving away can
        // at most flood the dry cell (mass flux <= 0 into the left cell),
        // never drain it.
        for ur in [-3.0, -0.5, 0.0, 0.5, 3.0] {
            let f = face_flux(side(0.0, 0.0, 0.0, 0.0), side(1.0, ur, 0.0, 0.0));
            assert!(
                f.mass <= 1e-14,
                "ur={ur}: mass {0} out of a dry cell",
                f.mass
            );
        }
    }
}
