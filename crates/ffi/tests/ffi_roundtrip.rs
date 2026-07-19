//! Rust-side exercise of the C ABI: the same call sequence the C smoke test
//! (examples/smoke.c, run by the ffi-smoke CI job on Linux) performs, but
//! runnable on every platform `cargo test` covers.

use surtgis_ffi::{
    SF_ABI_VERSION, SF_ERR_INVALID_ARG, SF_OK, SfSim, sf_abi_version, sf_create, sf_destroy,
    sf_read_arrival, sf_read_state, sf_set_params, sf_step, sf_time, sf_total_mass, sf_update_dem,
};

const W: i32 = 64;
const H: i32 = 48;
const N: usize = (W as usize) * (H as usize);
const CELL: f32 = 5.0;

fn inputs() -> (Vec<f32>, Vec<f32>) {
    let slope = 15.0f32.to_radians().tan();
    let mut dem = vec![0.0f32; N];
    let mut release = vec![0.0f32; N];
    for r in 0..H as usize {
        for c in 0..W as usize {
            dem[r * W as usize + c] = (W as usize - 1 - c) as f32 * CELL * slope;
            if (16..32).contains(&r) && (8..20).contains(&c) {
                release[r * W as usize + c] = 2.0;
            }
        }
    }
    (dem, release)
}

#[test]
fn c_abi_roundtrip() {
    assert_eq!(sf_abi_version(), SF_ABI_VERSION);
    let (dem, release) = inputs();

    let mut sim: *mut SfSim = std::ptr::null_mut();
    let st = unsafe {
        sf_create(
            dem.as_ptr(),
            release.as_ptr(),
            W,
            H,
            CELL,
            0.15,
            200.0,
            &raw mut sim,
        )
    };
    assert_eq!(st, SF_OK);
    assert!(!sim.is_null());

    let mass0 = unsafe { sf_total_mass(sim) };
    assert!(mass0 > 0.0);

    assert_eq!(unsafe { sf_set_params(sim, 0.12, 300.0, 0.01) }, SF_OK);
    assert_eq!(
        unsafe { sf_set_params(sim, -1.0, 300.0, 0.01) },
        SF_ERR_INVALID_ARG,
        "mu < 0 must be rejected"
    );

    let mut substeps: u32 = 0;
    for _ in 0..5 {
        assert_eq!(unsafe { sf_step(sim, 1.0, &raw mut substeps) }, SF_OK);
        assert!(substeps > 0);
    }
    assert!((unsafe { sf_time(sim) } - 5.0).abs() < 1e-6);

    let mut h = vec![0.0f32; N];
    let mut u = vec![0.0f32; N];
    let mut v = vec![0.0f32; N];
    let mut arrival = vec![0.0f32; N];
    assert_eq!(
        unsafe { sf_read_state(sim, h.as_mut_ptr(), u.as_mut_ptr(), v.as_mut_ptr()) },
        SF_OK
    );
    assert_eq!(unsafe { sf_read_arrival(sim, arrival.as_mut_ptr()) }, SF_OK);

    let mass = unsafe { sf_total_mass(sim) };
    assert!((mass - mass0).abs() / mass0 < 1e-5, "mass drifted: {mass}");
    assert!(h.iter().all(|x| x.is_finite() && *x >= 0.0));
    assert!(u.iter().chain(&v).all(|x| x.is_finite()));
    let wet = h.iter().filter(|&&x| x > 1e-3).count();
    let moving = u.iter().filter(|&&x| x > 0.1).count();
    assert!(
        wet > 100 && moving > 10,
        "flow did not propagate (wet={wet})"
    );

    assert_eq!(unsafe { sf_update_dem(sim, dem.as_ptr()) }, SF_OK);
    assert_eq!(
        unsafe { sf_update_dem(sim, std::ptr::null()) },
        SF_ERR_INVALID_ARG
    );
    assert_eq!(
        unsafe { sf_step(std::ptr::null_mut(), 1.0, std::ptr::null_mut()) },
        SF_ERR_INVALID_ARG
    );
    // dt = NaN must be rejected as a caller error, not a crash.
    assert_eq!(
        unsafe { sf_step(sim, f32::NAN, std::ptr::null_mut()) },
        SF_ERR_INVALID_ARG
    );

    unsafe { sf_destroy(sim) };
    unsafe { sf_destroy(std::ptr::null_mut()) }; // no-op
}

#[test]
fn create_rejects_bad_args() {
    let (dem, release) = inputs();
    let mut sim: *mut SfSim = std::ptr::null_mut();
    for (d, r, w, h, cs) in [
        (std::ptr::null(), release.as_ptr(), W, H, CELL),
        (dem.as_ptr(), std::ptr::null(), W, H, CELL),
        (dem.as_ptr(), release.as_ptr(), 0, H, CELL),
        (dem.as_ptr(), release.as_ptr(), W, -1, CELL),
        (dem.as_ptr(), release.as_ptr(), W, H, 0.0),
        (dem.as_ptr(), release.as_ptr(), W, H, f32::NAN),
    ] {
        let st = unsafe { sf_create(d, r, w, h, cs, 0.15, 200.0, &raw mut sim) };
        assert_eq!(st, SF_ERR_INVALID_ARG);
        assert!(sim.is_null(), "out must be nulled on failure");
    }
    assert_eq!(
        unsafe {
            sf_create(
                dem.as_ptr(),
                release.as_ptr(),
                W,
                H,
                CELL,
                0.15,
                200.0,
                std::ptr::null_mut(),
            )
        },
        SF_ERR_INVALID_ARG
    );
}
