/* Smoke test for the surtgis-flow C ABI (spec §9, M5 gate): build a small
 * ramp DEM, run a short simulation through every entry point and verify the
 * results are sane. Compiled and executed in CI (job ffi-smoke).
 *
 * Build (from the repo root, after `cargo build -p surtgis-ffi --release`):
 *   cc -Icrates/ffi/include crates/ffi/examples/smoke.c \
 *      -Ltarget/release -lsurtgis_ffi -lm -o sf_smoke
 *   LD_LIBRARY_PATH=target/release ./sf_smoke
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "surtgis_flow.h"

#define W 64
#define H 48
#define CELL 5.0f

static int fail(const char* what, int code) {
  fprintf(stderr, "FAIL: %s (status %d)\n", what, code);
  return 1;
}

int main(void) {
  if (sf_abi_version() != SF_ABI_VERSION)
    return fail("ABI version mismatch", sf_abi_version());

  static float dem[W * H], release[W * H], h[W * H], u[W * H], v[W * H],
      arrival[W * H];
  /* 15 deg ramp dipping east, 2 m release block near the west edge. */
  const float slope = 0.2679f; /* tan(15 deg) */
  for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++) {
      dem[r * W + c] = (W - 1 - c) * CELL * slope;
      release[r * W + c] = (r >= 16 && r < 32 && c >= 8 && c < 20) ? 2.0f : 0.0f;
    }

  sf_sim* sim = NULL;
  sf_status st = sf_create(dem, release, W, H, CELL, 0.15f, 200.0f, &sim);
  if (st != SF_OK || sim == NULL) return fail("sf_create", st);

  const double mass0 = sf_total_mass(sim);
  if (!(mass0 > 0.0)) return fail("initial mass", 0);

  if ((st = sf_set_params(sim, 0.12f, 300.0f, 0.01f)) != SF_OK)
    return fail("sf_set_params", st);
  /* Out-of-range params must be rejected without corrupting the sim. */
  if (sf_set_params(sim, -1.0f, 300.0f, 0.01f) != SF_ERR_INVALID_ARG)
    return fail("sf_set_params should reject mu < 0", 0);

  uint32_t substeps = 0;
  for (int i = 0; i < 5; i++) {
    if ((st = sf_step(sim, 1.0f, &substeps)) != SF_OK) return fail("sf_step", st);
    if (substeps == 0) return fail("no substeps executed", 0);
  }
  if (fabs(sf_time(sim) - 5.0) > 1e-6) return fail("sf_time", 0);

  if ((st = sf_read_state(sim, h, u, v)) != SF_OK) return fail("sf_read_state", st);
  if ((st = sf_read_arrival(sim, arrival)) != SF_OK) return fail("sf_read_arrival", st);

  /* Mass conserved (nothing reaches the borders in 5 s), state finite,
   * flow actually moving eastward. */
  const double mass = sf_total_mass(sim);
  if (fabs(mass - mass0) / mass0 > 1e-5) return fail("mass drift", 0);
  int wet = 0, moving = 0, arrived = 0;
  for (int i = 0; i < W * H; i++) {
    if (!isfinite(h[i]) || !isfinite(u[i]) || !isfinite(v[i]))
      return fail("non-finite state", i);
    if (h[i] < 0.0f) return fail("negative depth", i);
    if (h[i] > 1e-3f) wet++;
    if (u[i] > 0.1f) moving++;
    if (isfinite(arrival[i])) arrived++;
  }
  if (wet < 100 || moving < 10 || arrived <= wet / 2)
    return fail("flow did not propagate", wet);

  /* Entrainment (ABI v2): a fresh sim with 1 m erodible everywhere must
   * erode, keep the budget, and reject activation after stepping. */
  {
    static float emax[W * H], er[W * H];
    for (int i = 0; i < W * H; i++) emax[i] = 1.0f;
    sf_sim* s2 = NULL;
    if ((st = sf_create(dem, release, W, H, CELL, 0.15f, 200.0f, &s2)) != SF_OK)
      return fail("sf_create (entrainment)", st);
    if ((st = sf_set_erodible(s2, emax, 1e-2f, 0.05f, 0.1f)) != SF_OK)
      return fail("sf_set_erodible", st);
    for (int i = 0; i < 5; i++)
      if ((st = sf_step(s2, 1.0f, NULL)) != SF_OK) return fail("sf_step (ent)", st);
    if (!(sf_total_eroded(s2) > 0.0)) return fail("no erosion", 0);
    if ((st = sf_read_erosion(s2, er)) != SF_OK) return fail("sf_read_erosion", st);
    double er_sum = 0.0;
    for (int i = 0; i < W * H; i++) {
      if (!isfinite(er[i]) || er[i] < 0.0f || er[i] > 1.0f)
        return fail("erosion out of [0, e_max]", i);
      er_sum += er[i];
    }
    if (!(er_sum > 0.0)) return fail("erosion buffer empty", 0);
    /* Conservation: flow grew by exactly what the bed lost (1e-5 rel). */
    double m2 = sf_total_mass(s2);
    if (fabs(m2 - mass0 - sf_total_eroded(s2)) / mass0 > 1e-5)
      return fail("entrainment mass balance", 0);
    /* Activation after stepping must be rejected. */
    if (sf_set_erodible(s2, emax, 1e-2f, 0.05f, 0.1f) != SF_ERR_INVALID_ARG)
      return fail("sf_set_erodible after step should fail", 0);
    sf_destroy(s2);
  }

  /* update_dem round-trip and null-arg rejection. */
  if ((st = sf_update_dem(sim, dem)) != SF_OK) return fail("sf_update_dem", st);
  if (sf_update_dem(sim, NULL) != SF_ERR_INVALID_ARG)
    return fail("sf_update_dem(NULL) should fail", 0);
  if (sf_step(NULL, 1.0f, NULL) != SF_ERR_INVALID_ARG)
    return fail("sf_step(NULL) should fail", 0);

  sf_destroy(sim);
  sf_destroy(NULL); /* must be a no-op */

  printf("surtgis-ffi smoke: OK (abi %d, %d wet cells, mass %.1f m3, entrainment OK)\n",
         sf_abi_version(), wet, mass);
  return 0;
}
