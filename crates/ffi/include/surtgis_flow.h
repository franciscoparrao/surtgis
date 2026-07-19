/* surtgis_flow.h — C ABI for the surtgis-flow debris-flow solver.
 *
 * Spec: surtgis-flow v1.0 §5. This header is the contract with the
 * GeodeoSim Unreal plugin. Hand-maintained (no cbindgen in v1); keep in
 * sync with crates/ffi/src/lib.rs and bump SF_ABI_VERSION on ANY signature
 * change.
 *
 * Thread-safety: an sf_sim may be moved between threads but MUST only be
 * used from one thread at a time (Rust: Send, not Sync). Drive it from a
 * single simulation thread.
 *
 * Buffers: row-major, row 0 = north, length w*h, NoData = NaN. The solver
 * COPIES every input buffer: the caller keeps ownership.
 *
 * Errors: no call ever aborts the process; Rust panics are caught and
 * reported as SF_ERR_INTERNAL. After SF_ERR_DIVERGED the state stays frozen
 * at the last valid substep (read it out for debugging).
 */
#ifndef SURTGIS_FLOW_H
#define SURTGIS_FLOW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sf_sim sf_sim; /* opaque */

typedef enum {
  SF_OK = 0,
  SF_ERR_INVALID_ARG   = 1,
  SF_ERR_GRID_MISMATCH = 2,
  SF_ERR_DIVERGED      = 3, /* NaN/Inf detected in the state */
  SF_ERR_INTERNAL      = 4
} sf_status;

/* dem/release: row-major, fila 0 = norte, longitud w*h. NoData = NaN.
 * El solver COPIA los buffers: el caller conserva ownership. */
sf_status sf_create(const float* dem, const float* release,
                    int32_t w, int32_t h, float cellsize,
                    float mu, float xi, sf_sim** out);

sf_status sf_set_params(sf_sim*, float mu, float xi, float v_stop);
sf_status sf_update_dem(sf_sim*, const float* dem);

/* Avanza dt segundos físicos. substeps_out puede ser NULL. */
sf_status sf_step(sf_sim*, float dt, uint32_t* substeps_out);

/* Copia el estado a buffers del caller (preasignados, w*h floats c/u).
 * Cualquiera puede ser NULL si no se necesita. u,v son velocidades (no
 * hu,hv), en m/s, 0 bajo el umbral seco. */
sf_status sf_read_state(const sf_sim*, float* h, float* u, float* v);
sf_status sf_read_arrival(const sf_sim*, float* t_arrival);

double    sf_time(const sf_sim*);        /* s físicos; NaN si sim == NULL  */
double    sf_total_mass(const sf_sim*);  /* m^3;      NaN si sim == NULL  */
void      sf_destroy(sf_sim*);           /* NULL es no-op                 */

/* Versión del ABI. Incrementar SF_ABI_VERSION ante CUALQUIER cambio de
 * firma. Comparar contra sf_abi_version() al cargar la librería. */
#define SF_ABI_VERSION 1
int32_t   sf_abi_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SURTGIS_FLOW_H */
