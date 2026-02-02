# SurtGIS — Implementation Roadmap

> **Generated**: 2026-01-31
> **Sources**: SOTA review (1038 WOS refs + Florinsky 21 chapters + competitive analysis)
> **Format**: GitHub-compatible checklists — checkboxes are interactive on GitHub
> **Status**: Living document — update checkboxes as items are completed

---

## Progress Dashboard

### By Tier

| Tier | Total | Done | Remaining | % |
|------|-------|------|-----------|---|
| P0 — Critical Corrections | 4 | 4 | 0 | 100% |
| P0 — Critical Missing | 6 | 6 | 0 | 100% |
| P1 — High Priority | 21 | 21 | 0 | 100% |
| P2 — Medium Priority | 14 | 14 | 0 | 100% |
| P3 — Research/Future | 10 | 10 | 0 | 100% |
| Infrastructure | 6 | 6 | 0 | 100% |
| **Total** | **61** | **61** | **0** | **100%** |

### By Theme

| Theme | Items | Done | Key Blockers |
|-------|-------|------|--------------|
| Terrain / Curvatures | 12 | 16 | Complete |
| Hydrology | 10 | 10 | Complete |
| Imagery / Spectral | 3 | 2 | — |
| Interpolation | 7 | 8 | — |
| Solar Radiation | 4 | 8 | — |
| Viewshed | 3 | 3 | — |
| Smoothing / Filtering | 5 | 5 | Complete |
| Infrastructure | 6 | 6 | Complete |

---

## How to Use This Document

Each item follows this format:

```
- [ ] **ID** `module::function` — Description. Effort | Ref | Advantage
```

**Legend**:

| Symbol | Meaning |
|--------|---------|
| **Effort: Low** | < 200 lines, straightforward algebra or thin wrapper |
| **Effort: Med** | 200–800 lines, algorithm with some complexity |
| **Effort: High** | > 800 lines, complex data structures or multiple sub-algorithms |
| **UNIQUE** | No open-source competitor implements this |
| **WTE-FREE** | WhiteboxTools charges for this (WTE); SurtGIS would be the free alternative |
| **FIX** | Corrects a known error in the current implementation |

Dependencies are noted as: `Requires: P0-H1` (must complete that item first).

---

## Quick Wins — Start Here

These 9 items have **low effort + high impact**. They are a subset of items from other tiers (not double-counted in totals).

| # | ID | Description | Effort | Lines | Why |
|---|-----|-------------|--------|-------|-----|
| QW-1 | P1-T3 | Northness/Eastness (cos A, sin A) | Low | ~20 | Needed for regression; trivial |
| QW-2 | P0-T1 | DEV — Deviation from Mean Elevation | Low | ~100 | 509 citations; superior to TPI |
| QW-3 | P2-T1 | Log transform for curvature visualization | Low | ~20 | UNIQUE; Florinsky Eq. 8.1 |
| QW-4 | P1-T6 | Multidirectional hillshade | Low | ~60 | Better visualization; only WBT has it |
| QW-5 | P1-T2 | 8 derived curvatures (pure algebra) | Low | ~120 | WTE-FREE; completes Florinsky 12 |
| QW-6 | P1-T4 | Shape index + curvedness | Low | ~40 | WTE-FREE; continuous landform classification |
| QW-7 | P0-H5 | HAND (Height Above Nearest Drainage) | Low | ~80 | Key for flood mapping |
| QW-8 | P2-T3 | Accumulation zones (kh<0 ∧ kv<0) | Low | ~30 | Florinsky Ch. 15; fault intersections |
| QW-9 | P1-I2 | 4 additional spectral indices | Low | ~80 | NDRE, GNDVI, NGRDI, RECI |

---

## P0 — Critical Corrections (FIX)

These fix known errors in the current codebase. **Do these first.**

- [x] **P0-C1** `terrain::multiscale_curvatures` — Fix polynomial model: current code uses **quadratic** (2nd order) least-squares on the 5×5 window, but Florinsky's method requires a **cubic** (3rd order) polynomial to compute 3rd-order derivatives (g, h, k, m). Replace with closed-form formulas Eqs. 4.14–4.22 from Florinsky (2025, Ch. 4). Effort: Med | Ref: Florinsky 2009, 2025 §4.2 | FIX
  - Current: Generic LS fit with quadratic terms
  - Required: 9 closed-form derivative estimates (r, t, s, p, q + g, h, k, m) from 25-cell window
  - Impact: Enables horizontal deflection (Dkh), correct curvature RMSE (~6× better than Evans-Young for 2nd-order derivatives per Table 5.1)

- [x] **P0-C2** `terrain::curvature` — ~~Add missing denominator in Z&T curvatures.~~ ✅ Done. Added `CurvatureFormula` enum (`Full`/`Simplified`). Full formulas with √(1+p²+q²) denominators are now default. Effort: Low | Ref: Florinsky 2025 §2.4, §7.2.4 | FIX

- [x] **P0-C3** `terrain::smoothing` — Fix FPDEMS: current implementation is a single-stage bilateral filter. The real FPDEMS (Lindsay et al. 2019 / Florinsky Eqs. 6.11–6.12) is **two-stage**: (1) smooth normals with cos²(angle) weighting + hard threshold Θ_t, (2) adjust elevations from tangent planes of neighbors with smoothed normals. Effort: Med | Ref: Lindsay 2019, Florinsky 2025 §6.3 | FIX

- [x] **P0-C4** `terrain::curvature` — ~~Set Evans-Young as default 3×3 method.~~ ✅ Done. Added `DerivativeMethod` enum (`EvansYoung`/`ZevenbergenThorne`). Evans-Young is now default. Effort: Low | Ref: Florinsky 2025 §4.1, Table 5.1 | FIX

---

## P0 — Critical Missing Algorithms

These close the most visible gaps versus competitors and enable downstream workflows.

- [x] **P0-H1** `hydrology::flow_accumulation_mfd` — ~~Implement FD8/Quinn multiple flow direction.~~ ✅ Done. `flow_accumulation_mfd()` with configurable exponent (default 1.1). Uses topological sort (highest→lowest). Contour-weighted Quinn formula. 5 tests including D8 comparison. Effort: Med | Ref: Quinn et al. 1991, Quinn et al. 1995

- [x] **P0-H2** `hydrology::flow_direction_dinf` — ~~Implement D-infinity (Tarboton 1997).~~ ✅ Done. `flow_direction_dinf()` for angles + `flow_dinf()` for direction+accumulation. Triangular facet decomposition, continuous angles. 5 tests. Effort: Med | Ref: Tarboton 1997

- [x] **P0-H3** `hydrology::breach` — ~~Implement breach depressions (Lindsay 2016).~~ ✅ Done. `breach_depressions()` with Dijkstra least-cost path + Priority-Flood for remaining pits. max_depth, max_length params. 4 tests. Effort: Med | Ref: Lindsay 2016

- [x] **P0-H4** `hydrology::priority_flood` — ~~Implement Priority-Flood (Barnes 2014).~~ ✅ Done. `priority_flood()` with min-heap, O(n log n). Epsilon parameter for gradient enforcement. `priority_flood_flat()` convenience function. 7 tests. Effort: Med | Ref: Barnes 2014 (181 cites) | UNIQUE

- [x] **P0-T1** `terrain::dev` — ~~Implement DEV (Deviation from Mean Elevation).~~ ✅ Done. `dev()` with configurable radius and stddev normalization. Effort: Low | Ref: De Reu 2013 (509 cites), Newman 2018 (43 cites)

- [x] **P0-H5** `hydrology::hand` — ~~Implement HAND (Height Above Nearest Drainage).~~ ✅ Done. `hand()` traces D8 downstream to stream cells, computes elevation difference. Caches path results for performance. 4 tests. Effort: Low | Ref: Nobre 2011

---

## P1 — High Priority

### Terrain / Curvatures

- [x] **P1-T1** `terrain::derivatives` — Create shared derivatives_3x3 module with Evans-Young formulas as reusable building block. All terrain functions (slope, aspect, curvature, hillshade) should call this instead of reimplementing derivative computation. Effort: Med | Ref: Evans 1979, Florinsky 2025 §4.1
  - Requires: P0-C1 (for 5×5 variant), P0-C4

- [x] **P1-T2** `terrain::curvature_advanced` — ~~Implement 12 complete Florinsky curvatures.~~ ✅ Done. `curvature_advanced.rs` with 14 curvature types (12 Florinsky + rotor + Laplacian). `all_curvatures()` single-pass, `advanced_curvatures()` for individual types. WTE-FREE. Effort: Low | Ref: Florinsky 2025 §2.4, Eqs. in florinsky_ch2_equations.tex | WTE-FREE

- [x] **P1-T3** `terrain::northness_eastness` — ~~Implement Northness (cos A), Eastness (sin A).~~ ✅ Done. `northness()`, `eastness()`, `northness_eastness()` (single-pass). Effort: Low | Ref: Florinsky 2025 §2.2, Amatulli 2018

- [x] **P1-T4** `terrain::shape_index` — ~~Implement shape index and curvedness.~~ ✅ Done. `shape_index()` and `curvedness()` from principal curvatures. Effort: Low | Ref: Florinsky 2025 §13.3 | WTE-FREE

- [x] **P1-T5** `terrain::chebyshev_spectral` — Implement Chebyshev spectral analytical method. Unified framework for interpolation + filtering + derivative computation. 4 stages: Chebyshev expansion, Gauss quadrature coefficients, Fejér attenuation, spectral derivatives. Continuous scale parameter l. Matrix implementation (Eqs. 7.18–7.22), only matrix multiplication (no linear system solving). Uses `ndarray` crate. Effort: High | Ref: Florinsky & Pankratov 2016, Florinsky 2025 Ch. 7 | UNIQUE
  - **No open-source library implements this.** SurtGIS would be the first.

- [x] **P1-T6** `terrain::multidirectional_hillshade` — ~~Add multidirectional hillshade.~~ ✅ Done. 6-azimuth weighted blend with aspect-dependent weighting. Effort: Low | Ref: Amatulli 2018, Mark 1992

- [x] **P1-T7** `terrain::gaussian_scale_space` — Implement Gaussian scale-space (fGSS) framework for optimal multiscale analysis. Newman (2022) identifies fGSS as the optimal scaling method. Effort: Med | Ref: Newman 2022 (9 cites), Newman 2023 (5 cites)

### Smoothing / Filtering

- [x] **P1-S1** `terrain::smoothing` — ~~Add Gaussian smoothing.~~ ✅ Done. `gaussian_smoothing()` with configurable radius and sigma. Precomputed kernel, parallel. 3 tests. Effort: Low | Ref: Florinsky 2025 §6.2

- [x] **P1-S2** `terrain::smoothing` — ~~Add iterative weighted mean.~~ ✅ Done. `iterative_mean_smoothing()` with configurable iterations and weight exponent m=0,1,2. 4 tests. Effort: Low | Ref: Florinsky 2025 §6.2

### Interpolation

- [x] **P1-I1** `interpolation::variogram` — Implement empirical variogram computation and theoretical model fitting (spherical, exponential, Gaussian, Matérn). Prerequisite for all kriging variants. Effort: Med | Ref: Florinsky 2025 §3.3, 149 papers in WOS bibliography

- [x] **P1-I2** `imagery::indices` — ~~Add NDRE, GNDVI, NGRDI, RECI spectral indices.~~ ✅ Done. All 4 indices added with tests. Effort: Low | Ref: Remote sensing analysis (31 high-relevance papers)

- [x] **P1-I3** `interpolation::ordinary_kriging` — Implement Ordinary Kriging. Gold standard geostatistical method (79% of 188 papers). Effort: Med | Ref: Workneh 2024 (71 cites), Rata 2020 (34 cites)
  - Requires: P1-I1 (variogram)

- [x] **P1-I4** `interpolation::regression_kriging` — Implement Regression Kriging. Consistently outperforms IDW by 15–30% RMSE in mountainous terrain. Hybrid: trend surface + OK on residuals. Effort: Med | Ref: Zhu 2010 (35 cites), Li 2010 (37 cites)
  - Requires: P1-I3 (OK)

- [x] **P1-I5** `interpolation::universal_kriging` — Implement Universal Kriging. 5–15% improvement over OK when strong spatial trends exist. Effort: Med | Ref: Kravchenko & Bullock 1999 (27 cites)
  - Requires: P1-I1 (variogram)

- [x] **P1-I6** `interpolation::tps` — Implement Thin Plate Spline. Confirmed critical by both WOS bibliography and Florinsky Ch. 3. Minimizes bending energy seminorm. Standard in ArcGIS, QGIS, GRASS. Effort: Med | Ref: Florinsky 2025 §3.2, Eq. 3.2

### Solar Radiation

- [x] **P1-SO1** `terrain::horizon_angles` — Implement shared horizon angle computation module. Bresenham-line scan in 36 azimuthal directions. Shared between solar radiation (shadow casting), SVF, and viewshed. Single most impactful improvement for solar module (eliminates 10–40% overestimation in mountains). Effort: Med | Ref: Corripio 2003 (161 cites), Marsh 2012 (49 cites)

- [x] **P1-SO2** `terrain::solar_radiation` — Add shadow casting using horizon angles from P1-SO1. For each timestep, check if sun altitude < horizon angle at sun azimuth → cell is in shadow. Effort: Low | Ref: Corripio 2003, Steger 2022
  - Requires: P1-SO1

- [x] **P1-SO3** `terrain::solar_radiation` — ~~Implement Klucher anisotropic diffuse model.~~ ✅ Done. `DiffuseModel::Klucher` enum variant. Circumsolar + horizon brightening correction. 1 test. Effort: Low | Ref: Klucher 1979, Long 2010

### Imagery

- [x] **P1-IM1** `imagery::index_builder` — ~~Implement generic n-band spectral index builder.~~ ✅ Done. Recursive descent parser for arithmetic expressions with band references. Supports +, -, *, /, parentheses, numeric constants. 7 tests including NDVI, EVI, 3-band formulas. Effort: Med | Ref: Wang et al. 2019 (19 cites) | UNIQUE
  - **No competitor** (WBT/SAGA/GRASS) offers this as a generic function

### Viewshed

- [x] **P1-V1** `terrain::viewshed` — Implement XDraw approximate viewshed algorithm. 2–5× speedup over current Bresenham ray tracing with good accuracy. Most widely studied viewshed algorithm (12 papers). Effort: Med | Ref: Cauchi-Saunders 2015 (20 cites), Fisher 1993 (118 cites)

### Hydrology

- [x] **P1-H1** `hydrology::stream_network` — ~~Extract stream network from flow accumulation using threshold.~~ ✅ Done. `stream_network()` with configurable threshold. Binary output (1=stream, 0=non-stream). 3 tests. Effort: Low | Ref: Multiple

---

## P2 — Medium Priority

### Terrain

- [x] **P2-T1** `terrain::log_transform` — ~~Implement log transform for visualization.~~ ✅ Done. `log_transform()` with sign-preserving ln(1+|x|). Effort: Low | Ref: Florinsky 2025 §8.1, Eq. 8.1 | UNIQUE

- [x] **P2-T2** `terrain::vrm` — ~~Implement Vector Ruggedness Measure (Sappington 2007).~~ ✅ Done. `vrm()` with configurable radius, 5 tests. Effort: Low | Ref: Sappington 2007

- [x] **P2-T3** `terrain::accumulation_zones` — ~~Implement accumulation zone classification.~~ ✅ Done. 4-class zone classification from plan/profile curvature signs. Effort: Low | Ref: Florinsky 2025 §15.2

- [x] **P2-T4** `terrain::lineament_pipeline` — ~~Implement lineament detection pipeline.~~ ✅ Done. `lineament_detection()` with zero-crossing binarization, Zhang-Suen thinning, segment filtering, strike-slip/dip-slip/oblique classification. 6 tests. Effort: High | Ref: Florinsky 2025 Ch. 14

- [x] **P2-T5** `terrain::rea_analysis` — ~~Implement REA diagnostic function.~~ ✅ Done. `rea_analysis()` with slope/curvature variables, multi-scale correlation, REA identification. 6 tests. Effort: Med | Ref: Florinsky 2025 Ch. 10 | UNIQUE

### Solar Radiation

- [x] **P2-SO1** `terrain::solar_radiation` — ~~Add reflected radiation component.~~ ✅ Done. `albedo` param in SolarParams, `reflected` field in result. Formula: albedo × GHI × (1−cos β)/2. 4 new tests. Effort: Low | Ref: Long 2010 (71 cites)

- [x] **P2-SO2** `terrain::solar_radiation` — ~~Implement Linke turbidity factor.~~ ✅ Done. `linke_turbidity` param in SolarParams with Kasten (1996) δ_R(m) formula. 2 new tests. Effort: Low | Ref: Hofierka & Šúri 2002, Ruiz-Arias 2009 (93 cites)

- [x] **P2-SO3** `terrain::solar_radiation` — ~~Add monthly and annual integration.~~ ✅ Done. `solar_radiation_annual()` with Klein (1977) representative days, MonthlySolarResult. 3 tests. Effort: Med | Ref: Potic 2016, Effat 2022

- [x] **P2-SO4** `terrain::horizon` — ~~Implement HORAYZON-style efficient horizon computation.~~ ✅ Done. `horizon_angles_fast()` and `horizon_angle_map_fast()` with LOD pyramid (max-elevation aggregation), adaptive stepping. 8 new tests. Effort: High | Ref: Steger 2022 (15 cites)

### Viewshed

- [x] **P2-V1** `terrain::viewshed` — ~~Add probabilistic/fuzzy viewshed.~~ ✅ Done. `viewshed_probabilistic()` Monte Carlo with configurable RMSE, seed, N realizations. 4 tests. Effort: Med | Ref: Fisher 1993 (118 cites)

- [x] **P2-V2** `terrain::viewshed` — ~~Add observer optimization (MCLP/greedy).~~ ✅ Done. `observer_optimization()` greedy MCLP with precomputed viewsheds. 4 tests. Effort: High | Ref: Bao 2015 (92 cites), Wang & Dou 2020 (22 cites)

### Interpolation

- [x] **P2-I1** `interpolation::natural_neighbor` — ~~Implement Natural Neighbor (Sibson).~~ ✅ Done. `natural_neighbor()` with discrete Sibson area-stealing. C1 continuous, exact at data points. 7 tests. Effort: Med | Ref: Sibson 1981

- [x] **P2-I2** `interpolation::idw` — ~~Enhance IDW with adaptive power and anisotropy.~~ ✅ Done. `AdaptivePower` (density-dependent exponent) + `Anisotropy` (elliptical search). 3 new tests. Effort: Low | Ref: Chen 2015 (37 cites)

### Smoothing

- [x] **P2-SM1** `terrain::smoothing` — ~~Add FFT low-pass filter.~~ ✅ Done. `fft_low_pass()` with built-in Cooley-Tukey radix-2 FFT, circular frequency cutoff, mirror padding. 4 tests. Effort: Med | Ref: Florinsky 2025 §6.1

---

## P3 — Research / Future

These are published algorithms that **no competitor implements** or cutting-edge research.

- [x] **P3-T1** `terrain::msv` — Implement Multi-Scale Valleyness (Wang 2009). Quadratic surface fitting to characterize diffuse valley features. Detects areas missed by MRVBF, D8, D-inf, and flow accumulation. Effort: Med | Ref: Wang & Laffan 2009 (9 cites) | UNIQUE

- [x] **P3-T2** `terrain::spheroidal_grid` — Add spheroidal grid support for global DEMs. Implement Florinsky's (2017b) formulas: geodetic inverse problem (Eqs. 4.32–4.42), variable cell area (Eq. 4.43), spheroidal partial derivatives (Eqs. 4.27–4.31). Required for correct processing of SRTM, Copernicus GLO-30, ETOPO, etc. Effort: High | Ref: Florinsky 2025 §4.3, Ch. 16

- [x] **P3-H1** `hydrology::flow_direction_tfga` — Implement TFGA (Facet-to-Facet MFD). Divides central pixel into 8 sub-facets with strict mathematical flow proportion derivation. ~1 order of magnitude more precise than existing MFD. Published 2024, nobody implements it. Effort: High | Ref: Li Z. et al. 2024 | UNIQUE

- [x] **P3-H2** `hydrology::nested_depressions` — Implement nested depression delineation using level-set method (graph theory). ~150× faster than contour tree vector methods. Relevant for fill-merge-spill simulation. Effort: High | Ref: Wu 2019 (55 cites) | UNIQUE

- [x] **P3-H3** `hydrology::flow_direction_mfd_adaptive` — Implement adaptive MFD (Qin 2011). Uses maximum downslope gradient as beta instead of local slope. Lower TWI errors at 1–30m resolution. Effort: Med | Ref: Qin 2011 (128 cites)

- [x] **P3-H4** `hydrology::watershed` — Implement O(N) parallel watershed delineation (Zhou 2026). Flow path traversal from flow direction grids. Fastest published algorithm. Effort: Med | Ref: Zhou et al. 2026 | UNIQUE

- [x] **P3-V1** `terrain::viewshed` — Implement PDERL viewshed. R3-exact accuracy at speeds comparable to XDraw (~2× XDraw). Novel PDE coordinate system. Effort: High | Ref: Wu et al. 2021 (13 cites)

- [x] **P3-SO1** `terrain::solar_radiation` — Implement Perez anisotropic diffuse model. Most accurate diffuse model: 8 sky condition bins with circumsolar and horizon brightness. Effort: Med | Ref: Perez et al. 2002

- [x] **P3-SO2** `terrain::solar_radiation` — Implement Corripio vectorial algebra approach. Replace scalar trigonometry with rotation matrices on 3D vectors. Eliminates trigonometric edge-cases. More elegant and extensible. Effort: Med | Ref: Corripio 2003 (161 cites)

- [x] **P3-SM1** `terrain::ssa_2d` — Implement 2D-SSA (Singular Spectrum Analysis) for DEM denoising. Model-free denoising + multiscale decomposition. Florinsky's preferred method for research-quality results. Effort: High | Ref: Golyandina, Usevich & Florinsky 2007 | UNIQUE

---

## Infrastructure

- [x] **INF-1** `crates/wasm` — Expand WASM bindings to cover all new algorithms. Each algorithm auto-runs in browser via `maybe_rayon`. No competitor offers terrain/hydrology in WASM. Effort: Med (ongoing)

- [x] **INF-2** `crates/python` — Add Python bindings via PyO3. Expose core algorithms to Python/Jupyter ecosystem. Effort: High

- [x] **INF-3** `interpolation::kdtree` — Implement k-d tree spatial index. Current O(n·m) neighbor search is impractical for point clouds with millions of points. Prerequisite for performant kriging and IDW. Effort: Med | Ref: Florinsky 2025 §3.2

- [x] **INF-4** `terrain::derivatives` — Create shared derivatives module (Evans-Young 3×3 + Florinsky 5×5). All terrain functions call this instead of reimplementing derivative computation. Eliminates code duplication across slope, aspect, curvature, hillshade. Effort: Med
  - Same as P1-T1 — counted once

- [x] **INF-5** CI/CD — Set up GitHub Actions: build (native + WASM), test, clippy, benchmarks. Effort: Med

- [x] **INF-6** `terrain::uncertainty` — Implement uncertainty maps using Florinsky Ch. 5 propagation formulas (Eqs. 5.2–5.12). Given DEM RMSE (mz), compute per-pixel RMSE for slope, curvatures, TWI, etc. Computationally free — uses same partial derivatives already calculated. Effort: Med | Ref: Florinsky 2025 Ch. 5 | UNIQUE
  - **No open-source library** provides per-pixel uncertainty maps alongside morphometric outputs

---

## Dependency Graph

### Hydrology Chain

```
P0-H4 (Priority-Flood)  ──→  P0-H1 (FD8/MFD)  ──→  P0-H5 (HAND)
       │                           │                       │
       │                           ├──→  TWI (improved)    │
       │                           │                       │
       └──→  P0-H3 (Breach)       └──→  P1-H1 (Stream network)
                                                │
                                                └──→  P0-H5 (HAND, refined)
```

Note: P0-H1 (FD8) and P0-H2 (D-infinity) are independent of P0-H4 (Priority-Flood) — they can use the existing fill_sinks as preprocessing. Priority-Flood is a *better* fill, not a prerequisite.

### Curvature Chain

```
P0-C1 (Fix 5×5 formulas)  ──→  P1-T1 (Shared derivatives)  ──→  P1-T2 (12 curvatures)
       │                                                              │
P0-C4 (Evans-Young default) ──→  P1-T1                               ├──→  P1-T4 (Shape index)
       │                                                              │
P0-C2 (Fix Z&T denominator)                                          └──→  QW-5/QW-6 (quick wins)
```

### Interpolation Chain

```
INF-3 (k-d tree)  ──→  P1-I1 (Variogram)  ──→  P1-I3 (Ordinary Kriging)
                                                       │
                                                       ├──→  P1-I4 (Regression Kriging)
                                                       │
                                                       └──→  P1-I5 (Universal Kriging)
```

### Solar / Viewshed Shared Infrastructure

```
P1-SO1 (Horizon angles)  ──→  P1-SO2 (Shadow casting)
       │                            │
       ├──→  SVF (already exists)   └──→  P2-SO3 (Monthly/annual)
       │
       └──→  P1-V1 (XDraw) — independent but shares concepts
```

### Smoothing Chain

```
P1-S1 (Gaussian)          ──→  P2-SM1 (FFT low-pass)
P1-S2 (Iterative mean)    ──→  P3-SM1 (2D-SSA)
P0-C3 (Fix FPDEMS 2-stage)
```

---

## Appendix: Validation Criteria

### P0 Corrections

| Item | Acceptance Test |
|------|-----------------|
| P0-C1 | Compute all 9 partial derivatives (r,t,s,p,q,g,h,k,m) on a synthetic cone (analytical solution known). RMSE must match Florinsky Table 5.1 bounds: mr,mt = √(2/35)·mz/w² |
| P0-C2 | On a 45° slope synthetic DEM, curvature with florinsky method must differ from simplified by ~30% (confirming correction) |
| P0-C3 | Compare smoothed output on a synthetic ridge DEM: 2-stage FPDEMS must preserve ridge crest sharper than single-stage bilateral |
| P0-C4 | Evans-Young produces identical slope/aspect to Z&T on flat terrain (<10°); on 30°+ terrain, differences emerge in curvatures |

### P0 Missing Algorithms

| Item | Acceptance Test |
|------|-----------------|
| P0-H1 (FD8) | TWI from FD8 correlates with measured soil moisture at r > 0.5 on standard test DEM (Kopecky benchmark); TWI from D8 at r < 0.3 |
| P0-H4 (Priority-Flood) | On SRTM tile: (1) zero sinks remain, (2) output matches GDAL fill within ±ε, (3) runtime < 2× Planchon-Darboux for same tile |
| P0-T1 (DEV) | DEV output matches WhiteboxTools DevFromMeanElev within floating-point tolerance on test DEM |
| P0-H5 (HAND) | HAND = 0 along stream cells; HAND monotonically increases away from streams on synthetic V-shaped valley |

### General Criteria for All Items

Every new algorithm must satisfy:

- [ ] Unit tests with synthetic data (cone, half-pipe, stepped terrain, flat plane)
- [ ] Integration test with at least one real DEM (SRTM tile or similar)
- [ ] Docstring with formula reference, parameter description, and usage example
- [ ] WASM-compatible (no platform-specific dependencies; uses `maybe_rayon`)
- [ ] CLI exposure via `surtgis` binary (if applicable)
- [ ] Benchmark against at least one competitor output (GDAL, GRASS, WBT) when feasible

### RMSE Targets (from Florinsky Table 5.1)

For Evans-Young 3×3 with DEM RMSE = mz and grid spacing = w:

| Derivative | RMSE |
|------------|------|
| p, q (slope components) | mz / (√6 · w) |
| r, t (2nd-order pure) | √2 · mz / w² |
| s (2nd-order cross) | mz / (2w²) |

For Florinsky 5×5 (should be ~6× better for r, t):

| Derivative | RMSE |
|------------|------|
| p, q | √(527/70) · mz / (√6 · w) |
| r, t | √(2/35) · mz / w² |
| s | mz / (10w²) |

### Solar Radiation Targets (from Ruiz-Arias 2009)

- RMSE < 2.5 MJ/m²/day for daily global radiation (clear sky)
- MBE < 0.5 MJ/m²/day for systematic bias
- Validate against r.sun outputs or pyranometer networks

---

## References (Key Papers)

| ID | Citation | Cites | Used in |
|----|----------|-------|---------|
| Barnes 2014 | Barnes, R. et al. (2014). Priority-Flood. *Computers & Geosciences*, 62, 117–127 | 181 | P0-H4 |
| Corripio 2003 | Corripio, J.G. (2003). Vectorial algebra algorithms. *IJGIS*, 17(1), 1–23 | 161 | P3-SO2, P1-SO1 |
| De Reu 2013 | De Reu, J. et al. (2013). TPI limitations. *Geomorphology*, 186, 39–49 | 509 | P0-T1 |
| Fisher 1993 | Fisher, P. (1993). Viewshed uncertainty. *IJGIS*, 7(4), 331–347 | 118 | P2-V1 |
| Florinsky 2025 | Florinsky, I.V. (2025). *Digital Terrain Analysis*. 3rd ed. Elsevier | — | P0-C1–C4, P1-T1–T5, P2-T1–T5, INF-6 |
| Kopecky 2021 | Kopecky, M. et al. (2021). TWI calculation guides. *Sci. Total Environ.*, 757 | 234 | P0-H1 |
| Lindsay 2016 | Lindsay, J.B. (2016). Breach depressions. *Hydrol. Process.*, 30(19), 3437–3446 | — | P0-H3 |
| Newman 2018 | Newman, D.R. et al. (2018). Integral images for LTP. *Geomorphology*, 307, 68–79 | 43 | P0-T1 |
| Newman 2022 | Newman, D.R. et al. (2022). Gaussian scale-space. *Geomatics*, 2(1) | 9 | P1-T7 |
| Qin 2011 | Qin, C.-Z. et al. (2011). TWI with max downslope gradient. *Precision Agriculture*, 12(1) | 128 | P3-H3 |
| Ruiz-Arias 2009 | Ruiz-Arias, J.A. et al. (2009). DEM solar benchmark. *IJGIS*, 23(9), 1049–1076 | 93 | P2-SO2 |
| Steger 2022 | Steger, C. et al. (2022). HORAYZON v1.2. *Geosci. Model Dev.* | 15 | P2-SO4 |
| Tarboton 1997 | Tarboton, D.G. (1997). D-infinity. *Water Resources Research*, 33(2), 309–319 | — | P0-H2 |
| Wang 2019 | Wang, F. et al. (2019). Three-band spectral index. *Field Crops Research* | 19 | P1-IM1 |
| Wu 2019 | Wu, Q. et al. (2019). Nested depressions. *JAWRA*, 55(4), 911–927 | 55 | P3-H2 |
