# SurtGIS EMS Paper — Revision Checklist

**Manuscript:** ENVSOFT-D-26-00XXX
**Status:** Major Revision
**Deadline:** 60 days from submission

---

## Sprint 1: Critical (must-fix for acceptance)

### [x] R2-M1: WebAssembly — claim without evidence
**Reviewer:** R2 | **Severity:** MAJOR | **Effort:** Alto
**Issue:** WASM is in the title and abstract but no performance data, no deployment example, no browser benchmark.
**Options:**
  - (A) Create browser demo + basic benchmarks (Chrome DevTools timing, memory)
  - (B) Remove "WebAssembly" from title, reduce claims to brief mention in discussion
**Action taken:** Opción A. Benchmark WASM ejecutado en V8/Node.js 22.19 con los mismos DEMs (1K², 5K²). 6 algoritmos en 1K, 3 en 5K. Resultados: slope 10.7× overhead vs Rust MT, hillshade 31.7×. Tabla `tab:crossplatform` expandida con columna WASM. Nueva tabla `tab:wasm_1k` con benchmark 1K. Texto actualizado con análisis de overhead WASM vs ST y mención del web demo con 33 algoritmos.
**Files modified:** `paper/paper.tex` (Experiment 3), `benchmarks/bench_wasm.mjs` (nuevo), `benchmarks/results/experiment_wasm.csv` (nuevo)

---

### [x] R1-M1: Environmental case study lacks environmental question
**Reviewer:** R1 | **Severity:** MAJOR | **Effort:** Alto
**Issue:** Section 4 demonstrates computation but not environmental application. EMS requires software to address environmental questions.
**Options:**
  - (A) Add landslide susceptibility analysis using derived layers as predictors
  - **(B) Add HAND-based flood zone delineation** ← elegida
  - (C) Add geomorphon validation against field-mapped landforms
  - (D) Reframe as "demonstration" and acknowledge limitation
**Action taken:** Opción B. Se agregó párrafo "Flood susceptibility mapping" que usa HAND para delinear 5 zonas de riesgo (high <5m, moderate 5-10m, low 10-15m, minimal 15-25m, negligible >25m). Cuantifica: zona alto-moderado = 21.0 km² (5.1% del área). Nueva Figura `fig_flood_zones.pdf` con mapa continuo HAND + zonas clasificadas. Se cita Nobre et al. 2011 como referencia metodológica. Párrafo "Integration" describe cómo las capas derivadas forman un stack de predictores para modelos de susceptibilidad.
**Files modified:** `paper/paper.tex` (Section 4), `benchmarks/results/figures/fig_flood_zones.pdf` (nueva)

---

### [x] R1-M2: Incomplete algorithm verification (only 4 of 100+)
**Reviewer:** R1 | **Severity:** MAJOR | **Effort:** Medio
**Issue:** Accuracy validation covers slope, aspect, H, K only. Hydrological algorithms (fill, flow_acc, TWI, HAND) compared by speed but not correctness.
**Required:**
  - [x] Cross-tool comparison of fill output (pixel-by-pixel SurtGIS vs GRASS vs WBT)
  - [x] Cross-tool comparison of D8 flow accumulation grids
  - [x] Qualitative validation of stream networks against known features
  - [ ] Geomorphons comparison (if feasible) — omitido, no herramienta de referencia directa
**Action taken:** Tabla `tab:hydro_validation` con fill (RMSE<0.15m, R²=1.000), flow acc log (R²=0.949, r=0.987 vs WBT), streams (F1=0.973 vs GRASS, F1=0.938 vs WBT). Figura `fig_hydro_validation.pdf` con scatter plots y mapas de overlap. Datos en `benchmarks/results/hydro_validation.csv`.
**Files modified:** `paper/paper.tex` (Experiment 2, nueva subsection + tabla + figura), `benchmarks/results/hydro_validation.csv`, `benchmarks/results/figures/fig_hydro_validation.pdf`

---

## Sprint 2: Necessary (expected in revision)

### [x] R2-M2: Reproducibility of benchmarks
**Reviewer:** R2 | **Severity:** MAJOR | **Effort:** Medio
**Required:**
  - [x] Exact commit hash / version tag in paper
  - [x] Benchmark reproduction script (or reference to existing)
  - [x] Exact command lines for GDAL, GRASS, WBT
  - [x] System config (kernel, CPU governor, NUMA)
  - [x] Confirm DEM generation is deterministic (seed + library version)
  - [x] Raw CSV results as supplementary material
**Action taken:** Agregado al paper: commit 7493bc1, SurtGIS v0.1.1, Ubuntu 24.04 kernel 6.14.0, CPU governor=powersave, NUMA single-node. Command lines exactos para cada herramienta (gdaldem, grass --tmp-location, whitebox_tools). Referencia explícita a los 3 scripts de benchmark como supplementary. Seed=42 con NumPy 2.4 documentado.
**Files modified:** `paper/paper.tex` (sección Performance evaluation, párrafo de setup)

---

### [x] R1-M3: "100+ algorithms" claim unverifiable
**Reviewer:** R1 | **Severity:** MAJOR | **Effort:** Bajo
**Issue:** Tables list ~40-50 algorithms. Reader cannot verify "100+" from paper.
**Required:**
  - [x] Complete algorithm list as supplementary table
  - [x] Clarify counting methodology (e.g., NDVI and NDWI = 2 algorithms?)
**Action taken:** Creada tabla suplementaria con 127 algoritmos enumerados individualmente, organizados en 9 categorías. Se incluye metodología de conteo explícita: cada operación computacional distinta cuenta como 1 algoritmo; los 14 Florinsky curvatures se cuentan individualmente. Se indica cobertura por target (33 WASM, 56 Python, 127 native). Claim actualizado de "over 100" a "127 algorithms" en paper.tex.
**Files modified:** `paper/supplementary_algorithms.tex` (nuevo), `paper/paper.tex` (item 1 en enumerate)

---

### [x] R2-m2: Memory usage data
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Medio
**Issue:** Peak memory mentioned once (3.2 GB) but not systematically measured.
**Required:**
  - [x] Measure peak RSS for each tool at 10K for at least slope, fill, flow_acc
  - [x] Add column to Table 4 or new table
**Action taken:** Medido peak RSS con `/usr/bin/time -v` a 10K² para 4 herramientas × 3 algoritmos. Resultados: SurtGIS 1.9-2.6 GB (in-memory arrays), GDAL 435 MB, GRASS 347 MB (tiled/streaming I/O), WBT 1.7-1.9 GB. Nueva tabla `tab:memory` en Experiment 1. Referencia en Discussion actualizada de "approximately 3.2 GB" a datos medidos.
**Files modified:** `paper/paper.tex` (nueva tabla + Discussion), `benchmarks/results/experiment_memory.csv` (nuevo)

---

## Sprint 3: Minor comments

### [x] R1-M4: Missing TauDEM in comparison
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** TauDEM is the reference D-infinity implementation. Omission is notable.
**Action taken:** Agregado TauDEM en la introducción junto con RichDEM y SAGA. Justificación explícita: modelo MPI para clusters distribuidos, no comparable en benchmarks single-node.
**Files modified:** `paper/paper.tex` (Introduction)

---

### [x] R1-m1: Verify WBT slope method
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Paper says "Zevenbergen-Thorne" — verify against WBT source code/docs.
**Action taken:** Verificado en código fuente WBT (slope.rs). WBT NO usa Zevenbergen-Thorne, usa polinomio Taylor bivariado 3er orden en ventana 5×5 (Florinsky 2016). Corregido en paper. Agregada referencia florinsky2016 al .bib.
**Files modified:** `paper/paper.tex` (Section 2.2.1), `paper/paper.bib`

---

### [x] R1-m2: Python binding overhead details
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Clarify whether NumPy<->Rust uses copy or buffer protocol.
**Action taken:** Agregado: "Data transfer uses NumPy's buffer protocol for zero-copy access where possible; when the Rust function requires owned data, a single memcpy is performed."
**Files modified:** `paper/paper.tex` (Section 2.5 Python paragraph)

---

### [x] R1-m3: Case study 0.3s timing uninformative
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** 0.3s on 0.5M cells is not meaningful for performance. Contextualize or remove.
**Action taken:** Contextualizado: "(0.5 M cells; for performance on larger DEMs, see Section 5)" — redirige al lector a los benchmarks reales.
**Files modified:** `paper/paper.tex` (Section 4 case study)

---

### [x] R1-m4 (bis): "Open-source first" — verify SAGA curvatures
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Medio
**Issue:** SAGA may implement some Florinsky curvatures. Verify claim.
**Action taken:** Verificado: SAGA implementa un subconjunto (profile, plan, tangential, longitudinal, minimal, maximal). NO implementa excess curvatures (k_he, k_ve), accumulation (K_a), ring (K_r), unsphericity (M), ni difference (E). Claim actualizado a "first open-source to implement the complete 14-type system" con mención explícita de lo que SAGA cubre.
**Files modified:** `paper/paper.tex` (Section 2.2.2 Florinsky)

---

### [x] R1-m5: Curvature sign convention
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Different references use different H sign conventions. State explicitly.
**Action taken:** Agregado después de Eq. 2: "H > 0 for convex surfaces (ridges), H < 0 for concave (valleys), following Florinsky 2025. Some references use opposite convention."
**Files modified:** `paper/paper.tex` (Section 2.2.2)

---

### [x] R1-m6: Compressed data caveat more prominent
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Caveat about compressed GeoTIFFs reducing SurtGIS advantage should be in abstract or conclusions.
**Action taken:** Agregado párrafo prominente en Discussion antes de "Future development": "An important caveat for practitioners: the benchmarks use uncompressed Float32 GeoTIFFs... SurtGIS's pure-Rust tiff crate decoder is slower than GDAL's optimized C codecs for compressed data."
**Files modified:** `paper/paper.tex` (Discussion section)

---

### [x] R1-m7: Identify measurements with CV > 20%
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** "Most below 20%" — which exceeded? Flag in tables.
**Action taken:** Verificado: ninguna medición excede CV 20%. Máximo observado: 6.4% (slope 20K²). Texto actualizado de "most below 20%" a "all below 10%, max 6.4%".
**Files modified:** `paper/paper.tex` (Experiment 1 methodology)

---

### [x] R1-m8: Exact gdaldem options
**Reviewer:** R1 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Specify -compute_edges and algorithm options used.
**Action taken:** Command line exacto ya estaba en R2-M2. Agregada nota: "(default Horn algorithm, -compute_edges not used to match SurtGIS boundary handling)".
**Files modified:** `paper/paper.tex` (Performance evaluation, command lines)

---

### [x] R2-m1: Explain maybe_rayon in more detail
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Pattern mentioned but not explained. Useful for Rust community.
**Action taken:** Expandido con detalle técnico: Cargo feature flag `parallel`, re-export de rayon::prelude vs API secuencial compatible, `maybe_par_iter()` como abstracción.
**Files modified:** `paper/paper.tex` (Section 2.1 Architecture)

---

### [x] R2-m3: 41/100+ Python coverage
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Why only 41 functions exposed? Policy or limitation?
**Action taken:** Actualizado de 41 a 56 funciones (conteo actual). Agregado: "The remaining 71 algorithms are progressively exposed in each release; all 127 are accessible via the native Rust API."
**Files modified:** `paper/paper.tex` (Section 2.5 Python paragraph)

---

### [x] R2-m4: Thread scaling curve
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Medio
**Issue:** MT vs ST shown, but no scaling curve.
**Required:**
  - [x] Benchmark slope at 10K with 1,2,4,8,12,16 threads
  - [x] Add figure or table showing scaling
**Action taken:** Benchmark ejecutado: 1→13.9s, 2→9.5s, 4→9.5s, 8→7.3s, 12→7.4s, 16→7.7s. Speedup limitado a 1.9× end-to-end por I/O secuencial (~6.3s). Compute-only escala ~7.6× con 8 threads. Tabla `tab:scaling` agregada con análisis de Amdahl's law.
**Files modified:** `paper/paper.tex` (Experiment 1, nueva tabla + texto)

---

### [x] R2-m5: Error handling / robustness
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** No mention of NaN, NoData boundaries, non-square pixels, geographic CRS.
**Action taken:** Nuevo párrafo "Data robustness" en Section 2.1: propagación NoData, non-square pixels, Result types, geographic CRS caveat, NaN/infinity treatment.
**Files modified:** `paper/paper.tex` (Section 2.1 Architecture)

---

### [x] R2-m6: CI/CD and test count
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Software maturity indicators missing.
**Required:**
  - [x] Count unit/integration tests
  - [x] Describe CI pipeline (GitHub Actions?)
**Action taken:** 637 tests (unit + integration). CI: GitHub Actions con 9 jobs (check, clippy, test, test-no-parallel, wasm build, web demo, python build, format, integration). Agregado en Software Availability.
**Files modified:** `paper/paper.tex` (Software availability section)

---

### [x] R2-m7: "Auto" parallel imprecise in comparison table
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** GRASS r.watershed uses OpenMP; GDAL threads for I/O. "Manual" is imprecise.
**Action taken:** Revisado: SurtGIS "Rayon (auto)", WBT "Multi-thread", GRASS "OpenMP" con nota "r.watershed uses OpenMP; most other modules are single-threaded", SAGA "OpenMP", GDAL "I/O threads".
**Files modified:** `paper/paper.tex` (Table comparison)

---

### [x] R2-m8: Prebuilt wheels platforms
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** Which platforms have prebuilt wheels?
**Action taken:** Especificado en Python paragraph: "Prebuilt wheels are provided for Linux (x86_64, aarch64), macOS (x86_64, ARM64), and Windows (x86_64)."
**Files modified:** `paper/paper.tex` (Section 2.5 Python paragraph)

---

### [x] R2-m9: Graphical abstract AI policy
**Reviewer:** R2 | **Severity:** MINOR | **Effort:** Bajo
**Issue:** EMS policy: "The use of generative AI in production of artwork such as graphical abstracts is not permitted."
**Action taken:** AI declaration actualizada: "All figures were generated programmatically using matplotlib; no generative AI was used in the production of artwork."
**Files modified:** `paper/paper.tex` (AI declaration)

---

## Pre-submission checklist (editorial)

### [x] Abstract ≤ 150 words
Reducido de 161 a 130 palabras. Se incluyeron los datos WASM (10-32× overhead) y el conteo exacto (127 algoritmos).

### [x] Highlights: 3-5 items, each ≤ 85 characters
5 items, todos ≤85 chars (item 4 exactamente 85).

### [x] CRediT author contributions statement
Agregado: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft/review & editing, Visualization.

### [x] Data availability statement (Option C)
Agregado: referencia al repositorio GitHub con scripts, CSVs, y DEM generation code. Copernicus DEM referenciado como fuente libre.

### [x] Funding statement
Agregado: "This research did not receive any specific grant..."

### [x] All figures as separate PDF files
8 figuras + graphical abstract ya existen como PDFs en `benchmarks/results/figures/`.

### [x] Supplementary material: algorithm list + benchmark scripts
Creado `paper/supplementary_algorithms.tex` (127 algoritmos). Scripts: `benchmarks/run_benchmarks.py`, `bench_comparison.rs`, `bench_wasm.mjs`.

### [ ] Response letter: point-by-point to all reviewer comments
Pendiente — se creará al finalizar todas las correcciones.

---

## Progress Tracker

| Sprint | Total | Done | Remaining |
|--------|-------|------|-----------|
| 1 (Critical) | 3 | 3 | 0 |
| 2 (Necessary) | 3 | 3 | 0 |
| 3 (Minor) | 17 | 17 | 0 |
| Editorial | 8 | 7 | 1 |
| **TOTAL** | **31** | **30** | **1** |
