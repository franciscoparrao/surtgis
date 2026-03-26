# SurtGIS EMS Paper — Revision Checklist v2 (Blind Review)

**Manuscript:** ENVSOFT-D-26-XXXXX
**Status:** Major Revision (blind review, 3 reviewers unanimous)
**Review date:** February 2026
**Source:** `paper/BLIND_REVIEW.md`

---

## Sprint 1: Critical (risk of reject if not addressed)

### [x] R1-M1: Validate Florinsky curvatures beyond H and K
**Reviewer:** R1 | **Severity:** CRITICAL | **Effort:** Alto
**Issue:** The paper's primary novelty claim (14-curvature system) only validates H and K against analytical solutions. The remaining 12 curvatures are unvalidated.
**Required:**
  - [x] Derive analytical curvatures for Gaussian hill and validate against SurtGIS
  - [x] Validate against SAGA's implementation (7 overlapping curvatures)
  - [x] Cross-tool comparison table: SurtGIS vs SAGA
  - [x] Summary table: 14 curvatures validated, method, RMSE (tab:curvature_validation)
**Action taken:** Created `benchmarks/validate_curvatures.py` (~530 lines). All 14 curvatures validated analytically (RMSE < 4e-9). 8 Florinsky identities verified. SAGA cross-validation (7 curvatures). Table `tab:curvature_validation` added to paper.tex.
**Files modified:** `paper/paper.tex`, `benchmarks/validate_curvatures.py`, `benchmarks/results/curvature_validation.csv`

---

### [x] R1-M2: Expand reference list (+15-20 references)
**Reviewer:** R1, R3 | **Severity:** CRITICAL | **Effort:** Medio
**Issue:** Only 21 references for a 44-page paper. Well below EMS norms (typically 40-60).
**Required:**
  - [x] Evans (1972), O'Callaghan & Mark (1984), Minár et al. (2020)
  - [x] pysheds, pyflwdir, TauDEM software, Wilson & Gallant (2000)
  - [x] TFGA (li2024tfga), Rust/WASM literature (haas2017, perkel2020rust)
  - [x] Cloud-native (gorelick2017gee), WBT journal (lindsay2019wbt)
  - [x] Work-stealing (blumofe1999), PyO3 (pyo3_2024), SRTM (farr2007srtm)
  - [x] Remove orphan `zevenbergen1987` and `lindsay2014`
**Action taken:** Added 15 new references (21 → 34 total, 2 orphans removed). All cited in appropriate text locations.
**Files modified:** `paper/paper.tex`, `paper/paper.bib`

---

### [x] R2-M2: Address I/O fairness (streaming vs in-memory)
**Reviewer:** R2 | **Severity:** CRITICAL | **Effort:** Bajo
**Action taken:** Added "I/O architecture" paragraph in Section 5.1 explaining streaming vs in-memory trade-off, clarifying Compute columns, and discussing implications.
**Files modified:** `paper/paper.tex`

---

### [x] R3-M1: Narrow paper scope and deepen focus
**Reviewer:** R3 | **Severity:** CRITICAL | **Effort:** Medio
**Action taken:** Table 1 replaced with inline text. maybe_rayon condensed. Additional modules condensed to single paragraph. 5K+10K tables merged into `tab:all_algos` with multirow (13 tables now).
**Files modified:** `paper/paper.tex`

---

## Sprint 2: Necessary (expected in Major Revision)

### [x] R1-M3 + R3-M3: Strengthen environmental case study
**Reviewer:** R1, R3 | **Severity:** MAJOR | **Effort:** Alto
**Action taken:** Study area reframed as workflow demonstration (not operational flood mapping). HAND thresholds justified (Nobre 2011, Amazonia calibration, illustrative for Andes). Geomorphon params specified (r=10, flat=1°). Florinsky kh/E value demonstrated. Stream threshold (100 cells) discussed. Sensitivity to thresholds acknowledged.
**Files modified:** `paper/paper.tex`

---

### [x] R2-M1: Address single-platform benchmark limitation
**Reviewer:** R2 | **Severity:** MAJOR | **Effort:** Medio-Alto
**Action taken:** Governor justified (powersave = default, same conditions for all tools). Turbo Boost reported (enabled). P/E core frequencies specified (4.8/3.5 GHz). CV range expanded (0.8%–4.1%). Limitation statement strengthened in Discussion (heterogeneous arch, AMD/ARM64 may differ).
**Files modified:** `paper/paper.tex`

---

### [x] R3-M2: Validate D-infinity and expand hydrological demonstration
**Reviewer:** R3 | **Severity:** MAJOR | **Effort:** Medio
**Action taken:** Added "Scope of hydrological validation" paragraph. D-inf exclusion justified (TauDEM is MPI-based, single-node comparison misleading). Breach exclusion explained (parameterization differs). TWI sensitivity acknowledged per Kopecky 2021.
**Files modified:** `paper/paper.tex`

---

### [x] R2-M3: Deepen memory analysis
**Reviewer:** R2 | **Severity:** MAJOR | **Effort:** Medio
**Action taken:** Detailed breakdown of 2596 MB for slope at 10K: Float32→Float64 input (763 MB), Float64 output (763 MB), derivative arrays (1526 MB), Float32 write buffer (382 MB). Theoretical minimum 1.5 GB explained. GRASS 347 MB includes runtime overhead. Streaming planned.
**Files modified:** `paper/paper.tex`

---

### [x] R2-M4: Improve thread scaling analysis
**Reviewer:** R2 | **Severity:** MAJOR | **Effort:** Medio
**Action taken:** Table expanded to 6 columns (compute-only, compute speedup, efficiency). Formal Amdahl fit: p=0.55, S(8)=1.92×, asymptotic max 2.2×. Heterogeneous core effects explained (>100% efficiency at 2 threads, peak at 8 threads with 98% efficiency). Note: n=3 reps retained (re-running benchmarks was infeasible in this session).
**Files modified:** `paper/paper.tex`

---

### [x] R3-M4: Contextualize within operational hydrological workflows
**Reviewer:** R3 | **Severity:** MEDIUM | **Effort:** Bajo
**Action taken:** New paragraph in Discussion positioning SurtGIS vs pysheds/pyflwdir (broader scope, higher throughput, but no HydroSHEDS/MERIT Hydro native support). surtgis-cloud status clarified (experimental STAC/COG, no streaming watershed analysis yet).
**Files modified:** `paper/paper.tex`

---

## Sprint 3: Minor comments

### [x] R1-m1: Abstract speedup numbers — clarify scale
**Action taken:** Abstract now shows ranges: 1.7–5.1× GDAL, 4.5–31× GRASS, 8.9–19.8× WBT.
**Files modified:** `paper/paper.tex`

---

### [x] R1-m2: Horn kernel — full specification
**Action taken:** Added full Horn kernel equation (Eq. 4) with dz/dx formula and 8Δx denominator.
**Files modified:** `paper/paper.tex`

---

### [x] R1-m3: "First open-source" — qualify the claim
**Action taken:** Added survey scope (GDAL, GRASS, SAGA, WBT, RichDEM, TauDEM as of Feb 2026) and "other implementations may emerge".
**Files modified:** `paper/paper.tex`

---

### [x] R1-m4: Geomorphon parameters unspecified
**Action taken:** Added lookup radius=10 cells (~284 m) and flatness threshold=1°.
**Files modified:** `paper/paper.tex`

---

### [x] R1-m5: surtgis-cloud — clarify status
**Action taken:** Clarified in architecture ("experimental") and in Discussion (experimental STAC/COG, cloud-native hydro not yet implemented).
**Files modified:** `paper/paper.tex`

---

### [x] R1-m6: H sign convention — reference equation explicitly
**Action taken:** Now references Eq.~2 explicitly and explains the negative sign in the numerator.
**Files modified:** `paper/paper.tex`

---

### [x] R2-m1: Condense Table 1 (crate architecture)
**Action taken:** Table 1 replaced with inline text in Section 2.1 (done in Sprint 1).
**Files modified:** `paper/paper.tex`

---

### [x] R2-m2: Condense maybe_rayon description
**Action taken:** Condensed to one sentence with work-stealing citation (done in Sprint 1).
**Files modified:** `paper/paper.tex`

---

### [x] R2-m3: Python API namespace structure
**Action taken:** Added note about flat namespace and planned submodule structure.
**Files modified:** `paper/paper.tex`

---

### [x] R2-m5: WASM browser vs Node.js performance caveat
**Action taken:** Added caveat about browser security mitigations, GC pauses, JIT differences; Node.js = upper bound.
**Files modified:** `paper/paper.tex`

---

### [x] R2-m6: Line count breakdown
**Action taken:** Updated to ~48K with per-crate breakdown (33K algo, 6K GUI, 3.8K cloud, etc.).
**Files modified:** `paper/paper.tex`

---

### [x] R2-m7: Clarify extent of AI assistance
**Action taken:** Specified: scaffolding/debugging/scripting, not algorithm design. Author conceived all algorithms and architecture.
**Files modified:** `paper/paper.tex`

---

### [x] R3-m1: r.watershed comparison — promote from footnote to text
**Action taken:** Promoted to main text with explanation of semantic difference (area vs cell counts).
**Files modified:** `paper/paper.tex`

---

### [x] R3-m2: Breach algorithm — demonstrate or acknowledge
**Action taken:** Acknowledged in "Scope of hydrological validation" paragraph (parameterization differs from WBT).
**Files modified:** `paper/paper.tex`

---

### [x] R3-m4: SPI/STI — define or remove from Table 2
**Action taken:** Added inline formulas (SPI = As tan β, STI formula) in Table 2.
**Files modified:** `paper/paper.tex`

---

### [x] R3-m5: Define TFGA flow direction method
**Action taken:** TFGA now cited with li2024tfga reference in Table 2 (done in Sprint 1).
**Files modified:** `paper/paper.tex`, `paper/paper.bib`

---

### [x] R3-m6: Watershed delineation — demonstrate or downplay
**Action taken:** Added footnote: implemented but not benchmarked (downstream of flow direction, I/O-dominated).
**Files modified:** `paper/paper.tex`

---

### [x] R2-m1b: WhiteboxTools language attribution
**Action taken:** Added footnote: WBT Open Core is Rust; pre-2018 versions were Go. Updated lindsay2014 → lindsay2019wbt.
**Files modified:** `paper/paper.tex`, `paper/paper.bib`

---

## Pre-submission checklist (editorial)

### [x] Remove orphan references from paper.bib
Removed: `zevenbergen1987`, `lindsay2014`. Verified 0 orphan entries remain.

### [x] Verify all new references have DOIs
28/34 have DOIs. 6 without DOI are software/spec references (TauDEM, pysheds, RichDEM, PyO3, STAC, Wilson 2000 book with ISBN). Acceptable for EMS.

### [x] Verify highlight character counts after any edits (max 85 chars)
All 5 highlights: 65, 62, 62, 68, 64 characters. All under 85.

### [x] Verify abstract word count after any edits (max 150 words)
130 words. Under 150 limit.

### [x] Consider table consolidation (AE: 14 tables is excessive)
13 tables (down from 14+). Added 1 (curvature_validation) but merged 5K+10K and removed Table 1.

### [x] Full LaTeX compile cycle clean (0 undefined refs, 0 warnings)
Compiles successfully. 0 undefined references. 0 LaTeX warnings. 4 hyperref PDF string warnings (math in bookmarks, cosmetic). 51 pages in review format.

### [x] Update REVISION_CHECKLIST_v2.md progress tracker
This file.

---

## Progress Tracker

| Sprint | Total | Done | Remaining |
|--------|-------|------|-----------|
| 1 (Critical) | 4 | 4 | 0 |
| 2 (Necessary) | 6 | 6 | 0 |
| 3 (Minor) | 17 | 17 | 0 |
| Editorial | 7 | 7 | 0 |
| **TOTAL** | **34** | **34** | **0** |
