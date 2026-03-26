# Blind Peer Review Simulation

**Journal**: Environmental Modelling & Software (EMS)
**Manuscript**: ENVSOFT-D-26-XXXXX
**Title**: "A High-Performance Geospatial Analysis Library in Rust with WebAssembly and Python Support"
**Review type**: Single-anonymized (reviewers know authors are blinded)
**Date**: February 2026

---

## Pre-Review Format Compliance Check

| Requirement | Status | Detail |
|---|---|---|
| Abstract <= 150 words | PASS | ~126 words |
| Highlights: 3-5 bullets | PASS | 5 highlights |
| Highlights: each <= 85 chars | PASS | Max 65 chars |
| Graphical abstract | PASS | Included (graphical_abstract.pdf) |
| Keywords: 1-7 | PASS | 6 keywords |
| Keywords: no "and"/"of" | PASS | All clean |
| Software availability section | PASS | Present with all required fields |
| CRediT statement | PASS | Present |
| Competing interests | PASS | Declared |
| AI declaration | PASS | Present (Claude for code + drafting) |
| Funding statement | PASS | Present (no funding) |
| References cited = bib entries | WARNING | `zevenbergen1987` in .bib but never cited |
| Data availability | PASS | Present |
| LaTeX format | PASS | elsarticle class |

**Quantitative summary**: 44 pages, ~5,500 words body text, 14 tables, 6 figures, 4 equations, 21 references.

---

## Phase 1: Associate Editor Desk Assessment

**AE**: Dr. [Redacted], Associate Editor, Environmental Modelling & Software

### Scope Assessment

The manuscript describes a new open-source Rust library for terrain analysis, hydrology, and remote sensing that compiles to native, WebAssembly, and Python targets. It falls within the journal's scope of "Development and application of environmental software" and "GIS, remote sensing and image processing."

### Desk Assessment

The manuscript presents a software contribution with systematic benchmarking against established tools (GDAL, GRASS, WhiteboxTools) and includes an environmental case study. The work is relevant to the EMS readership.

**Concerns for reviewers to evaluate**:
1. The paper is lengthy (44 pages, 14 tables). Reviewers should assess whether this length is justified or if material should move to supplements.
2. Single-author paper claiming 127 algorithms and three deployment targets is ambitious. Reviewers should assess the credibility and completeness of the claimed contributions.
3. The AI disclosure states Claude was used for "code development assistance and manuscript drafting." Reviewers should evaluate whether the intellectual contribution is clearly the author's own.
4. The reference list (21 entries) appears light for a paper of this scope.

**Decision**: Send to review. Assigning three reviewers with complementary expertise.

**Reviewer assignments**:
- Reviewer 1: Expert in digital terrain analysis and geomorphometry
- Reviewer 2: Expert in scientific software engineering and HPC
- Reviewer 3: Expert in computational hydrology and applied GIS

---

## Phase 2: Reviewer Reports

---

### Reviewer 1 — Digital Terrain Analysis / Geomorphometry

**Expertise**: Geomorphometry, surface derivatives, curvature analysis, landform classification
**Confidence**: High

#### Summary

This manuscript presents a Rust-based geospatial library implementing 127 algorithms with native, WebAssembly, and Python deployment targets. The key claimed contributions are: (1) the first open-source implementation of Florinsky's complete 14-curvature system, (2) WebAssembly-enabled terrain analysis, and (3) systematic performance benchmarks against GDAL, GRASS, and WhiteboxTools.

The paper addresses a genuine need for high-performance, cross-platform terrain analysis tools. However, I have significant concerns about the depth of validation, the breadth of the literature review, and certain methodological choices that need to be addressed before publication.

#### Major Issues

**M1. Insufficient validation of the 14-curvature system — the paper's primary novelty claim.**

The paper claims as a "key contribution" the open-source implementation of Florinsky's complete 14-curvature morphometric system. However, the validation (Section 5.2) only tests slope, aspect, mean curvature (H), and Gaussian curvature (K) against analytical solutions. The remaining 12 curvatures — which are the novel part of the contribution — are not validated at all. The statement that they are "derived from combinations of H, K, and the partial derivatives" does not guarantee correctness of the implementation.

At minimum, the authors should:
- Validate principal curvatures ($k_{min}$, $k_{max}$) against analytical solutions (these have closed-form expressions for the Gaussian hill)
- Validate at least horizontal ($k_h$) and vertical ($k_v$) curvatures against SAGA's implementation (since SAGA implements these)
- Provide visual or quantitative comparison of the 6 curvatures that overlap with SAGA
- Include a table showing which of the 14 curvatures were validated and how

Without this, the primary novelty claim rests on untested code.

**M2. Incomplete literature review and missing comparisons.**

The reference list contains only 21 entries for a 44-page manuscript claiming comprehensive terrain analysis capabilities. Several important omissions:

- **RichDEM** (Barnes, 2016) is mentioned in the introduction but never compared in benchmarks. RichDEM is a C++ library with Python bindings implementing Priority-Flood, flow accumulation, and terrain derivatives — it is the closest functional analog to the presented software and should be included in performance comparisons.
- **No citation of the Evans-Young method** despite using it for curvature computation (Section 2.2.2, "Evans-Young method on a 3×3 moving window"). The original references (Evans, 1972; Young, 1978) should be cited.
- **No citation for the D8 flow direction algorithm** (O'Callaghan and Mark, 1984).
- **No citation for TFGA** flow direction, which is listed in Table 2 but undefined.
- **No citation for specific WASM benchmarking methodology** or V8 engine characteristics.
- The geomorphon algorithm reference (Jasiewicz and Stepinski, 2013) is cited but not validated.
- No reference to the extensive curvature comparison literature (e.g., Minár et al., 2020, "A comprehensive system of definitions of land surface curvatures...").

**M3. The environmental case study lacks scientific depth.**

The case study (Section 4) demonstrates that the software can compute terrain derivatives from a DEM. However, it does not generate any novel environmental insight. The flood susceptibility mapping via HAND thresholds is a simple threshold classification that has been demonstrated many times. For EMS, case studies should "be illustrated with applications in the environmental fields" that reveal something meaningful about the environmental system.

Suggestions:
- Compare the HAND-based flood mapping against observed flood extents or other flood hazard maps
- Show how the Florinsky curvatures provide additional discriminating power beyond traditional derivatives for the specific study area
- Quantify the added value of having all algorithms in a single workflow versus the traditional multi-tool approach (beyond the assertion that it "eliminates the need to chain" tools)

#### Minor Issues

**m1.** The abstract mentions "4.5× over GRASS GIS" for slope, which comes specifically from the 20K benchmark. At smaller scales (1K), the advantage is 31×. The abstract should clarify which scale is being reported or state a range.

**m2.** Section 2.2.1 states that Horn's method uses "weights [1, 2, 1]" but does not give the full kernel. For completeness and reproducibility, the full 3×3 weighting scheme for $\partial z / \partial x$ and $\partial z / \partial y$ should be presented, or a precise reference to the specific formulation used.

**m3.** The claim that SurtGIS is "the first open-source library to implement the complete 14-type system" needs qualification. The 14-type system is from Florinsky (2025), which was published very recently. The priority claim should acknowledge that prior implementations may exist in other packages not surveyed.

**m4.** The geomorphon results in the case study (44.9% slope, 27.1% hollow, 26.5% spur) use unspecified parameters. What lookup radius and flatness threshold were used? These substantially affect the classification.

**m5.** Table 1 lists "surtgis-cloud" as providing "STAC catalog client, COG streaming reader" but this functionality is never benchmarked or demonstrated. If it is mentioned, it should be demonstrated or its status clarified (e.g., "experimental").

**m6.** The sign convention note for H (Section 2.2.2) is helpful but should reference which specific equation defines the convention. As written, the reader must check Eq. 2 and deduce that the negative sign in the numerator produces the stated convention. Making this explicit would prevent errors.

**m7.** The `zevenbergen1987` entry is present in the bibliography file but never cited in the text. This orphan reference should be removed.

#### Recommendation: **Major Revision**

The primary novelty claim (14-curvature system) requires proper validation. The literature review needs significant expansion. The case study should demonstrate scientific value beyond software demonstration.

---

### Reviewer 2 — Scientific Software Engineering / HPC

**Expertise**: High-performance computing, scientific software design, benchmarking methodology, Rust/C++
**Confidence**: High

#### Summary

The manuscript presents a well-engineered Rust geospatial library with cross-platform deployment capabilities. The benchmarking methodology is generally sound, with honest reporting including cases where competitors outperform the presented software (GDAL for hillshade). The architectural decisions (workspace crates, maybe_rayon pattern, triple-target compilation) are well-motivated. However, I have concerns about the benchmarking completeness, memory analysis, and several claims that need more rigorous support.

#### Major Issues

**M1. Benchmarks on a single hardware platform with a non-standard CPU governor.**

All benchmarks use an Intel i7-1270P with the `powersave` CPU governor (acknowledged in Section 5). This is problematic:

- `powersave` governor dynamically scales frequency, introducing measurement variance. Performance benchmarks should use `performance` governor to minimize variance, or at minimum report CPU frequency during measurement.
- The i7-1270P is a mobile processor with a heterogeneous P-core/E-core architecture. The authors acknowledge this affects thread scaling (Section 5.1.2) but do not address how it affects the *main benchmark results*. Rayon's work-stealing may schedule differently across P/E cores depending on system load, introducing non-deterministic performance.
- A single hardware platform makes it impossible to assess generalizability. The paper should include at minimum one additional platform (e.g., AMD Ryzen or Intel Xeon) or explicitly limit the claim scope.

The 10-repetition measurements partially mitigate variance, but the CV analysis (Section 5.1, "maximum observed CV was 6.4%") should report per-algorithm CVs rather than just the maximum. A table of CVs across all algorithm×size×tool combinations would strengthen confidence.

**M2. The I/O comparison is not fully fair despite the authors' efforts.**

The paper commendably uses uncompressed Float32 GeoTIFFs and full-pipeline measurements. However:

- GDAL uses streaming/tiled I/O even for uncompressed files. SurtGIS reads the entire file into memory. This means GDAL can begin computing before the full file is read, while SurtGIS must wait for complete deserialization. The "full pipeline" time for GDAL thus includes overlapped I/O+compute, while SurtGIS is strictly sequential (read → compute → write). This is acknowledged implicitly in the I/O breakdown but never discussed as a fundamental architectural advantage of GDAL's approach.
- The authors report SurtGIS "compute-only" time separately, which is useful. However, the "Compute" column in Tables 3-6 likely excludes memory allocation overhead (creating output arrays), which is part of the real algorithmic cost. This should be clarified.
- WhiteboxTools 2.4 is described as "Rust" in Table 8 (Feature comparison), but WhiteboxTools Open Core is actually written in Go/Rust hybrid. The language attribution should be verified and corrected if wrong.

**M3. Memory analysis is superficial.**

Table 7 reports peak RSS measured with `/usr/bin/time -v`, which is a coarse metric:
- Peak RSS includes the entire process image, shared libraries, and runtime overhead. For GRASS, the 347 MB includes the GRASS runtime environment loaded into the `grass --tmp-location` process, not just the algorithm's data.
- No memory profiling over time (e.g., heaptrack, valgrind massif) is presented. A temporal memory profile would reveal whether SurtGIS holds multiple full copies simultaneously (input + output + intermediate) and where optimizations are possible.
- The claim "4.4-6.8× the raw DEM size" should be supported by analysis: which copies are held simultaneously? For slope (input → output, no intermediate), the theoretical minimum is 2× (input + output = 764 MB for 10K Float64). The actual 2,596 MB is 6.8× the raw DEM or 3.4× the theoretical minimum for Float64. This gap should be explained.

**M4. Thread scaling analysis is weak.**

Table 9 shows n=3 repetitions, which is insufficient for meaningful conclusions about scaling efficiency. The interpretation also has issues:

- The claim "near-ideal scaling: ~7.6s at 1 thread to ~1s at 8 threads (7.6×)" is stated but the table only shows *total* time, not compute-only. These compute-only numbers should be presented in the table.
- No parallel efficiency metric (speedup/threads) is computed.
- The Amdahl's law argument is qualitative. A formal fit of Amdahl's model ($S(n) = 1/((1-p) + p/n)$) to the data would yield the parallel fraction $p$ and strengthen the analysis.
- The 2→4 thread "stall" (9.53s → 9.48s) is explained by P-core/E-core heterogeneity, but this is speculative. An experiment pinning threads to P-cores only (e.g., via `taskset`) would confirm this hypothesis.

#### Minor Issues

**m1.** The Cargo workspace architecture (Section 2.1, Table 1) is implementation detail more appropriate for a software manual than a journal paper. Consider condensing to text.

**m2.** The `maybe_rayon` pattern description (Section 2.1) is thorough but the paragraph reads like documentation. The journal audience may not need the level of detail about `par_iter()` vs `iter()` re-exports. A sentence or two would suffice with a link to the documentation.

**m3.** The code listing in Section 3 (Illustrative Example) shows a Python session. The function names (`surtgis.slope`, `surtgis.advanced_curvature`) suggest a flat API without namespacing. For 56 functions this may lead to namespace pollution. Is there a submodule structure? This is relevant to software design quality.

**m4.** Section 5 states "CPU governor set to powersave (default)" — it should also report whether Turbo Boost was enabled, as this significantly affects single-threaded vs. multi-threaded comparisons.

**m5.** The WASM benchmark uses Node.js 22.19 (V8 engine). Browser WASM performance can differ significantly from Node.js due to different memory management, GC behavior, and JIT compilation. Since the paper claims browser-based analysis as a key contribution, at least one browser benchmark (e.g., Chrome, Firefox) should be included or the limitation acknowledged.

**m6.** Line counts: "~45,000 lines of Rust code" — what does this include? Production code only, or tests + examples + benchmarks? A breakdown (e.g., via `tokei` or `cloc`) would be informative.

**m7.** The AI declaration states Claude was used for "code development assistance and manuscript drafting." This is a significant statement for a software paper where the code IS the contribution. The author should clarify the extent of AI assistance: was it used for algorithm design, debugging, writing specific functions, or only for boilerplate? EMS readers will want to assess the intellectual authorship of the code itself.

#### Recommendation: **Major Revision**

The benchmarking methodology, while honest, has significant gaps that weaken the paper's claims. The memory and thread scaling analyses are too superficial for a performance-focused paper. The I/O fairness question needs explicit discussion.

---

### Reviewer 3 — Computational Hydrology / Applied GIS

**Expertise**: Hydrological modeling, DEM processing, watershed analysis, GIS applications in environmental science
**Confidence**: Medium-High

#### Summary

This paper presents a new geospatial library that provides terrain analysis and hydrological algorithms across multiple deployment platforms. As a user of terrain analysis tools in hydrological research, I find the software potentially useful. The hydrological algorithm validation (Section 5.2.3) is one of the stronger parts of the paper. However, the paper has an identity problem: it tries to be both a comprehensive software description paper AND a benchmarking study, and neither is fully developed as a result.

#### Major Issues

**M1. The paper's scope is too broad and unfocused.**

The manuscript attempts to cover:
- 127 algorithms across 6 categories (terrain, hydrology, imagery, ecology, texture, classification)
- 3 deployment platforms (native, WASM, Python)
- Performance benchmarks against 3 competitors
- Numerical accuracy validation
- Cross-platform performance comparison
- A case study
- Flood susceptibility mapping
- Thread scaling analysis
- Memory analysis

This is too much for one paper. The result is that important topics are treated superficially:
- Only 6 of 127 algorithms are benchmarked
- Only slope is validated analytically (plus basic H and K)
- Only one real-world DEM is used for validation
- The case study adds little beyond demonstrating that the software works
- Remote sensing, ecology, texture, and classification modules are mentioned in lists but never demonstrated

**Suggestion**: Split this into (a) a focused software description paper covering architecture, curvature contribution, and deployment model, with (b) a separate benchmarking paper, or (c) significantly reduce scope by dropping modules not directly related to terrain/hydrology (which is EMS's core audience).

**M2. Hydrological algorithms are underdeveloped relative to the claims.**

The paper lists three depression-handling algorithms, four flow direction methods, and derived indices (TWI, SPI, STI, HAND) in Table 2. However:

- Only Priority-Flood filling and D8 flow accumulation are benchmarked. Where are the breach algorithm benchmarks? Where is MFD vs D-infinity comparison?
- The D-infinity implementation is never validated against TauDEM's reference implementation, which is the standard in the field. This is a significant omission for a paper published in an environmental modeling journal.
- TWI computation is shown in the code listing (Section 3) but never validated or benchmarked. TWI depends critically on the flow accumulation method used; a sensitivity analysis (TWI from D8 vs MFD vs D-inf) would be valuable for the hydrology audience.
- The HAND computation is used in the case study but never cross-validated against GRASS's r.stream.distance or other implementations.
- Stream network extraction is validated (Table 6, F1=0.973 vs GRASS) but the threshold sensitivity is not discussed. The threshold of 100 cells is stated but not justified.

**M3. The case study does not follow environmental modeling best practices.**

The HAND-based flood susceptibility mapping (Section 4) has several issues:

- The threshold values (5, 10, 15, 25 m) for flood susceptibility classification are presented without justification. Nobre et al. (2011) used different thresholds calibrated to their study area. Are these thresholds appropriate for a deeply incised Andean valley with median HAND of 271 m?
- There is no validation against observed flooding events, historical records, or other flood hazard products.
- The study area (high Andes, 2859-5979 m elevation) is unusual for flood susceptibility mapping. Most HAND applications target lowland floodplains. The choice of study area should be justified.
- The areal statistics (3.4% high risk, 1.7% moderate) are presented without uncertainty quantification. How sensitive are these to the stream extraction threshold or the flow algorithm choice?

**M4. Missing comparison with operational hydrological workflows.**

For an environmental modeling audience, the paper should demonstrate how SurtGIS fits into real hydrological workflows:

- How does it compare with established Python hydrological packages like pysheds, pyflwdir, or TauDEM's Python wrappers?
- Can it ingest standard hydrological datasets (e.g., HydroSHEDS, MERIT Hydro)?
- Does it support common output formats for hydrological model input (e.g., HEC-RAS cross-sections, SWAT subbasin delineation)?
- The cloud-native capabilities (STAC, COG) are mentioned but never demonstrated for a hydrological use case.

#### Minor Issues

**m1.** The footnote in Table 6 explaining GRASS r.watershed semantics is important but insufficient. `r.watershed` computes a different quantity than pure D8 flow accumulation: it uses a least-cost path approach with internal depression filling. This means the comparison is between different algorithms, not different implementations of the same algorithm. This should be stated clearly in the text, not buried in a footnote.

**m2.** The breach algorithm (Lindsay, 2016) is listed but never demonstrated. Since breaching often produces more hydrologically realistic results than filling (especially in low-relief areas), a comparison between the three depression-handling approaches on real data would be valuable.

**m3.** The TWI reference (Kopecky et al., 2021) specifically discusses guidelines for TWI calculation including the importance of flow algorithm choice and contributing area threshold. The paper should acknowledge that their TWI implementation follows (or departs from) these guidelines.

**m4.** The SPI and STI indices are listed in Table 2 but never defined, validated, or demonstrated. If they are not central to the contribution, they should not be highlighted in the paper.

**m5.** The "4 flow direction methods" claim counts D8, MFD, D-infinity, and "TFGA". What is TFGA? It is never defined or referenced anywhere in the paper.

**m6.** The watershed delineation capability is listed in Table 2 but never demonstrated or validated. For a hydrology-focused claim, this is a notable omission.

#### Recommendation: **Major Revision**

The hydrological components need significantly more validation and demonstration. The paper's scope should be narrowed or the treatment of each topic deepened. The case study needs methodological improvement to meet EMS standards for environmental applications.

---

## Phase 3: Associate Editor Decision

**Manuscript**: ENVSOFT-D-26-XXXXX
**Decision**: **Major Revision**

Dear Author,

Thank you for submitting your manuscript to Environmental Modelling & Software. Your paper has been reviewed by three experts with complementary expertise in terrain analysis, software engineering, and computational hydrology. All three reviewers recognize the potential value of the software contribution but recommend Major Revision.

### Summary of Key Concerns

The reviewers identified several converging concerns:

**1. Validation depth (R1-M1, R3-M2)**: The primary novelty claim — the 14-curvature system — is not validated beyond the basic H and K curvatures. The hydrological algorithms (D-inf, breach, TWI, HAND) are similarly under-validated. For a software paper in EMS, comprehensive validation is essential.

**2. Benchmarking rigor (R2-M1, R2-M2, R2-M4)**: While the benchmarking methodology is generally honest, the single-platform limitation, the `powersave` governor choice, and the superficial thread scaling analysis weaken the performance claims. The I/O fairness between streaming (GDAL) and in-memory (SurtGIS) approaches should be explicitly discussed.

**3. Paper scope and focus (R3-M1)**: All reviewers noted that the paper attempts to cover too much ground. The result is that important topics (curvature validation, hydrological workflows, case study rigor) are treated superficially. I recommend the authors focus the paper on the core contributions and move peripheral material to supplements.

**4. Literature coverage (R1-M2)**: The reference list (21 entries) is thin for a paper of this scope. Important tools (RichDEM, pysheds), foundational algorithms (D8, Evans-Young), and relevant methodological literature are missing.

**5. Case study rigor (R1-M3, R3-M3)**: The environmental case study needs to demonstrate scientific value beyond software demonstration. Threshold justification, validation against independent data, and sensitivity analysis are expected for EMS.

**6. AI disclosure (R2-m7)**: Given the extensive AI disclosure (code development + manuscript drafting), the revision should clarify the extent of AI assistance to assure readers of intellectual authorship.

### Required Actions for Revision

| # | Issue | Priority |
|---|---|---|
| 1 | Validate additional curvatures (at minimum $k_{min}$, $k_{max}$, $k_h$, $k_v$) analytically and/or against SAGA | HIGH |
| 2 | Expand reference list (+15-20 references minimum): RichDEM, Evans-Young, O'Callaghan-Mark, D8, TFGA, pysheds, Minár et al. | HIGH |
| 3 | Benchmark RichDEM for comparable algorithms or justify exclusion | HIGH |
| 4 | Address I/O fairness: streaming vs. in-memory architectural difference | HIGH |
| 5 | Narrow paper scope: move imagery/ecology/texture details to supplement | MEDIUM |
| 6 | Case study: justify HAND thresholds, discuss study area choice, add sensitivity analysis | MEDIUM |
| 7 | Define TFGA or remove from claims | MEDIUM |
| 8 | Validate D-infinity against TauDEM or justify omission | MEDIUM |
| 9 | Add at least one additional hardware platform or explicitly limit claim scope | MEDIUM |
| 10 | Thread scaling: increase n, add compute-only column, fit Amdahl's model | LOW |
| 11 | Acknowledge compressed GeoTIFF limitation more prominently (already partly done) | LOW |
| 12 | Remove orphan reference (zevenbergen1987) | LOW |

### Formatting Notes

- The manuscript is 44 pages in review format. Consider condensing: move some tables (e.g., per-size benchmark tables) to supplementary material, retaining only the summary table and key figures.
- 14 tables is excessive. Consider merging related tables (e.g., 5K and 10K results).
- The code listing (Section 3) could be shortened or moved to a supplementary notebook.

I look forward to receiving your revised manuscript.

Sincerely,
Associate Editor
Environmental Modelling & Software

---

## Appendix: Issue Severity Classification

### Issues likely to cause Reject if not addressed:
- R1-M1 (curvature validation): Without validating the claimed primary contribution, the paper's core novelty is unsubstantiated
- R1-M2 (literature): 21 references for a 44-page paper is well below EMS norms
- R3-M1 (scope): The unfocused scope weakens every individual claim

### Issues that merit Major Revision:
- R2-M1 (single platform + powersave governor)
- R2-M2 (I/O fairness discussion)
- R2-M3 (memory analysis)
- R3-M2 (hydrological validation gaps)
- R3-M3 (case study rigor)
- R3-M4 (operational workflow context)

### Issues that are Minor Revision:
- R1-m1 through m7, R2-m1 through m7, R3-m1 through m6

---

## Scoring Matrix (1-5, where 5 = excellent)

| Criterion | R1 | R2 | R3 | Mean |
|---|---|---|---|---|
| Novelty/originality | 3 | 3 | 2 | 2.7 |
| Technical soundness | 2 | 3 | 3 | 2.7 |
| Significance to EMS readership | 3 | 3 | 3 | 3.0 |
| Clarity of presentation | 4 | 4 | 3 | 3.7 |
| Completeness of literature review | 2 | 3 | 2 | 2.3 |
| Reproducibility | 4 | 3 | 3 | 3.3 |
| Environmental application | 2 | --- | 2 | 2.0 |
| **Overall** | **2.9** | **3.2** | **2.6** | **2.9** |

**Consensus**: The software has potential but the paper is not ready for publication in its current form. Major revision required on validation, literature, scope, and case study.
