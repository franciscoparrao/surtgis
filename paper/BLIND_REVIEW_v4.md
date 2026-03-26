# Blind Peer Review Simulation v4 — Environmental Modelling & Software

**Manuscript:** ENVSOFT-D-26-XXXXX
**Title:** SurtGIS: A High-Performance Geospatial Analysis Library in Rust with WebAssembly and Python Support
**Date:** February 2026
**Review type:** Single anonymized (per EMS policy), minimum 2 reviewers
**Status:** Fresh submission (reviewers have NOT seen prior versions)

---

## Reviewer 1 — Geomorphometry / Terrain Analysis Specialist

**Expertise:** Digital terrain analysis, curvature systems, geomorphological mapping
**Confidence:** High
**Recommendation:** Minor Revision

### Summary

This manuscript presents SurtGIS, a Rust-based geospatial library with 127 algorithms, cross-platform deployment (native, WASM, Python), and an open-source implementation of Florinsky's 14-curvature system. The paper includes systematic benchmarks against GDAL, GRASS, and WhiteboxTools, analytical validation of curvatures, and an environmental case study. The work is technically sound and the curvature validation is particularly thorough. However, several issues regarding methodological comparability in benchmarks, claim consistency, and presentation need attention before publication.

### Major Issues

**M1. WhiteboxTools slope comparison is methodologically invalid and must be removed or reframed**

This is the most significant issue in the manuscript. The paper clearly establishes (Section 2.2, line 132; Section 5.2, Table 5) that WhiteboxTools uses a fundamentally different derivative estimation method (Florinsky 2016, 3rd-order Taylor on 5×5) compared to SurtGIS/GDAL/GRASS (Horn 1981, 3×3 kernel), with RMSE = 6.53° between methods. Yet the paper repeatedly claims "speedup" over WBT for slope:

- Abstract: "8.9--11.2× over WhiteboxTools"
- Highlights (item 4): "up to 11.2× over WBT"
- Table 1: WBT slope times with speedup ratios at all 4 scales
- Section 6 (line 548): "SurtGIS is 4.5--11.2× faster than... WhiteboxTools (11.2×) for slope"
- Discussion (line 554): "4.2--11.2× over WhiteboxTools for terrain derivatives"

**Claiming speedup over a tool that computes a different mathematical quantity is misleading.** WBT's 5×5 method performs ~2.8× more floating-point operations per pixel than Horn's 3×3 — it is expected to be slower. This is akin to benchmarking bilinear vs bicubic interpolation and calling the faster one "superior."

The paper already acknowledges the methodological difference (Section 2.2) but then ignores it when reporting performance. This internal inconsistency damages credibility.

**Required action:** Remove WBT slope from all speedup claims in abstract, highlights, Section 6, and discussion. WBT slope data may remain in the tables for completeness but must be clearly marked as "not directly comparable (different derivative method)" and excluded from speedup calculations. The aspect and hydrology WBT comparisons are valid and sufficient.

**M2. Abstract exceeds the 150-word limit**

EMS requires abstracts to be "a concise and factual abstract which does not exceed 150 words." The current abstract appears to contain approximately 160-170 words. It must be shortened, which the removal of WBT slope claims would partially address.

### Minor Issues

**m1. Curvature "first open-source implementation" claim needs qualification**

Line 148: "To our knowledge... SurtGIS is the first open-source library to implement the complete 14-type system as defined by Florinsky (2025)." The qualification "as of February 2026" is appropriate, but the claim rests on a survey of only six tools (GDAL, GRASS, SAGA, WBT, RichDEM, TauDEM). Consider adding: "based on a survey of major open-source terrain analysis libraries" to avoid implying an exhaustive search. Also, since the Florinsky (2025) book is very recent, the novelty window is narrow — this should be noted more explicitly.

**m2. Geomorphon parameter sensitivity acknowledged but not demonstrated**

Section 4 uses a single parameter configuration (radius=10, flatness=1°) for the case study. Line 245 notes that "These proportions are sensitive to the lookup radius parameter" with a citation to Jasiewicz & Stepinski (2013). A brief sensitivity test (e.g., 3 radii) or a supplementary figure would strengthen this claim and demonstrate practical utility.

**m3. Anisotropic curvature validation — exclusion criterion needs justification**

Line 422 states that the "bottom 10% of gradient magnitude" was excluded for the anisotropic sinusoidal ridge validation. The 10% threshold appears arbitrary. Please provide either (a) a mathematical justification relating the threshold to numerical conditioning of the curvature formulas, or (b) a sensitivity analysis showing that results are robust to threshold choice (e.g., 5%, 15%, 20%).

**m4. SAGA cross-validation interpretation**

Table S5 shows poor agreement between SurtGIS and SAGA for several curvatures ($R^2 < 0$ for Gaussian $K$ and $k_h$). While the text attributes this to different formulations (Evans 1979 vs Florinsky 2025), this deserves more careful analysis. Can the authors identify which specific formula differences account for each discrepancy? The Laplacian agreement (R² = 0.9999) is reassuring but insufficient to validate the complete system against an independent implementation.

**m5. Missing discussion of non-square pixel handling**

Line 91 mentions "Non-square pixels... are supported via the affine transform metadata" but no validation is presented for this case. Real-world DEMs from UTM reprojection near zone boundaries or from certain sensors have non-square pixels. At minimum, note this as an untested case.

### Editorial Issues

- Line 45 (highlights): After removing WBT slope claim, consider replacing with a Florinsky curvature or hydrology highlight
- Section 2.3 (line 152): The geomorphon description could note the connection to the line-of-sight algorithm more explicitly
- Table 5 (cross-tool): The WBT rows show identical values for all three comparisons (RMSE=6.53, R²=0.771, 16.2%) — add a note explaining this reflects WBT vs the Horn group collectively

---

## Reviewer 2 — Software Engineering / HPC / Benchmarking Specialist

**Expertise:** High-performance computing, benchmark methodology, software reproducibility
**Confidence:** High
**Recommendation:** Minor Revision

### Summary

The paper presents a well-engineered Rust library with thoughtful cross-platform design and systematic benchmarking. The benchmark methodology is generally sound: full-pipeline measurements, warm-up runs, median of 10 repetitions, and transparent reporting of I/O overhead. The paper is commendably honest about several limitations (uncompressed data caveat, in-memory model, GDAL hillshade advantage). However, issues with statistical rigor, reproducibility practices, and manuscript length warrant revision.

### Major Issues

**M1. Reproducibility gap: no version-pinned archival DOI**

EMS follows Option C for research data, requiring deposit in a relevant data repository. While the paper provides GitHub URLs (line 584), GitHub is not an archival repository — branches can be rebased, tags deleted, and repositories transferred. The commit hash (7493bc1) is cited but could be lost in a force-push.

**Required action:** Create a Zenodo release (or equivalent) with a DOI for the exact version benchmarked (v0.1.1, commit 7493bc1). This should include the source code, benchmark scripts, raw CSV results, and DEM generation scripts. Update the Software availability section with the DOI. This is standard practice for EMS software papers and ensures long-term reproducibility.

**M2. Manuscript length: 48 pages with 9 tables is excessive for EMS**

Even with 5 tables moved to supplementary, 48 pages is substantially longer than typical EMS research articles (typically 20-30 pages). The paper contains redundancy:

- Table 1 (slope by size) and Table 2 (all algorithms at 5K/10K) duplicate the slope data, as noted in the Table 2 caption
- The case study (Section 4) repeats information from Section 2 (algorithm descriptions)
- Section 6 (comparison) largely restates Section 5 results

**Required action:** Target ≤40 pages. Specific suggestions:
- Merge Tables 1 and 2 into a single comprehensive table
- Condense the case study to ~1 page (currently ~2 pages of text + 2 figures)
- Move Table 8 (WASM 1K) to supplementary
- Shorten Section 6 by referring to tables rather than restating numbers

### Minor Issues

**m1. Statistical reporting lacks confidence intervals on speedup ratios**

The paper reports CV < 10% and mentions that "speedup differences smaller than ~20% should be interpreted with caution" (line 290), which is good practice. However, no confidence intervals or uncertainty ranges are given for the speedup ratios themselves. Since speedups are ratios of two random variables, their uncertainty is not simply the CV of either. At minimum, report IQR-based ranges for key speedup claims (e.g., slope vs GDAL at 10K: 1.8× [1.6–2.0]).

**m2. Cross-platform experiment uses different measurement harnesses**

Table 8 caption (line 470) notes that "Compute-only values may differ from Table 3 due to separate measurement harnesses and system load conditions." This undermines cross-experiment comparability. Why were different harnesses used? Can the results be reconciled? At minimum, explain what differs and whether it affects conclusions.

**m3. Thread scaling data is too preliminary for publication**

The thread scaling results (Table S3) use only n=3 repetitions and show anomalies (>100% efficiency at 2 threads, non-monotonic behavior). While marked as "preliminary," publishing these data with Amdahl's law fitting (p=0.55) gives them an unwarranted appearance of rigor. Either:
(a) Re-run with n≥10 and include in supplementary, or
(b) Remove the Amdahl's law fit and present as qualitative evidence only

**m4. `powersave` governor choice**

The justification for retaining the default `powersave` governor (line 273) — that it "reflects the out-of-the-box configuration" — is reasonable but has a subtle issue: DVFS (Dynamic Voltage and Frequency Scaling) under powersave causes frequency to vary with thermal state, which could systematically bias later measurements (thermal throttling after warm-up). The warm-up runs mitigate this, but please confirm that no thermal throttling occurred (e.g., check `dmesg` for throttling events or log CPU frequency during benchmarks).

**m5. WASM benchmark environment**

Node.js V8 benchmarks (line 466) may not represent browser WASM performance — the paper acknowledges this (line 508) but still uses these numbers as the primary WASM assessment. Consider adding: a brief footnote about SharedArrayBuffer-based parallelism (already mentioned) and the upcoming WASM threads proposal timeline.

**m6. Test suite coverage**

637 tests are mentioned (line 582) but no code coverage metric is reported. For a library claiming 127 algorithms, average coverage of ~5 tests per algorithm raises questions about edge case handling. Consider reporting a coverage percentage.

### Editorial Issues

- The Software availability section is comprehensive and well-structured — commendable
- The AI declaration is appropriately detailed and honest
- Consider adding an ORCID for the author

---

## Reviewer 3 — Hydrology / Environmental Applications Specialist

**Expertise:** Hydrological modeling, flood risk assessment, environmental GIS applications
**Confidence:** High
**Recommendation:** Minor Revision

### Summary

The manuscript presents a comprehensive geospatial library with a strong algorithmic foundation. The hydrology module (3 depression handling methods, 4 flow direction algorithms, multiple derived indices) is well-designed, and the D-infinity validation against TauDEM is rigorous. However, the environmental case study lacks depth for an Environmental Modelling & Software paper, and several hydrological algorithms are listed but never validated. The paper reads more as a software description than as a contribution to environmental modeling methodology.

### Major Issues

**M1. Environmental case study is a demonstration, not an application**

EMS explicitly states that methodological developments "should be illustrated with applications in the environmental fields" and that insights should "contribute to the store of knowledge." The Andean case study (Section 4) computes eight terrain derivatives and presents them as maps, but does not:

- Apply them to solve an environmental problem
- Compare results against field observations or existing maps
- Demonstrate integration with environmental models (e.g., landslide susceptibility, species distribution, soil erosion)
- Quantify any environmental outcome

The flood susceptibility mapping (Figure 7) uses HAND thresholds from Nobre et al. (2011) — calibrated for Amazonian floodplains — applied to high-altitude Andean terrain at 3000-6000m elevation. The paper itself acknowledges these are "illustrative" (line 249). This is insufficient for EMS.

**Required action:** Either:
(a) Develop the flood susceptibility or landslide susceptibility into a proper case study with appropriate thresholds and some form of validation (even qualitative, e.g., comparison with known landslide inventories), or
(b) Reframe the entire section as "Illustrative workflow" rather than "Environmental case study" and explicitly state in the discussion that environmental application validation is deferred to future work. Option (b) is acceptable if the performance and accuracy contributions are strong enough to stand alone.

**M2. Multiple hydrological algorithms listed but never validated**

Table 2 lists algorithms that are never mentioned again:
- **TFGA flow accumulation**: listed in Table 2, never benchmarked, validated, or discussed
- **MFD flow accumulation**: benchmarked (Table S6 mentions GRASS uses modified D8) but no direct MFD-to-MFD comparison
- **Planchon-Darboux fill**: listed but only Priority-Flood is benchmarked
- **Watershed delineation**: Table 2 footnote says "implemented but not benchmarked"
- **Stream order (Strahler)**: listed in Supplementary Table S1 but never mentioned in text

For a paper that claims "127 algorithms," the validation coverage is thin. The credibility of the algorithm count is undermined when many algorithms are listed without any demonstration of correctness.

**Required action:** Either validate the key missing algorithms (at minimum: MFD vs a reference implementation, Planchon-Darboux vs Priority-Flood output equivalence) or explicitly acknowledge in the discussion which algorithm categories have been validated and which remain to be verified. A table summarizing validation status per algorithm category would be transparent and appropriate.

### Minor Issues

**m1. D-infinity "RMSE = 0.0000°" — precision and reporting**

Line 454 reports RMSE = 0.0000° for D-infinity comparison with TauDEM. Reporting a metric as exactly zero is unusual and invites skepticism. Please report to appropriate precision (e.g., RMSE < 10⁻⁴°) or provide the actual floating-point value. Also clarify: is this exact bitwise agreement, or agreement to some tolerance? If exact, that is a remarkable claim that should be stated more precisely.

**m2. TWI validation is incomplete**

The TWI cross-tool comparison (RMSE=2.72, r=0.44) compares SurtGIS D8-based TWI against GRASS MFD-based TWI. This does not validate SurtGIS's TWI implementation — it demonstrates a known methodological difference. For validation, the authors need a D8-based TWI from another tool. WhiteboxTools provides D8 flow routing; was D8-TWI compared across tools?

**m3. Breach algorithm: timing without validation**

The breach algorithm (Lindsay 2016) is timed (29.8s) but never validated for correctness. The paper notes (line 454) that "WhiteboxTools's breach implementation differs in parameterization (constrained vs unconstrained)" — but this prevents any cross-tool verification. At minimum, confirm that the breach output satisfies the basic correctness criterion: all cells in the output can drain to the edge via downslope routing.

**m4. HAND sensitivity to stream threshold**

The paper correctly notes HAND sensitivity to threshold (line 249), but the range tested (50-200 cells) is narrow. In practice, stream thresholds vary by orders of magnitude depending on application (10 cells for detailed networks to 10,000+ for major rivers). A brief sensitivity figure in supplementary would demonstrate this point more effectively than text.

**m5. Geographic CRS limitation is underplayed**

Line 91 buries a significant limitation: "Geographic CRS inputs (degrees) are currently handled by computing derivatives in degree units." This means that if a user loads a typical global DEM (e.g., SRTM in WGS84), all slope, curvature, and flow computations will produce physically meaningless results. This should be prominently flagged, not buried in a parenthetical. Many potential users will encounter this as their first experience with the tool.

**m6. The "127 algorithms" count is inflated**

Reviewing Supplementary Table S1:
- Band math binary ops (#86) and band math expressions (#87) are the same operation with different interfaces
- Generic normalized difference (#85) subsumes all individual spectral indices (#69-84)
- Multiple spectral indices (NDVI, NDWI, MNDWI, NDRE, GNDVI, NGRDI, RECI, SAVI, MSAVI, EVI, EVI2, NBR, NDSI, NDBI, NDMI, BSI) are trivial ratio computations — listing each as a separate "algorithm" inflates the count
- Morphological operations (erosion, dilation, opening, closing, gradient, top-hat, black-hat) are standard image processing primitives, not geospatial algorithms

A more honest framing would be "127 functions" or "127 operations" rather than "127 algorithms," or alternatively state "50+ distinct algorithms covering terrain, hydrology, imagery, and spatial statistics" which more accurately reflects the intellectual contribution.

### Editorial Issues

- The hydrology pipeline description (Section 2.3) is well-organized and the table is clear
- The D-infinity validation paragraph (line 454) is exemplary in its precision — this level of detail should be the standard for all validation claims
- Figure 6 (hydro validation) is well-designed and informative

---

## Associate Editor Decision

**Manuscript:** ENVSOFT-D-26-XXXXX
**Decision:** Minor Revision

### Summary of Reviews

All three reviewers find the manuscript technically sound and recommend Minor Revision. The key issues cluster into four themes:

### Theme 1: WhiteboxTools Slope Comparison (Critical — all reviewers)
The paper claims speedup over WBT for slope computation, but the paper itself demonstrates that WBT uses a fundamentally different algorithm (RMSE = 6.53°). **All speedup claims involving WBT slope must be removed from the abstract, highlights, comparison section, and discussion.** The benchmark data may remain in tables with clear caveats. This is the single most important revision.

**Impact on paper:** The abstract currently leads with the 11.2× WBT claim. After removal, the headline numbers become: 1.7-1.8× over GDAL (slope), 2.0× over GDAL (aspect), 4.5-4.9× over GRASS, and 7.5-23.1× for hydrology. These are still compelling and, importantly, methodologically valid.

### Theme 2: Abstract and Length (R1-M2, R2-M2)
The abstract exceeds the 150-word EMS limit and must be shortened. The manuscript at 48 pages remains long for EMS; target ≤40 pages through table consolidation and case study condensation.

### Theme 3: Reproducibility and Archival (R2-M1)
A version-pinned Zenodo DOI (or equivalent) is required for EMS Option C compliance. This is standard practice and straightforward to implement.

### Theme 4: Environmental Application Depth (R3-M1)
The case study reads as a software demonstration rather than an environmental modeling application. At minimum, reframe as "Illustrative workflow" rather than "Environmental case study" if a proper application is not feasible within the revision timeline.

### Required Changes

| # | Issue | Source | Severity | Action |
|---|-------|--------|----------|--------|
| 1 | Remove WBT slope from all speedup claims | R1-M1 | **Critical** | Remove from abstract, highlights, Sec 6, discussion |
| 2 | Abstract ≤150 words | R1-M2 | **Required** | Shorten abstract |
| 3 | Zenodo DOI for reproducibility | R2-M1 | **Required** | Create archival release with DOI |
| 4 | Manuscript length ≤40 pages | R2-M2 | **Expected** | Consolidate tables, condense case study |
| 5 | Reframe case study or strengthen application | R3-M1 | **Expected** | Reframe as "Illustrative workflow" or add validation |
| 6 | Acknowledge unvalidated algorithms | R3-M2 | **Expected** | Add validation status summary |
| 7 | D-inf RMSE precision | R3-m1 | Minor | Report to appropriate precision |
| 8 | Confidence intervals on speedups | R2-m1 | Minor | Add IQR-based ranges |
| 9 | Curvature exclusion threshold justification | R1-m3 | Minor | Add justification or sensitivity |
| 10 | Geographic CRS warning prominence | R3-m5 | Minor | Make limitation more visible |
| 11 | "127 algorithms" framing | R3-m6 | Minor | Consider "127 operations/functions" |
| 12 | Thread scaling: fix or downgrade | R2-m3 | Minor | Remove Amdahl fit or re-run with n≥10 |

### Strengths to Preserve

- The curvature validation (Gaussian hill + anisotropic sinusoidal ridge, 14/14 R²=1.000000) is exemplary
- The D-infinity bug fix and TauDEM validation is a genuine contribution to the community
- Benchmark methodology is transparent (full-pipeline, I/O breakdown, uncompressed caveat)
- The cross-platform architecture (one codebase → native/WASM/Python) is genuinely novel
- The software availability section is comprehensive and well-structured
- The AI declaration is appropriately detailed

### Timeline

The required revisions are addressable within **4 weeks**. The critical WBT slope issue requires text changes only (no new experiments). The Zenodo DOI is a mechanical step. The manuscript shortening requires editorial work but no new content.

---

## Progress Tracker

| Category | Count | Items |
|----------|-------|-------|
| Critical | 1 | WBT slope claims |
| Required | 2 | Abstract length, Zenodo DOI |
| Expected | 3 | Manuscript length, case study framing, algorithm validation acknowledgment |
| Minor | 6 | D-inf precision, CIs, curvature threshold, CRS warning, algorithm count, thread scaling |
| **Total** | **12** | |
