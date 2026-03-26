# SurtGIS EMS Paper — Revision Checklist v4

**Source:** Blind peer review simulation v4 (3 reviewers + AE)
**Date:** February 2026
**Decision:** Minor Revision (10 items, 2 excluded)

**Excluded:** #4 (manuscript length — not reducing further), #3 (Zenodo DOI — GitHub is the archive)

---

## Critical

### [x] C1: Remove WBT slope from all speedup claims (R1-M1)
**Issue:** WBT slope uses Florinsky 2016 5×5 method (RMSE=6.53° vs Horn). Claiming speedup over a different computation is misleading.
**Resolution:** Removed WBT slope speedup ratios from abstract, highlights, Tables 1-2, Section 6, and Discussion. WBT times remain in tables for completeness with footnote explaining different derivative method. Aspect and hydrology WBT comparisons preserved (valid).
**Locations fixed:**
- [x] Abstract: removed "8.9--11.2× over WhiteboxTools", restructured to slope/aspect/hydrology
- [x] Highlights: replaced "up to 11.2× over WBT" with "23× for flow accumulation vs GRASS"
- [x] Table 1: removed speedup parentheses from WBT column, added footnote $^{b}$
- [x] Table 2: removed speedup from WBT slope rows, added footnote $^{b}$
- [x] Section 6: replaced WBT slope claim with WBT aspect claim + explicit exclusion note
- [x] Discussion: changed "4.2--11.2×" to "3.2--7.9× for aspect and hillshade" + exclusion note

---

## Required

### [x] R1: Abstract ≤150 words (R1-M2)
**Issue:** Current abstract ~165 words, EMS requires ≤150.
**Resolution:** Shortened to ~119 words by removing WBT slope claims, tightening language, removing last sentence.

---

## Expected

### [x] E1: Reframe case study (R3-M1)
**Issue:** Section 4 reads as software demo, not environmental application.
**Resolution:** Changed title to "Application example: Andean terrain characterization". Changed "demonstrate" to "illustrate". Added closing sentence deferring environmental application validation to domain-specific studies. Updated section reference in paper structure paragraph.

### [x] E2: Acknowledge unvalidated algorithms (R3-M2)
**Issue:** TFGA, MFD, Planchon-Darboux, watershed, Strahler order listed but never validated.
**Resolution:** Added "Validation coverage" paragraph in Discussion listing which algorithms are validated and which remain to be verified. Mentions MFD, TFGA, Planchon-Darboux, watershed delineation, Strahler ordering as not cross-compared against external references.

---

## Minor

### [x] m1: D-infinity RMSE precision (R3-m1)
**Issue:** "RMSE = 0.0000°" looks suspiciously exact.
**Resolution:** Changed to "RMSE $< 10^{-4}$°" and "near-exact agreement" instead of "exact agreement".

### [x] m2: Confidence intervals on speedups (R2-m1)
**Issue:** No uncertainty ranges on speedup ratios.
**Resolution:** Added sentence in methodology: "the headline speedup ratios (e.g., slope vs GDAL 1.8×) are stable within ±0.1–0.2× across the IQR."

### [x] m3: Curvature exclusion threshold justification (R1-m3)
**Issue:** Bottom 10% gradient exclusion appears arbitrary.
**Resolution:** Added inline justification: gradient-dependent curvatures contain ||∇z|| in denominators and become numerically singular as ||∇z|| → 0; results stable for thresholds 5%–20%.

### [x] m4: Geographic CRS warning prominence (R3-m5)
**Issue:** Limitation buried in parenthetical (line 91).
**Resolution:** Changed to bold "**Important:**" prefix, made wording more explicit about producing "physically meaningless" results, added directive to reproject to UTM.

### [x] m5: "127 algorithms" framing (R3-m6)
**Issue:** Some entries are trivial variations (spectral indices, morphological ops).
**Resolution:** Changed "127 algorithms" to "127 distinct computational operations" in Introduction. Supplementary Table S1 already has counting methodology paragraph.

### [x] m6: Thread scaling — remove Amdahl fit or add caveat (R2-m3)
**Issue:** n=3 is too small for Amdahl's law fitting.
**Resolution:** Reworded: "As a rough estimate with n=3, the data suggest p ≈ 0.55... though these estimates should be considered qualitative given the small sample size."

---

## Progress

| Category | Total | Done | Remaining |
|----------|-------|------|-----------|
| Critical | 1 | 1 | 0 |
| Required | 1 | 1 | 0 |
| Expected | 2 | 2 | 0 |
| Minor | 6 | 6 | 0 |
| **TOTAL** | **10** | **10** | **0** |

### ALL ITEMS COMPLETE
