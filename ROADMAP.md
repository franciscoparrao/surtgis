# SurtGIS Roadmap

Post EMS-revision (2026-05-15+) strategic roadmap. Captures the
agreed-upon axes of work and the rationale for the chosen ordering.

## Strategic framing

### What SurtGIS does better than anyone (real differentiators)

- **WASM in-browser execution** — no other comprehensive terrain library
  compiles to WebAssembly. Reviewer 2 of the EMS paper specifically liked
  this angle.
- **Single binary, no GDAL dependency** — 22 MB static-linked CLI vs
  GDAL's 315 MB and 197 transitive Debian packages.
- **Florinsky's complete 14-curvature system** — the only open-source
  implementation.
- **Rust performance + memory safety** — predictable latency, no GIL,
  no GC pauses.

### What SurtGIS should not try to be

- A competitor to `stackstac + dask` for industrial STAC composite
  workloads on managed cloud hubs. That race is won; the right answer
  for users with that workload is to use stackstac.
- A replacement for the `xarray + numpy + dask` Python ecosystem.
- A horizontally scaled cluster runtime (no K8s, no MPI). Single-node
  + cloud-native is the design point (see EMS Section 2.2).

## Axes of work

Each axis is sized in person-days; treat as rough order-of-magnitude.
Priority is **my recommendation as of 2026-05-15**; user has final
say.

### A. Expand the WASM differentiator (strategic, paper-aligned)

**State**: 33 of 127 algorithms exposed in WebAssembly bindings (paper §5.3).

**Scope**:
- Port remaining terrain algorithms to WASM, especially the Florinsky
  14 curvatures (paper's headline contribution; reviewer 2 explicitly
  liked this).
- Port hydrology basics (D8 flow direction, flow accumulation) — clear
  pedagogical demos.
- Improve the live demo page with usage scenarios (uploaded-DEM slope,
  uploaded-NIR/red NDVI, etc.).

**Size**: 2-3 weeks.

**Impact**:
- Reinforces the paper's novel-deployment narrative for any R2 review.
- Opens uses in teaching, edge computing, offline PWAs that nothing
  else can serve.

### B. Close the standalone workflow gap (operational)

**State**: SurtGIS depends on `gdalwarp` for reprojection. The mdBook
how-to guides explicitly tell users to fall back to it. This breaks the
"no GDAL dependency" claim for the full pipeline.

**Scope**:
- Native `surtgis reproject input.tif output.tif --to EPSG:32719`
  command.
- `proj4rs` is already a dependency; missing pieces are the CLI wrapper
  and an interpolation kernel (nearest / bilinear / cubic).

**Size**: 3-5 days.

**Impact**: Closes the last operational gap; the auto-sufficiency
claim becomes complete.

### C. Investigate even/odd RAM peak pattern (postdoc-driven)

**State**: `BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md` item #6. Even-numbered
strips (2, 4) reach +2.5 GB transient peak vs odd strips (1, 3)
floor. Pattern reproducible across at least two pairs.

**Hypotheses** (in order of likelihood):

1. **mimalloc page-retention cycle**: allocator retains pages at end of
   strip, reuses them in next strip before returning to OS. Drop of
   -3.4 GB in Phase A of strip 3 is suggestive.
2. **`scene_masks` double-tenancy**: the explicit `drop(scene_masks)`
   added in v0.7.1 happens at end of strip body, but the next strip
   starts allocating before the drop fully releases to the OS heap.
3. **Tile cache cross-strip**: COG tile cache may not invalidate
   correctly between strips, accumulating until the tile set changes.

**Scope**:
- Read code in `crates/cli/src/handlers/stac.rs` around the strip loop
  and tile caching.
- Reproduce with a small synthetic STAC bench (5 strips on a tiny bbox).
- v0.7.1 `phase A teardown` log lines should make the issue visible.
- Fix: likely `mimalloc::collect(true)` or `malloc_trim` at end of strip,
  or restructure drop order, or explicit cache invalidation.

**Size**: 1-2 days investigation + 1 day fix.

**Impact**: ~20% peak RAM reduction (13 GB → ~10 GB on Maule). Direct
benefit for the postdoc's 15-cuencas pipeline.

### D. Defensive paper-revision work (anticipatory)

**State**: EMS revision submitted 2026-05-15. If R1 (the most demanding
reviewer) pushes back on R2 review, the likely targets are:

- Hybrid-arch caveat in §5.4 — could re-run on a homogeneous-core
  cloud VM (AWS c7i.4xlarge / Hetzner ax-series) to remove the caveat.
- Hillshade SIMD optimization (paper acknowledged GDAL wins here at
  large scale).

**Size**: 1-2 weeks total.

**Impact**: Only relevant if R1 returns with the same concern in R2.
If R1 accepts the current responses, this work is unnecessary.

### E. Python binding completeness (incremental)

**State**: 97 of 127 algorithms exposed in Python (was 56 at v0.1.1,
significantly improved through v0.6.x). Remaining ~30 are niche terrain
+ advanced interpolation.

**Scope**:
- Add PyO3 wrappers + unit tests for the 30 missing algorithms.

**Size**: 3-5 days.

**Impact**: Completeness, not a differentiator. No specific user
asking for this today.

### F. New capabilities (ambitious, future papers)

Each is potentially its own paper:

- **DEM-from-stereo** photogrammetry pipeline.
- **Image registration** (cross-correlation, phase correlation).
- **`extract-patches` v2** with augmentation flags for CNN training.
- **GPU acceleration** via WebGPU for cross-platform compute.

**Size**: 4-8 weeks per item.

**Impact**: New papers possible. Only worth starting once one of A-C
is fully done.

## Agreed priority (2026-05-15)

User chose: **C → B → (then revisit)**.

Rationale:

- **C first**: real bug, active user (postdoc) waiting, ~3 days work.
  Direct measurable impact.
- **B second**: closes the standalone workflow gap, ~3-5 days.
  Strengthens the "no GDAL" claim for any future paper/talk.

### Progress

- **C** ✓ done in v0.7.3 (2026-05-16, commit 7a2371f).
  `mi_collect(true)` at strip end eliminates the even/odd peak pattern.
  Validated with `crates/cli/examples/test_mi_collect.rs`. Expected
  postdoc impact: Maule peak 13 GB → 10 GB.
- **B** ✓ done in v0.7.4 (2026-05-16). Native `surtgis reproject`
  command using proj4rs + Rayon-parallelised inverse-mapping.
  Supports nearest/bilinear, any EPSG→EPSG. Smoke tested UTM 19S →
  WGS84 (0.13 s) and UTM 19S → Web Mercator (0.59 s) on a 1000×1000 DEM.

### Next reassessment

- Reassess based on EMS editor response. If revision accepted clean,
  pivot to **A** (WASM expansion) as the strategic differentiator.
- If R1 pushes back, **D** becomes mandatory before further axes.

Defer indefinitely (until external demand): E, F.

## Tracking

Updates to this file as axes start/close. Each axis should produce
either:
- A version bump (v0.7.x for C, B; v0.8.0+ for A) with CHANGELOG entry.
- A summary commit linking back to this roadmap.

## Out-of-scope reminders (write down so we don't drift)

- Do not attempt parity with stackstac+dask for cloud-managed STAC
  workloads. Cite EMS §2.2 positioning.
- Do not add MPI / distributed runtime. Cite EMS §2.2 positioning.
- Do not promise "drop-in GDAL replacement". Cite EMS §6 service-
  publication framing.
