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

## Survey 2026-06-06 — post-P4, looking beyond rayshader frame

After closing the **P4 quadtree LOD** sprint (v0.14.0 + v0.14.1),
the "rayshader peer" narrative is essentially complete: native 16 M-
vertex DEMs at 60 FPS, browser 4 K under the 256 MB WebGL2 cap, CLI
headless `--lod` for 10 K DEMs. The only remaining rayshader-wins
axis is path-tracing — discussed separately below.

This survey asked: **what would be most useful to incorporate from
adjacent tools beyond rayshader?** Three candidates surfaced. The
agreed decision: **document and defer**, no commit yet.

### Candidates surveyed

#### G. Martini RTIN terrain mesher — *backlog*

[Martini](https://github.com/mapbox/martini) replaces the uniform-
chunk LOD subdivision with Right-Triangulated Irregular Network
adaptive triangulation. Same triangle budget, much better
distribution: places density where curvature is high, saves on flat
regions.

**Where it fits**: `crates/relief-3d/src/lod.rs` already has the
`ChunkLod` interface — RTIN drops in as an alternative
`QuadtreeMesh::from_dem_rtin(dem, params)` builder. The rest of the
LOD pool + render path stays identical.

**Estimate**: ~3 days.

**Why defer**: P4's quadtree LOD already meets every acceptance bar.
RTIN is a quality upgrade, not a blocker. Pick it up when a specific
paper figure has large flat regions that the quadtree wastes
triangles on.

#### H. TopoToolbox-style channel network depth — *partially done*

[TopoToolbox](https://topotoolbox.wordpress.com/) (MATLAB) is the
gold standard for tectonic geomorphology paper figures: χ analysis,
ksn maps, longitudinal profiles, swath profiles, knickpoint
detection, drainage divide migration. Equivalent stack in the
LSDTopoTools ecosystem.

**What's already shipped**:
- `fill-sinks` (priority flood + Planchon-Darboux, with NaN-curtain
  fix from Smugglers Notch).
- Flow direction (D8 + Dinf with TauDEM-exact agreement, RMSE=0°).
- Flow accumulation.
- Stream extraction.
- χ + ksn at usable depth (see Smugglers Notch validation, R²≥0.7
  on dominant basin; 15-cuencas Chile production validated).
- Validation framework (`examples/smugglers_notch_validation/`,
  `examples/15_cuencas_chile/`).

**What is likely undercooked or missing** (to verify before any
follow-up sprint):
- Bayesian regression for ksn with uncertainty quantification
  (LSDTopoTools-style).
- Swath profile extraction with statistics envelope (min/max/median
  bands across a corridor).
- Knickpoint detection (Wobus et al. 2006 slope-area break method;
  Neely et al. 2017 KZP variant).
- Drainage divide migration metrics (Whipple et al. 2017 χ-anomaly
  approach).
- Stream-profile χ-plot figure rendering (long profiles ready, just
  no dedicated figure generator).

**Estimate**: depends entirely on which gaps survive verification.
Likely 1-2 sprints to reach LSDTopoTools-style parity on the items
above.

**Why partially done**: the core analysis is already shipped and
research-validated. The remaining items are paper-figure-shaped, not
core. Worth picking up when a specific paper demands them (tectonic
geomorphology submission, etc.) rather than speculatively.

#### I. 3D Tiles export — *backlog*

[3D Tiles](https://www.ogc.org/standards/3d-tiles) is the OGC
standard for streaming 3D geospatial. Exporting tilesets from DEMs
lets users open SurtGIS output in CesiumJS, ArcGIS, Bentley, etc.

**Where it fits**: new crate `surtgis-3dtiles` or extension of
`surtgis-relief-3d`. The P4 quadtree maps naturally to the 3D Tiles
tile hierarchy.

**Estimate**: ~5-7 days.

**Why defer**: real demand from users with Cesium pipelines, but no
current research paper is gated on it. Practical interop story, not
a differentiator.

### Surveyed and ruled out

#### J. Path-tracing — *out of scope*

The last named rayshader-wins axis. Discussed at length 2026-06-06,
decision was to skip:

- Browser path-tracing infeasible (WebGL2 has no compute shaders;
  ray queries need Vulkan KHR / Metal / DX12).
- CPU MVP would be ~5-7 days but produces a slice of `rayrender`
  quality at fraction of the polish.
- Real use case (cinematic geospatial posters) doesn't drive current
  research needs.
- Better answer in the README: "for path-traced renders use
  rayrender or Blender; SurtGIS aims at analysis + interactive 3D,
  not cinematics."

If revisited, the working name is `SPEC_SURTGIS_RELIEF_P5.md` with
MVP scope: Rust + rayon, CPU only, triangle BVH on DEM mesh, primary
rays + Lambertian + soft shadow (sun disk) + sky dome, 100-500 spp,
acceptance at 1024×768 Imhof-style cinematic in ~5-30 min wallclock.

#### K. Volume rendering (ParaView-style) — *out of scope*

Scalar volume viz for atmospheric / subsurface data. Niche outside
current user base.

#### L. Photogrammetry / LAS-LAZ pipelines — *out of scope*

Point cloud reading + rasterization. Well-served by PDAL and
OpenDroneMap; no value in re-implementing.

#### M. Browser interactive analysis (measure, draw, query) — *out of scope*

Large UI investment with marginal research-figure value. The web
viewer's job is "show me the surface"; analysis happens in the CLI /
library.

#### N. Cinematic camera path animation — *backlog, low priority*

Animated DEMs with scripted camera paths. Possible add-on to the
viewer. Not committed; depends on whether real demand materializes.

### Next reassessment trigger

Re-survey when one of the following hits:

- A specific paper figure needs RTIN-quality flat regions → pick up G.
- A specific paper submission needs Bayesian ksn / knickpoints /
  divide migration → re-scope H and pick up the missing pieces.
- A user with a Cesium pipeline asks for 3D Tiles export → pick up I.
- Sprint slot opens with no pulling external priority → re-survey
  from scratch (the landscape changes).
