# Tectonic geomorphology with the fluvial module

The five algorithms in `surtgis fluvial` answer different questions
about a river network. This chapter explains *which* algorithm to
reach for *when*, gives the briefest necessary geomorphological
context for a developer who doesn't have it, and documents the
parameter tuning that the spec calls out as easy to get wrong.

For the runnable end-to-end sequence, see
[Fluvial-tectonic morphometry from a DEM](../tutorials/fluvial-from-dem.md).
For the formal references, see the spec at
`docs/SPEC_morfometria_fluvial_tectonica.md` in the repo.

## The framework in one paragraph

The stream-power law (Howard 1994, Whipple & Tucker 1999) postulates
that a river's incision rate scales as `E = K · A^m · S^n` with
`A` the drainage area, `S` the local slope, and `K` an erodibility
constant determined by lithology and climate. In topographic steady
state — incision balanced by tectonic uplift — this inverts to a
slope-area scaling `S = ks · A^(-θ)` with `θ = m/n`. Fixing θ at a
reference value (0.45 is the convention for bedrock channels) and
solving for `ks` gives the normalised channel steepness `ksn`, which
is comparable across basins. Departures from this clean picture
— knickpoints, anomalous concavity, divide asymmetry — flag transient
landscape response or non-uniform erodibility.

That's the framework. The five algorithms each measure a different
facet of it.

## When to use what

| Question | Algorithm | Output |
|---|---|---|
| "Is this basin in steady state? Where's the base level?" | **chi** | Per-cell χ raster |
| "Which channels are climbing fastest? Where's the high U/K?" | **ksn** | Per-cell raster + per-segment vector |
| "Are there transient pulses propagating upstream?" | **knickpoints** | Vector points with polarity |
| "Is θ for this basin actually 0.45?" | **concavity** | Per-basin θ with bootstrap CI |
| "Are basin boundaries migrating?" | **divide-migration** | Vector lines with Δχ |

In the order they're typically computed: chi → ksn → knickpoints in
parallel; concavity once basins are defined; divide-migration last
because it consumes the χ raster from the first step.

## chi (χ)

The path integral of `(A₀/A(x))^θref` along the channel from a base
level upward. Geometric: it's a base-level-anchored reference distance
that climbs faster where drainage area is small (i.e. headwaters
"count more"). Plotted as elevation vs χ, a steady-state profile is a
straight line whose slope equals `ksn`.

**When χ alone is enough**: discriminating divide asymmetry, comparing
sub-basins, checking whether the network has reached topographic
steady state (look for the elevation~χ curve being roughly linear).

**Key parameter**: `--theta-ref` (default 0.45). Almost never worth
changing for inter-basin comparison; the literature standard exists
because comparing across studies requires it. If you want a
basin-specific θ, run `concavity` instead.

## ksn (channel steepness)

ksn = `S · A^θref` per cell, smoothed along the channel over a window
(default 500 m). High ksn = either high uplift or resistant bedrock or
both.

**When ksn is the right primary metric**: you have a basin or a region
and want to map relative U/K spatially. ksn is the workhorse — every
tectonic-geomorphology paper has a ksn map.

**Tuning that bites**:
- `--segment-length-m` (default 500 m): the smoothing window. With a
  10–30 m DEM, 500 m is the literature standard. With a 90 m DEM,
  bump to 1000 m or accept noise.
- `--min-drainage-area-m2` (default 1 km²): cells below this give
  unstable ksn (small numerator, noisy slope). Don't lower without
  a reason.
- `--theta-ref`: must match what you'd compare against in the
  literature. Don't tune this per dataset unless you're explicit.

ksn requires channel-following slope, **not** the 2-D terrain slope
output of `surtgis terrain slope`. The fluvial module computes the
correct version internally; if you find yourself reaching for
`terrain slope` to feed an analysis, stop — you've gone off the
geomorphology rails.

## knickpoints

Sharp breaks in the slope of a river's long profile. A knickpoint can
be a **transient** wave-form response to a tectonic perturbation
propagating upstream, or a **stationary** feature anchored to a
lithologic contrast. SurtGIS classifies each detection by polarity:

- **Convex** (slope increases downstream) — most often **transient**,
  flag for tectonic interpretation.
- **Concave** (slope decreases downstream) — most often
  **lithology-pinned**, expect a corresponding contact on the geology
  map.

The method (Neely et al. 2017) total-variation-denoises the χ-z
profile, computes the discrete second derivative `d²z/dχ²`, and flags
cells where the magnitude exceeds a threshold AND the elevation drop
across a small window exceeds `min_magnitude_m`. Cells within 5 of a
segment end are excluded (default) because confluences and outlets
induce spurious curvature.

**Tuning that bites**:
- `--tvd-lambda` (default 0.5): smoothing strength. Too low → every
  noise spike is a knickpoint. Too high → real knickpoints smoothed
  away. Default tuned for 10–30 m DEMs.
- `--min-magnitude-m` (default 10): the elevation-drop guard. Raise
  to 30+ if you're seeing dozens of false positives in pristine
  headwaters; the real knickpoints are usually >> 30 m.
- `--curvature-threshold` (default 1.0): controls how sharp a break
  has to be. Most users never touch this; tune `tvd_lambda` and
  `min_magnitude_m` first.

## concavity (θ per basin)

Grid-searches θ over [0.1, 0.9] for each basin, picking the value
that linearises the elevation~χ regression best (minimum RMSE), then
bootstraps for a 95 % CI.

**When**: you want to test whether assuming θ = 0.45 is reasonable
for your basins, OR you want θ per basin as its own measurement (some
geomorphologists treat it as a tectonic signal independent of ksn).

**What the values mean**:
- θ ∈ [0.4, 0.6]: textbook bedrock channel in steady state. Boring,
  confirmatory.
- θ < 0.3 or > 0.7: interesting. Could indicate transient response
  (Mudd et al. 2018), non-uniform erodibility, or a non-uniform
  uplift pattern. Investigate.
- Bootstrap CI that spans 0.2 or more: the basin has too few channels
  or too much scatter to pin down θ. Don't over-interpret.

**Tuning**:
- `--bootstrap-n` (default 200): more iterations = tighter CI, more
  compute. 200 is enough for most basins; 1000 if you need
  publication-grade error bars.
- `--min-basin-cells` (default 30): below this the estimator is
  unreliable.

## divide-migration

For every pair of adjacent basins, compute the median asymmetry
across their shared divide: Δχ (Willett 2014), Δelev (Gilbert
metric), Δrelief.

**When**: you suspect basin geometries are not stable. Active
tectonic uplift, base-level fall, or differential erosion all push
divides toward the "losing" basin. A sign in Δχ across a long divide
is the canonical detection.

**Interpretation**:
- Δχ ≈ 0 along the divide: basins in (chi) equilibrium, no migration
  signal.
- Consistent Δχ > 0 with basin_a (smaller numeric ID by convention)
  on the higher-elevation side: basin_a is being "captured" by
  basin_b. The basin with lower χ at the divide is winning.
- The Gilbert Δelev / Δrelief metric is the older indicator — useful
  as cross-check but Δχ is the modern primary.

**Tuning**:
- `--min-divide-length-m` (default 500): suppress divides too short
  to be statistically meaningful. Raise to 2000+ for regional
  studies.
- `--chi` (optional but recommended): provide the χ raster from
  `surtgis fluvial chi`. Without it, you only get the Gilbert
  metrics — usable but weaker.

## Pitfalls the spec calls out

These bit the spec author often enough that they're in the spec's
§8. They will bite you too.

1. **Cell size > 30 m**: ksn and knickpoints are sensitive to DEM
   resolution. With FabDEM at 25 m you're fine; with SRTM at 90 m
   you'll get noisier results that bias toward false positives.
   `--cell-size-m N` overrides the auto-detection; the algorithms
   warn when cell > 30 m.

2. **Stream threshold matters**: too low and convergent noise from
   zero-order hollows contaminates the analysis; too high and you
   miss real tributaries. 1000 cells at 30 m (≈ 0.9 km²) is a
   sensible default; for arid regions go higher, for humid regions
   lower.

3. **Short tributaries are unreliable**: by default the algorithms
   filter cells with drainage area below 1 km² or segments with
   fewer than ~20 cells. Don't relax these without a reason.

4. **NoData propagates upstream**: a single missing cell in the DEM
   knocks out χ for everything upstream. Fill or interpolate
   carefully.

5. **Boundary outlets**: outlets at the raster edge get partial
   trajectories. The algorithms flag and exclude them from medians;
   if your AOI is dominated by boundary outlets, expand the bbox.

6. **θ may not be unique**: the RMSE landscape can be flat. The
   bootstrap CI is honest about this — read it.

7. **TVD λ at the wrong scale**: see the knickpoints tuning section
   above.

8. **CRS must be projected**: every fluvial algorithm assumes
   metres. EPSG:4326 inputs are rejected with a clear error;
   reproject to UTM (or any local projected CRS) first.

9. **Confluence noise in knickpoints**: confluences cause sudden
   area discontinuities that look like knickpoints to the
   curvature detector. The default 5-cell buffer at segment ends
   filters this; widen it if you still see clustering at junctions.

## When SurtGIS isn't the right tool

For interactive stream-network pruning (drawing on a map, removing
specific channels), use TopoToolbox 2 (MATLAB) or pyTopoToolbox.
SurtGIS is batch-only; the interactive use case isn't its sweet spot.

For landscape evolution modelling (TTLEM / Landlab),
likewise — those are a different category of tool and out of scope
for the spec.

For prospectivity mapping or other geological-AI workflows, the
SurtGIS [GFM preprocessing pipeline](../how-to/gfm-prithvi-prep.md)
is the right entry point; fluvial outputs can be inputs to it but
shouldn't be its terminus.

## References

- Howard (1994) WRR 30(7) — detachment-limited stream-power law
- Whipple & Tucker (1999) JGR — stream-power model dynamics
- Wobus et al. (2006) GSA SP 398 — `ksn` operational paper
- Perron & Royden (2013) ESPL 38, 570 — χ-transform fundacional
- Willett et al. (2014) Science 343, 1248765 — divide reorganisation
- Neely et al. (2017) EPSL 469 — knickpoint detection method
- Whipple, Forte et al. (2017) JGR-Earth Surface 122 — divide migration
- Mudd et al. (2018) ESurf 6 — variable θ as signal
- Schwanghart & Scherler (2014) ESurf 2 — TopoToolbox 2 (the
  reference implementation in MATLAB; SurtGIS targets parity with
  its main outputs)
