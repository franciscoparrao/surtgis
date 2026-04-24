# Debug a `stac composite` using too much RAM

If `surtgis stac composite` is consuming more RAM than expected, work
through these in order.

## 1. Read the budget line

Every `stac composite` run prints at startup something like:

```text
RAM budget (16.0 GB target, band_chunk_size=1) — output: 3.3 GB | mask cache: 5.0 GB | scene strips: 1.3 GB | band working: 0.4 GB | decode: 0.1 GB (strip_rows=512) → ~10.1 GB peak
```

Five components:

| Component | What it is | Scales with |
|---|---|---|
| output | Pre-allocated output buffers, one per band | n_bands × total_cells |
| mask cache | Mosaicked cloud masks, one per scene | n_scenes × strip_rows |
| scene strips | Per-scene data for the active band chunk | band_chunk_size × n_scenes × strip_rows |
| band working | The active band's mosaicked tile cache | band_chunk_size × strip_rows |
| decode | In-flight tile decode buffers | tile_concurrency |

If `~X GB peak` is comfortably within `--max-memory` (or your host's free
RAM), proceed; if not, see step 2.

## 2. Watch the `[ram]` transition log lines

Since v0.6.26, SurtGIS prints RSS and cumulative-tile counters at
strip/phase/band-chunk transitions:

```text
[ram] baseline before strip loop: RSS=120 MB
[ram] strip 1/2 start: RSS=3300 MB, tiles_cumulative=0
[ram] strip 1/2 after Phase A (masks): RSS=5800 MB, phase_a_tiles=1890, cumulative=1890
[ram] strip 1/2 chunk bands [0..1] start: RSS=6200 MB, cumulative=1890
[ram] strip 1/2 chunk bands [0..1] end: RSS=6400 MB, cumulative=3780
```

Rule of thumb: if RSS grows between a `chunk start` and the matching
`chunk end` by more than the budget's `scene strips + band working`, the
model is wrong for this workload — [file an issue](https://github.com/franciscoparrao/surtgis/issues)
with the full log. That's exactly how the fragmentation bug in v0.6.24
was found.

## 3. Lower the budget, then the bbox

In order of impact:

**Cut `SURTGIS_RAM_BUDGET_GB`.** Halves `strip_rows`, proportionally
halves `mask cache` and `scene strips`:

```bash
SURTGIS_RAM_BUDGET_GB=8 surtgis stac composite ...
```

**Cut `--band-chunk-size`.** K=1 is the minimum; you can't go lower.
If you're already at 1 and still over budget, the workload is structurally
too big for the host.

**Cut `--max-scenes`.** Each scene you drop trims `mask cache` and
`scene strips` proportionally. Going from 42 scenes to 20 halves both.
Median composite quality degrades, but on a cloud-free month that can be
acceptable.

**Cut the bbox.** Halving the bbox area quarters `output` and
proportionally shrinks every other component. If the bbox is a watershed,
process by sub-watershed and mosaic the outputs afterwards.

## 4. If RSS grows linearly with no plateau

That's the fragmentation signature from the v0.6.19–v0.6.24 debugging
arc. Should not recur on v0.6.26+ (mimalloc fixed it). If it does on a
later version:

- Confirm you're on v0.6.26 or later: `surtgis --version`.
- Share the `[ram]` log plus `surtgis --version` plus OS + allocator
  in an issue.

## 5. Checklist for a stuck run

Before giving up:

- [ ] Current version ≥ v0.6.26?
- [ ] `--cache` enabled if re-running the same bbox?
- [ ] `SURTGIS_RAM_BUDGET_GB` set to ~0.6× host RAM?
- [ ] `--band-chunk-size 1`?
- [ ] bbox reasonable (< ~200 km × 200 km for S2 L2A at 10 m)?
- [ ] Host not competing with another RAM-heavy process?

If all yes and it still OOMs, that's a genuine bug, not a tuning problem.
