# Post-mortem: STAC composite RAM blow-up on Earth Search

**Affected:** `stac composite` multi-band on Earth Search, v0.6.19 – v0.6.23
**Resolved in:** v0.6.24 (structural) + v0.6.25 (consolidation)
**Reporter:** postdoc session processing 15 Chilean watersheds
**Date:** April 2026

## What went wrong

A postdoc processing Sentinel-2 composites for Maule (7274×5725 px, 10 bands, 42 dates) on Earth Search froze a 38 GB RAM host on v0.6.19. Over four patch releases we reduced the peak RAM but never matched the budget we promised. Final numbers:

| Version | Budget reported | Real peak | Error |
|---------|----------------:|----------:|------:|
| v0.6.19 | ~4.3 GB         | >30 GB    | >7×   |
| v0.6.22 | 12.9 GB         | 20 GB     | +55%  |
| v0.6.23 | 11.9 GB         | 18 GB     | +53%  |
| v0.6.24 | 10 GB           | ~5 GB     | −50% (conservative) |

## The path we took

### v0.6.19 → v0.6.22: "the constant was wrong"

Initial auto-cap modelled only the scene accumulator (`strip_rows × n_bands × n_scenes × 8`). Observed peak on Earth Search was 7× worse because the accumulator wasn't the dominant term. We added a `download_reserve_bytes` constant (6 GB for ES) and chunked tile downloads to bound in-flight decoded tiles. Peak dropped to 20 GB.

### v0.6.22 → v0.6.23: "the model is wrong-but-fixable"

Budget still off by +55%. Hypothesis: the per-scene `per_band_tiles` cache (decoded f64 rasters held until each band's mosaic consumed them) wasn't modelled. We added a `per_scene_cache` term that scaled linearly with `strip_rows`, calibrated `per_scene_inflation = 130` from the v0.6.19 data point. Peak: 18 GB at `strip_rows=83`. Still +53% over budget.

### v0.6.23 → v0.6.24: "the model is wrong-structurally"

Two empirical points:
- v0.6.22: S=127, real peak 20 GB → residual (excluding output + accumulator) ≈ 13.6 GB
- v0.6.23: S=83, real peak 18 GB → residual ≈ 12.4 GB

If the residual really scaled with S, reducing S from 127 to 83 (35% cut) should have cut residual ~35%. It cut 9%. **The residual is near-constant in S.**

What's near-constant in S? The per-scene tile cache is dominated by `n_bands × n_outer_tiles × decoded_tile_size`, where `decoded_tile_size` is driven by COG internal tile alignment (1024² on ES) and cross-UTM reprojection bloat — both independent of strip height. For Maule that's ~10 × 45 × 30 MB ≈ 13 GB regardless of S.

The only way to eliminate this was architectural: **don't hold all 10 bands' tiles simultaneously**. We flipped the loop nesting from scene-outer/band-inner to band-outer/scene-inner, pre-computing cloud masks once per scene and sharing them across bands. Peak dropped to ~5 GB real.

## Lessons

1. **Two empirical points contradicting a linear model means the model is wrong, not the constants.** The temptation after v0.6.22 was to re-calibrate `per_scene_inflation`. We did — and it failed the same way. At that point the correct response is "our equation has the wrong shape", not "our equation has the wrong coefficient."

2. **Instrumentation is the highest-ROI thing you can ship during debugging.** From v0.6.22 we printed the budget breakdown on every run: `output | mask cache | scene strips | band working | decode → peak`. That's what made the +55% error legible. Without it we'd have kept shipping patches blind.

3. **Structural refactors have obvious tradeoffs; name them up front.** The band-outer flip multiplies HTTP requests by `n_bands / band_chunk_size`. We accepted it and shipped `--band-chunk-size` so users on higher-RAM hosts can trade memory back for fewer requests. Hiding the tradeoff would have been worse than making the user choose.

4. **Fast feedback loop is a prerequisite, not a luxury.** Each iteration took hours because (a) the reporter printed exact peak RSS vs budget, (b) we had a publishing pipeline that cleared in minutes, (c) we didn't block on design reviews for what were clearly code-complete proposals. Five releases in four days, each with a real-user validation step.

## What we built to prevent recurrence

- **Budget breakdown stays in output** (v0.6.22+). Non-optional diagnostic; trivially parseable.
- **RAM regression benchmark** (v0.6.25): nightly integration test on a small ES bbox, asserts peak RSS stays within budget.
- **`--band-chunk-size` flag** (v0.6.25): exposes the RAM↔HTTP dial as a first-class knob, default 1 (current conservative default).
- **Exponential backoff + jitter** (v0.6.25): the band-outer refactor multiplied HTTP requests by `n_bands`; this cushions the extra rate-limit pressure on ES.

## What we'd do differently

If this recurs in another codepath (it will — every pipeline with multi-asset I/O has the same failure mode), skip v0.6.22/23 entirely. When the budget mismatch is >40% on the first data point, jump straight to "which per-pixel multiplier am I holding across the full scene that I could eliminate by restructuring?" — that's almost always cheaper than three rounds of analytical calibration.
