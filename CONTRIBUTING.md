# Contributing to SurtGIS

Bug reports, pull requests, and discussion welcome. Below is the
minimum process plus one project-specific rule that has earned its
place the hard way.

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo test --workspace --features cloud  # STAC + COG reader tests
```

`cargo fmt --all` and `cargo clippy --workspace --all-targets` must
pass — both are gated in CI (`Format` and `Check & Clippy` jobs).

## Pull request flow

1. Branch off `main` (`fix/<slug>` or `feat/<slug>`).
2. Keep the diff scoped — algorithm fixes go separate from refactors.
3. Open a PR. CI runs format, clippy, unit tests, integration tests,
   Python build, WASM build, web demo. All must be green.
4. After approval, squash-merge or merge (the repo accepts both).
5. For algorithm or CLI changes, add a CHANGELOG.md entry under
   `[Unreleased]` so the release manager can lift it during the next
   version bump.

## Releases

`Cargo.toml` workspace `version` is the single source of truth — all
crates inherit it. Bump it, tag `vX.Y.Z` on the merge commit, push the
tag. The tag push triggers `.github/workflows/release.yml` (binaries
for Linux / macOS / Windows) and `python-wheels.yml` (PyPI). The
6 crates publish manually to crates.io in this dependency order:

```
surtgis-core → surtgis-colormap → surtgis-parallel
              → surtgis-algorithms → surtgis-cloud → surtgis
```

## The real-data validation rule

**New algorithms (and bug fixes that touch I/O, projections, or
external-data adapters) must be validated against at least one real
dataset, end-to-end, before merge.** Synthetic unit tests are not
sufficient. This rule exists because the project keeps shipping bugs
that pass every unit test and only surface when the algorithm meets
real data:

| Bug | Surfaced by | Why unit tests missed it |
|---|---|---|
| `extract-patches` vector reprojection (#87) | `examples/maule_mini/` with Earth Search + UTM raster + WGS84 GeoJSON | Tests used pre-aligned synthetic coordinates; the CRS-mismatch path had no coverage |
| `stream-network` handler recomputed pipeline | Sprint 3 ksn smoke test on real DEM | CLI handler bug, not in the algorithm; unit tests went through the library API and never hit the handler path |
| TIFF predictor=3 decoder (cop-dem-glo-30) | Pipeline on real cop-dem-glo-30 STAC asset | The COG fixture had predictor=2; no floating-point predictor test |
| GeoJSON CRS in fluvial outputs (v0.10.1) | Postdoc loading fluvial outputs with geopandas | Tests validated GeoJSON schema, not RFC 7946 / WGS84 semantics |
| `fill-sinks` NaN-barrier (v0.10.2) | Sprint 7 Smugglers Notch validation against P&R 2013 | Synthetic DEMs were always fully populated; the NaN-curtain-after-reprojection path had no coverage |

Each of these bugs would have shipped to users (and in several cases
did ship) if the project had relied on unit tests alone. The fix
pattern in every case was small and obvious in hindsight, but the
*discovery* required end-to-end execution on data the algorithm had
never seen.

### Practical recipe for real-data validation

Add a self-contained directory under `examples/<name>/` with:

- `run_validation.sh` — a `set -euo pipefail` script that fetches
  open-data inputs (Earth Search, Microsoft Planetary Computer,
  OpenTopography), runs the pipeline end-to-end, and exits non-zero
  on numeric failure.
- `plot_validation.py` or `*.R` — produces a headline figure that
  another scientist could compare against a published reference.
- `README.md` — what the dataset is, what the expected result is,
  what known caveats apply.
- A reference: cite the paper or canonical implementation you are
  reproducing (TopoToolbox, TauDEM, WhiteboxTools, GDAL, etc.).

Templates live in `examples/smugglers_notch_validation/` (single
canonical-paper reproduction) and `examples/maule_mini/` (mixed-CRS
GFM preprocessing).

The example does not need to run in CI — many of them require GBs
of remote data and several minutes of wall time. It needs to be
**reproducible** by a maintainer at release time, and its results
need to be committed alongside the code so that drift is detectable
in PR diffs.

When opening a PR that adds or modifies an algorithm, link the
relevant validation example in the PR description (or add one if
none exists).

## Style

- Spanish or English in commit messages — both are fine.
- No emoji in code, docs, or commits.
- Comments only where the *why* is non-obvious; trust the code to
  explain *what* it does.
- Default to terse. The CHANGELOG, not the comment block, is where
  release-grade narrative lives.
