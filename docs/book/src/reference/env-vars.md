# Environment variables

SurtGIS reads the following environment variables at runtime. Most
configuration is via CLI flags; env vars are for settings that are
awkward to thread through every command.

## `SURTGIS_RAM_BUDGET_GB`

**Consumed by:** `stac composite`

**Default:** `16`

Sets the target peak-RAM budget for the STAC composite pipeline. The
command auto-sizes its `strip_rows` to fit within this budget, accounting
for output buffers, per-scene mask cache, active-band scene strips, band
working set, and decode overhead.

```bash
SURTGIS_RAM_BUDGET_GB=8 surtgis stac composite ...   # conservative
SURTGIS_RAM_BUDGET_GB=24 surtgis stac composite ...  # permissive
```

Real peak is typically within ±30% of the target. See
[Tune the RAM budget](../how-to/ram-budget.md) for sizing guidance.

## `XDG_CACHE_HOME`

**Consumed by:** `stac`, `cog` (when `--cache` is enabled)

**Default:** `$HOME/.cache` (Linux), `$HOME/Library/Caches` (macOS)

COG tile cache is stored under `$XDG_CACHE_HOME/surtgis/cog/`. Override
to put the cache on a different disk:

```bash
export XDG_CACHE_HOME=/mnt/fast-ssd/cache
```

## `HOME`

Used as a fallback for `XDG_CACHE_HOME` when the latter is unset. Also
consulted for conventional per-user config paths.

## `CARGO_TERM_COLOR`, `NO_COLOR`, `TERM`

Standard CLI-colour env vars. SurtGIS honours `NO_COLOR=1` to disable
ANSI escapes; useful for CI logs.

## `RUST_LOG`

Controls the `tracing` subscriber's verbosity when you pass `--verbose`.

```bash
RUST_LOG=debug surtgis --verbose terrain slope dem.tif slope.tif
RUST_LOG=surtgis_cloud=trace surtgis --verbose stac composite ...
```

Module-scoped levels (`surtgis_cloud=trace`) are useful for debugging
cloud read issues without drowning in unrelated debug output.

## Variables SurtGIS does **not** consume

Common ones we explicitly don't use:

- `GDAL_DATA`, `PROJ_LIB` — SurtGIS's reprojection uses `proj4rs` which
  embeds its own grid data, not the system GDAL/PROJ installation.
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — not used for
  cop-dem-glo-30 via `s3://` (the bucket is public). For private S3
  buckets, authenticated access is not yet supported.
- `PC_SDK_SUBSCRIPTION_KEY` — Planetary Computer's private-API key. Not
  consumed; SurtGIS uses the public anonymous endpoint.
