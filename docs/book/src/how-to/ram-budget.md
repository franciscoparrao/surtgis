# Tune the RAM budget

For workflows that approach or exceed your machine's RAM, SurtGIS gives
you two dials.

## Override the budget target (stac composite)

`stac composite` auto-sizes `strip_rows` to fit within a 16 GB budget by
default. Lower or raise it with an environment variable:

```bash
# Conservative — fit the pipeline in 8 GB
SURTGIS_RAM_BUDGET_GB=8 surtgis stac composite ...

# Permissive — machine has 64 GB, let strips be larger for fewer HTTP calls
SURTGIS_RAM_BUDGET_GB=32 surtgis stac composite ...
```

SurtGIS will print the budget and the derived `strip_rows` on startup:

```text
RAM budget (16.0 GB target, band_chunk_size=1) — output: 3.3 GB | mask cache: 5.0 GB | scene strips: 1.3 GB | band working: 0.4 GB | decode: 0.1 GB (strip_rows=512) → ~10.1 GB peak
```

That `~10.1 GB peak` is the estimate. Real peak is typically within ±30%.
If it comes in much higher, you've hit an edge case — see
[Debug a stac composite using too much RAM](debug-stac-ram.md).

## Override the memory ceiling (general commands)

For commands that operate on in-memory rasters:

```bash
surtgis --max-memory 4G terrain slope huge.tif slope.tif
```

Any raster larger than `--max-memory` (default 500 MB) triggers automatic
streaming mode on commands that support it. Force it explicitly with
`--streaming`:

```bash
surtgis --streaming terrain slope huge.tif slope.tif
```

If you pass `--streaming` on a command that hasn't been adapted to the
streaming path, you get a hard error. See [Memory model](../explanation/memory-model.md)
for the list of streamable algorithms.

## Rule of thumb for sizing

On a host with `X` GB of RAM, leave about `0.4 × X` GB free for the OS
and other processes. So:

| Host RAM | Safe `SURTGIS_RAM_BUDGET_GB` |
|---|---|
| 8 GB | 4 |
| 16 GB | 9 |
| 32 GB | 18 |
| 38 GB (postdoc machine) | 22 |
| 64 GB | 36 |

These aren't hard caps — SurtGIS won't strictly respect the budget, it
sizes `strip_rows` to aim for it. Overshoot is possible. When the
overshoot matters (you're on a shared machine, or the OOM killer is
watching), pair the budget override with an external watchdog:

```bash
# Kill if RSS exceeds 14 GB
( surtgis stac composite ... & PID=$!
  while kill -0 $PID 2>/dev/null; do
    RSS=$(awk '/VmRSS:/ {print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    if [ "$RSS" -gt 14000000 ]; then kill -9 $PID; break; fi
    sleep 30
  done
)
```

## Raising `band_chunk_size` for speed at RAM cost

The `--band-chunk-size K` flag on `stac composite` controls how many bands
are downloaded and processed together per scene. K=1 (default) is
minimum RAM; higher K reduces HTTP request count at proportional RAM
cost:

| K | Scene strips | HTTP requests per scene |
|---|---|---|
| 1 (default) | 1× baseline | n_bands × |
| 3 | 3× baseline | n_bands/3 × |
| 5 | 5× baseline | n_bands/5 × |

On a 38 GB host with a 10-band composite, K=5 is comfortable. The budget
printed at startup scales with K explicitly, so you can see the predicted
peak before committing.
