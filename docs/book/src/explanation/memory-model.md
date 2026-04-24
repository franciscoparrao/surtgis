# Memory model

SurtGIS has two distinct modes for handling raster data, and understanding
when each applies is the difference between a pipeline that finishes and
one that OOMs.

## In-memory mode

The default. A raster is read entirely into an `ndarray::Array2<T>`,
algorithms operate on the array in place, and the result is written back
to disk. This is what every command does unless a streaming condition is
met.

Peak RAM ≈ input size + output size, both fully decompressed. For a
`20,000 × 20,000` f32 DEM that's `20k² × 4 bytes = 1.6 GB` twice, so
about 3.2 GB working set. Most commands fit comfortably in modern host
RAM.

## Streaming mode (strip processing)

For about 10 window-based algorithms (Horn slope, aspect, hillshade,
curvature, Gaussian smoothing, Laplacian, fill-sinks, flow direction,
and a few more), SurtGIS can process the DEM in horizontal strips with
configurable overlap, reading and writing one strip at a time.

Peak RAM is independent of total raster size: roughly
`2 × halo × cols × sizeof(T)` plus one strip of output. For a
`100,000 × 100,000` f32 DEM with Horn slope (halo = 1), that's about
800 KB of data actively in memory at any moment, not 40 GB.

Triggered automatically when the raster on disk would exceed
`--max-memory` (default 500 MB). Force explicitly with `--streaming`.

## STAC composite mode

The `stac composite` pipeline is neither: it operates on a stream of
scenes fetched from STAC, mosaics them into per-strip outputs, and writes
results per band. Peak RAM is bounded by a 5-component model documented
in the command's startup log line and discussed in the
[debug-stac-ram how-to](../how-to/debug-stac-ram.md).

## Practical decision tree

```
Am I running a window-based algorithm on a DEM?
├── Yes, raster fits in RAM → in-memory mode, no action needed
├── Yes, raster exceeds --max-memory → streaming kicks in automatically
└── No, I'm running STAC composite → 5-component budget model applies,
                                      see --band-chunk-size and
                                      SURTGIS_RAM_BUDGET_GB
```

## Which algorithms support streaming?

Roughly the ones where the operation is definable in a bounded window:
`terrain slope`, `terrain aspect`, `terrain hillshade`, `terrain
curvature`, `terrain tpi`, `terrain tri`, `terrain gaussian-smoothing`,
`hydrology fill`, `hydrology flow-direction`, `morphology dilation`,
`morphology erosion`.

Algorithms that need global information (flow accumulation across an
entire drainage network, watershed delineation from any-point pour,
mosaic of unrelated rasters) don't fit the strip pattern and run
in-memory. For huge DEMs these would need a different approach
(Dask-style tile graphs, or out-of-core sort/scan algorithms). Not
currently implemented.
