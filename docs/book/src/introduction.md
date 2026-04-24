# SurtGIS

**High-performance geospatial analysis in Rust.** A single CLI binary and a
set of Rust crates that compute 250+ terrain, hydrology, imagery, morphology,
interpolation, and classification algorithms — with native GeoTIFF I/O,
streaming for rasters larger than RAM, and an end-to-end STAC composite
pipeline for Planetary Computer and Earth Search.

This book is the user guide. For the auto-generated API reference (for
embedding SurtGIS as a library in Rust code), see
[docs.rs/surtgis-core](https://docs.rs/surtgis-core).

## What this book is for

If you're looking for:

- **"Does this fit my workflow?"** → start with [Installation](installation.md)
  then the [First terrain analysis](tutorials/first-terrain-analysis.md)
  tutorial. 30 minutes and you'll know.
- **"How do I do X?"** → [How-to guides](how-to/reproject-utms.md) are
  task-focused and short. Pick the one that matches your problem.
- **"What flags does `surtgis terrain slope` take?"** → [CLI reference](reference/cli/index.md).
- **"Why does SurtGIS do it this way?"** → [Explanation](explanation/architecture.md)
  covers architecture, memory model, and side-by-side comparisons with
  GDAL, GRASS, and WhiteboxTools.

The four sections don't overlap by design. Tutorials teach. How-to guides
solve. Reference is authoritative. Explanation reasons. This is the
[Diátaxis](https://diataxis.fr/) framework.

## Design principles

1. **Native I/O, no GDAL required.** GeoTIFF read/write including
   predictor 1/2/3, DEFLATE, GDAL_NODATA, and Cloud Optimized (COG) over HTTP
   range requests. GDAL is available as an optional feature.

2. **Streaming over in-memory when data grows.** Algorithms that fit the
   strip-processing pattern (Horn slope, Gaussian smoothing, flow direction,
   fill-sinks, and about ten more) run with bounded RAM regardless of raster
   size. Detection is automatic; a `--streaming` flag forces the path when
   you want to be sure.

3. **One binary per platform.** The CLI is statically-linked where possible.
   The default feature set for the precompiled binary covers STAC + COG +
   climate data (Zarr) + UTM reprojection. System-library dependencies
   (libnetcdf, HDF5) are opt-in via `--all-features` when building from
   source.

4. **Validated against the canonical alternatives.** Slope, aspect,
   hillshade, flow accumulation, and watershed outputs agree with GDAL 3.11,
   GRASS 8.3, and WhiteboxTools within documented tolerances. The
   cross-validation numbers live in the paper (Environmental Modelling &
   Software, under review).

5. **Honest about tradeoffs.** SurtGIS is not faster than everything at
   everything — see [When to use SurtGIS vs alternatives](explanation/vs-alternatives.md)
   for where the Rust reimplementation wins (I/O pipelines, parallel
   terrain factors, streaming) and where you're better off reaching for
   GDAL or GRASS.

## Current status

As of v0.7.0, SurtGIS is in production use on a 15-watershed Chilean
landslide-susceptibility pipeline (Sentinel-2 composites, 41 M-cell areas,
60 labelled wells). No external users are known yet — feedback from people
outside the original developer/postdoc loop is exactly what would take this
from "works for one expert" to "works for a community". If that's you,
please [open an issue](https://github.com/franciscoparrao/surtgis/issues).
