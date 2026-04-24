# When to use SurtGIS vs alternatives

SurtGIS is a newer, narrower tool than the incumbents in this space. This
page is the honest "should I bother?" guide. It will not tell you SurtGIS is
better at everything — it isn't.

## The incumbents

| Tool | What it's great at |
|---|---|
| **GDAL** | I/O breadth (~150 formats), raster transforms, the reference implementation of nearly every GIS operation |
| **GRASS GIS** | Hydrology, mature topological algorithms, serious spatial analysis workflows with long history |
| **WhiteboxTools** | Hydrology specifically, very clean algorithm implementations, good defaults |
| **SAGA GIS** | Terrain analysis, geomorphometry, wide algorithm selection with a visual UI |
| **QGIS** | User interface, visualisation, integrating all of the above |

All four are freely available, battle-tested, and vastly more mature than
SurtGIS. If one of them already solves your problem, use it.

## Where SurtGIS is a genuine win

### You want a single-binary CLI with no system dependencies

No `libgdal.so` version matching. No Python environment. Works in Docker
`FROM scratch`. Works in CI runners without privileged installs. This
matters for:

- Reproducible benchmarks.
- Embedding geospatial processing into systems that can't afford a package
  manager (industrial control, embedded, browser via WASM).
- Distributing analysis scripts that a non-GIS user can actually run. The
  friction from "install Anaconda then conda-forge GDAL then pip pyproj" to
  "download this tarball" is substantial.

### You're processing rasters larger than your RAM

The streaming strip processor (see [Memory model](memory-model.md)) runs
slope, aspect, hillshade, curvature, fill-sinks, flow direction, Gaussian
smoothing, and ~10 others with bounded RAM regardless of input size. GDAL
and WhiteboxTools can also do this, but typically require explicit block
processing setup. SurtGIS does it automatically when `--max-memory` would
be exceeded.

### You want end-to-end STAC workflows with no glue code

`surtgis stac composite --catalog pc --asset red,nir,swir16,swir22 ...`
searches Planetary Computer, downloads only the tiles intersecting your
bbox via HTTP range requests, cloud-masks with SCL, and median-composites
across dates. One command. Peer tools (rasterio + pystac-client +
rioxarray + dask) can match this but typically require 30+ lines of Python
glue and an understanding of Dask.

### You need terrain + hydrology on the same binary

Many projects combine DEM preprocessing with spectral analysis. Mixing
WhiteboxTools for hydrology, GDAL for I/O, and a Python notebook for NDVI
is normal but operationally painful. SurtGIS covers all three in one tool,
which matters most for scripts that run daily and need to be debuggable
years later.

### You want honest, reproducible benchmarks against GDAL/GRASS

The [paper](https://github.com/franciscoparrao/surtgis/tree/main/paper)
cross-validates SurtGIS outputs against GDAL 3.11, GRASS 8.3, and
WhiteboxTools on identical DEMs, and publishes the tolerances as tables.
RMSE vs GDAL for slope is 1.6×10⁻⁴ degrees. That level of validation is a
design choice and is part of the CI today.

## Where the incumbents beat SurtGIS

### You need a format SurtGIS doesn't read natively

Netcdf, GRIB2, and Zarr are supported but gated behind feature flags that
require system libraries on some platforms. HDF4, JP2, ECW, MrSID, ERDAS
.img: not supported at all. Reach for GDAL.

### You need mature, edge-case-hardened algorithms

GRASS's hydrology has been tuned for 25 years against flat terrain, coastal
areas, pit-fill strategies, and every weird DEM edge case you've ever seen.
SurtGIS implements the _standard_ algorithms (D8, D∞, priority-flood fill,
watershed-from-pour-points) and validates them against GRASS on common
cases, but any production pipeline that hits edge cases will hit them
first on SurtGIS, because GRASS has already been debugged on them.

### You need a GUI

`surtgis-gui` exists but is experimental, has no real user base, and is
not the project's priority. QGIS is a dramatically better interactive
environment. SurtGIS integrates cleanly as a QGIS algorithm provider if
you want the CLI-backed pipeline with the QGIS UX.

### You need Python interoperability beyond a handful of functions

`surtgis-python` exposes `extract_at_points` and `predict_raster` for ML
pipelines. That's it. If you're building on numpy/xarray/rasterio and
want the full toolbox, rasterio + rioxarray + pyproj + scikit-image is the
mature stack. SurtGIS is for Rust-first users who occasionally want to
bridge to Python.

### You need a wider algorithm surface than 250 functions

That's a lot, but GRASS has ~500 modules and GDAL has ~100 high-level
operations on top of its I/O. If your workflow uses obscure morphological
filters, multi-criteria decision analysis, or mature geostatistical tools
beyond IDW/kriging/natural-neighbour, the incumbents have more.

## A decision rule

A blunt three-question filter:

1. **Is your data in GeoTIFF (+ maybe Zarr/NetCDF/GRIB2 with feature
   flags)?** If not, stop here and use GDAL.
2. **Do you want your pipeline to be one binary that installs in 10
   seconds and runs anywhere?** If not, GRASS + WhiteboxTools give you more
   at the cost of install complexity.
3. **Are you writing a Rust program that needs geospatial primitives
   embedded?** Here SurtGIS has no real competition — `gdal-rust` bindings
   exist but the FFI surface and build requirements are hostile.

If you answer yes–yes–maybe, SurtGIS is a strong fit. Otherwise you're
probably better served by the incumbents, and we'd rather you use them and
come back to SurtGIS when a concrete gap bites.

## Where we're honestly worse right now

- **Docs.** This book is the first attempt at a real user guide; the
  incumbents have decades of accumulated tutorials, stackoverflow
  answers, and textbook chapters.
- **Community.** No Slack / mailing list / forum. One postdoc in
  production; no GitHub stars to speak of.
- **Mature error messages.** GDAL has tuned its error text over two
  decades; SurtGIS still emits some Rust-ish errors that won't mean much
  to a GIS user.
- **Edge-case robustness.** Every weird TIFF file in the world has been
  thrown at GDAL. SurtGIS has been thrown at a few thousand.

These are not bugs; they are the cost of being young. If you hit one,
please [report it](https://github.com/franciscoparrao/surtgis/issues) —
that's the only way the list gets shorter.
