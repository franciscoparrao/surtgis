# Architecture

This page explains _why_ SurtGIS is structured the way it is. If you want to
learn what flags to pass, you're in the wrong section — try
[CLI reference](../reference/cli/index.md). If you want to know
_why_ `--streaming` exists at all, keep reading.

## Workspace layout

SurtGIS is a Cargo workspace of ten crates:

```
surtgis/
├── crates/core           # Raster<T>, GeoTransform, CRS, GeoTIFF I/O, streaming, mosaic, vector
├── crates/algorithms     # 250+ public functions across 10 modules
├── crates/parallel       # Rayon-based strategies, maybe_rayon compile-time switch
├── crates/cli            # 19 top-level subcommands, binary `surtgis`
├── crates/cloud          # COG reader, STAC client, bbox reprojection, s3:// normalisation
├── crates/wasm           # WebAssembly bindings (~17% of algorithms currently)
├── crates/python         # PyO3 bindings: extract_at_points, predict_raster
├── crates/gui            # egui desktop app (experimental)
└── crates/colormap       # Color schemes and raster → PNG rendering
```

The split exists because the dependency graphs differ wildly. `surtgis-core`
has zero non-trivial deps beyond `tiff` and `ndarray`; `surtgis-cloud` pulls
in `reqwest` + `tokio` for HTTP; `surtgis-python` needs PyO3 at build time;
`surtgis-gui` is egui and eframe. Jamming them into one crate would force
every user — including someone who just wants `Raster<f32>` — to compile a
GUI framework they don't need.

Concretely: embedding SurtGIS in a Rust program that only needs the
algorithm library pulls **two crates** (`surtgis-core`, `surtgis-algorithms`).
Adding STAC support adds a third. Nothing more.

## The `Raster<T>` core

Everything starts from one type:

```rust
pub struct Raster<T> {
    data: ndarray::Array2<T>,
    transform: GeoTransform,
    crs: Option<CRS>,
    nodata: Option<f64>,
}
```

Generic over the element type (`f32`, `f64`, `i16`, `u8`, `u16`), with an
affine `GeoTransform` (origin + pixel size + optional rotation) and an
optional CRS. Nodata is stored separately from the data array, and the
convention across the library is: for float element types, nodata values are
replaced by `NaN` on read, and `NaN` is the only signal algorithms check for
invalid data. No sentinel values buried inside floats.

This is deliberately more restrictive than GDAL's model, which supports
arbitrary block organisations, band counts, and offsets per band. SurtGIS
treats a multi-band raster as a `Vec<Raster<T>>` — one per band — and forces
each band through its own file on disk. The cost: you pay one HTTP round-trip
per band when reading from COGs. The benefit: the entire library's worth of
algorithms doesn't have to special-case multi-band layouts.

## Native I/O via the `tiff` crate

A deliberate non-decision: SurtGIS does not depend on GDAL by default. It
reads and writes GeoTIFFs using the pure-Rust [`tiff`](https://crates.io/crates/tiff)
crate plus a thin layer in `surtgis-core::io::native` that handles the
geospatial tags (`ModelTiepointTag`, `ModelPixelScaleTag`, `GeoKeyDirectory`,
`GDAL_NODATA`).

Trade-offs of this choice:

**Wins:**
- Single binary. Ship as precompiled. No `libgdal.so` version mismatch hell.
- Works on WASM. Most of the algorithm library compiles to
  `wasm32-unknown-unknown` because the I/O isn't platform-gated on GDAL.
- Cloud-optimised reads. The COG reader in `surtgis-cloud` issues HTTP range
  requests directly against the TIFF byte layout; no GDAL VSI layer in
  between.

**Losses:**
- Exotic formats: HDF4, JP2, ERDAS .img, etc. GDAL reads ~150 formats;
  SurtGIS natively reads GeoTIFF + (optionally) Zarr + NetCDF + GRIB2.
  If you need Pléiades JP2000s in, GDAL is the answer.
- TIFF edge cases that GDAL's robustness has absorbed over two decades: odd
  compression combos, non-standard predictor tags, malformed GeoKey
  directories. SurtGIS handles the common cases (predictor 1/2/3, DEFLATE,
  LZW, big/little endian, tiled and stripped), but not every ancient file
  your grandfather's theodolite produced.

A `gdal` feature flag exists as an escape hatch — `cargo install surtgis --features gdal`
— which falls back to the GDAL crate for I/O when the native path can't
handle something. Most users never need it.

## Streaming: strip processing for arbitrarily large rasters

The `StripProcessor` trait in `surtgis-core::streaming` is where SurtGIS
handles DEMs larger than RAM. About ten window-based algorithms (Horn slope,
aspect, hillshade, curvature, Gaussian smoothing, Laplacian, and a few more)
are written against this trait rather than the in-memory `Raster<T>`.

A strip is a horizontal band of rows read into memory together with enough
overlap above and below (the _halo_) to compute window-based operations
without cross-strip coordination. For a 3×3 Horn slope, halo = 1; for
Gaussian `sigma=5`, halo = ceil(3σ) ≈ 15.

```
+--- DEM on disk ---+
|                   |
|  strip 1 [halo]   |   ← read N rows + halo below
|  strip 1 [data]   |   ← compute, write
|  strip 1 [halo]   |
|                   |
|  strip 2 [halo]   |   ← reuse bottom halo of strip 1
|  strip 2 [data]   |
|  strip 2 [halo]   |
|  ...              |
+-------------------+
```

Peak RAM = `2 × halo × cols × sizeof(T)` plus one strip's worth of output.
For a 100,000 × 100,000 DEM at f32 with Horn slope, that's roughly 0.8 MB
working set regardless of total file size.

Automatic detection: if a raster on disk would exceed `--max-memory` (or
500 MB by default) when decompressed, algorithms that support streaming
switch automatically. `--streaming` forces the path; `--max-memory 16G`
raises the threshold; passing `--streaming` for an algorithm that doesn't
support it is a hard error rather than silent fallback.

The list of streamable algorithms is small on purpose — adding one requires
writing the algorithm against the halo-strip iterator pattern, which is
harder than writing against a fully-resident array. We only absorb that cost
for algorithms people actually want to run on huge DEMs.

## Parallelism: `maybe_rayon`

The `surtgis-parallel` crate provides a compile-time switch:

```rust
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(not(feature = "parallel"))]
use std::iter as rayon_fallback;
```

With the `parallel` default feature, algorithm loops use `par_iter` /
`par_bridge`. Without it (useful for WASM, where threading is a separate
setup), the same code compiles to sequential iterators. One source, two
targets.

The WASM build is currently serial because browser thread pools require
SharedArrayBuffer and CORS headers that most hosts don't provide by default.
Adding WASM threading is tracked but not a priority — the per-tile work in
a typical browser use case (small extents) doesn't benefit enough to justify
the deployment complexity.

## STAC + COG: the cloud read path

`surtgis-cloud` handles the cloud-native read path, which is structurally
different from "download the whole file then process it":

1. STAC search returns a list of items, each with asset hrefs (HTTPS or
   `s3://`) pointing at COG files on blob storage.
2. `CogReader::open(href)` fetches the first ~64 KiB — enough to parse the
   TIFF header, IFD chain, and GeoKey directory — without reading tile data.
3. `reader.read_bbox(bbox)` computes which internal tiles of the COG
   intersect the requested bbox, fetches exactly those tiles via HTTP range
   requests, decompresses them in parallel, and returns a `Raster<T>` of the
   intersection.

The key insight: for a 100 MB COG where your bbox only covers 1% of the
image, you transfer ~1 MB and decompress ~1 MB. The `stac composite`
pipeline composes this with strip processing so even very wide STAC extents
have bounded peak memory.

See [STAC integration philosophy](stac-philosophy.md) for why the design
chose this path over the alternatives (download-then-process, or
GDAL-VSI-based access).

## Why workspaces pin internal versions loosely

The ten crates share a version via `[workspace.package]` and reference each
other with `version = "0.7"` (caret range) plus `path = "../core"`. Local
builds always use the path; crates.io resolution uses the version. This
means:

- Bumping any one crate breaks the build unless they all bump together.
- Cutting a release = one version bump in the root `Cargo.toml`, one
  `cargo publish` per crate in dependency order.
- `cargo update` picks up patch releases automatically; a minor bump
  (v0.7 → v0.8) requires consumers to update their dependency declaration.

This is heavier than publishing one monolithic crate, but it lets someone
embed `surtgis-core` in a low-footprint context without dragging in the CLI
or the cloud stack.
