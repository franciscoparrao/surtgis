# Using surtgis-core as a library

`surtgis-core` is the foundation crate of the SurtGIS ecosystem: raster
types, georeferencing, native GeoTIFF I/O, streaming, and tiling — with
a deliberately small dependency footprint and **no GDAL requirement**.
Sibling engines (data cubes, embeddings, InSAR, hydrology models) build
on it instead of reimplementing I/O.

As of v0.15.0 the public API surface of `surtgis-core` is treated as
**stable under SemVer**: breaking changes are pre-announced one release
ahead in the CHANGELOG under a `Breaking` heading.

## Adding the dependency

```toml
[dependencies]
surtgis-core = "0.15"
```

That single line pulls types + I/O + streaming. The default build is
pure Rust (the `tiff` crate handles GeoTIFF); the optional features
are:

| Feature | Effect |
|---|---|
| `parallel` (default) | Rayon-backed `ndarray` operations |
| `gdal` | GDAL-based I/O as an alternative backend |
| `shapefile` | Shapefile vector reading |
| `geopackage` | GeoPackage vector reading (rusqlite) |
| `parquet` | GeoParquet point-table I/O (`PointTable`) |
| `complex` | `Raster<Complex<f32>>` cells + InSAR helpers (`magnitude`, `phase`, re/im split) |

You do **not** need `surtgis-algorithms`, `surtgis-cli` or any other
crate to consume rasters. Add `surtgis-algorithms = "0.15"` only when
you want the analysis functions (terrain factors, hydrology, sampling).

## The canonical pattern

Read a GeoTIFF into a `Raster<f64>`, process it, write the result.
Metadata (transform, CRS, nodata) rides along:

```rust
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::Raster;

fn main() -> surtgis_core::Result<()> {
    // None = no explicit nodata override; NaN is the float convention
    let dem: Raster<f64> = read_geotiff("dem.tif", None)?;

    let (rows, cols) = dem.shape();
    let mut out = dem.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));

    for row in 0..rows {
        for col in 0..cols {
            let z = dem.get(row, col)?;
            out.set(row, col, if z.is_finite() { z * 3.28084 } else { f64::NAN })?;
        }
    }

    write_geotiff(&out, "dem_feet.tif", None)?;
    Ok(())
}
```

Key types:

- **`Raster<T>`** — the exchange unit of the whole ecosystem. The
  type bound is **`RasterCell`** (copy + zero + a nodata convention);
  every numeric type (`u8`–`u64`, `i8`–`i64`, `f32`, `f64`)
  additionally implements **`RasterElement`** (ordering + numeric
  casts), which is what statistics, resampling and I/O bound on.
  With feature `complex`, `Complex<f32>`/`Complex<f64>` implement
  `RasterCell` for interferometric phase rasters. Carries
  `GeoTransform`, optional `CRS`, optional nodata.
- **`GeoTransform`** — affine georeferencing;
  `geo_to_pixel(x, y)` / `pixel_to_geo(col, row)` convert between
  world and grid space (`pixel_to_geo` returns cell centres).
- **`CRS`** — EPSG-based CRS handling, including reprojection support
  in the broader workspace.

Float rasters use **NaN as the universal in-memory nodata**: readers
map sentinel values to NaN, writers map back.

## Streaming and tiling

Two primitives cover the "raster larger than RAM" and "process in
windows" cases:

- **`StripProcessor` / `WindowAlgorithm`** — 1-D row strips with a
  vertical halo, file-to-file. Implement `WindowAlgorithm` for your
  kernel and the processor handles bounded-memory I/O.
- **`TileGrid`** — 2-D tiles with overlap, in memory. Cores tile the
  grid exactly; read extents add a clamped halo. This is the pattern
  for seam-free tiled inference and embeddings:

```rust
use surtgis_core::tiling::TileGrid;

for tile in TileGrid::new(rows, cols, 256, 16) {
    // 1. read tile.read_row..+tile.read_rows × tile.read_col..+tile.read_cols
    // 2. compute over the padded window
    // 3. keep only the core: offset (tile.core_offset_row(), tile.core_offset_col()),
    //    size tile.core_rows × tile.core_cols
}
```

## Cubes: aligned (time, band) stacks

`Cube` stacks `n_times × n_bands` single-band rasters verified to
share one grid (shape, transform, CRS — checked once at
construction). It is the container half of the data-cube story:
per-pixel time series and row-chunk streaming live here; temporal
*analysis* lives in consumer engines.

```rust
use surtgis_core::cube::Cube;

let cube = Cube::from_slices(times, vec!["red".into(), "nir".into()], slices)?;
let nir_series: Vec<f64> = cube.pixel_series(row, col, 1)?.collect();
for chunk in cube.chunks(256) {
    // chunk.views: one aligned 2-D view per (time, band) slice
}
```

Timestamps are Unix epoch seconds (`i64`), strictly increasing.

## Sampling (from surtgis-algorithms)

Point/patch extraction lives one crate up, in
`surtgis_algorithms::sampling`:

```rust
use surtgis_algorithms::sampling::{sample_at_points, extract_patches, PatchParams};

let x = sample_at_points(&[&slope, &aspect], &[(350_000.0, 6_300_000.0)])?;
let patches = extract_patches(&[&b02, &b03, &b04], &centers, PatchParams::default())?;
// patches.data is a flat [n, bands, size, size] f32 tensor
```

## Dependency footprint

`cargo tree --depth 1` for `surtgis-core` v0.15:

```text
anyhow, geo, geo-types, ndarray, num-traits,
ordered-float, serde, serde_json, thiserror, tiff
```

No GDAL, no tokio, no GUI. Heavy capabilities (cloud/STAC, Zarr,
colormaps, relief shading) live in sibling crates you opt into.
