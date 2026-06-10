# Extension contract

SurtGIS is the hub of a family of Rust geoscience engines (data cubes,
embeddings, InSAR, sediment routing, …). This chapter is the contract a
sibling engine signs when it builds on SurtGIS — four rules, designed
so that engines stay autonomous and the core stays small.

## 1. Dependency

A sibling crate depends on **`surtgis-core` only** for types, I/O and
streaming:

```toml
[dependencies]
surtgis-core = "0.15"
# optional, for analysis primitives (terrain factors, sampling, hydrology):
surtgis-algorithms = "0.15"
```

Never depend on `surtgis-cli` or `surtgis-gui` — those are
applications, not libraries, and their internals carry no stability
promise.

## 2. Interchange

The unit of exchange is **`Raster<T>`** together with its
`GeoTransform` and `CRS`. On disk, inputs and outputs travel as
GeoTIFF through `surtgis_core::io` (cloud-native formats are arriving
feature-gated: Zarr read exists in `surtgis-cloud`, GeoParquet is on
the roadmap).

Conventions your engine must respect:

- Float rasters use **NaN as in-memory nodata**.
- `pixel_to_geo` returns **cell centres**; grid origin is the top-left
  corner of the top-left cell.
- Multi-band data is exchanged as `&[&Raster<T>]` slices of
  grid-aligned single-band rasters (shape and transform must match).

## 3. Autonomy

Every engine is its own crate and binary, in its own repository. The
SurtGIS workspace does not absorb sibling engines, and the contract
never forces an engine to expose a `surtgis <x>` subcommand. (If an
engine *wants* CLI integration, propose it as an issue — the
mechanism would be a subcommand trait, opt-in.)

Domain logic stays in the engine: temporal analysis, interferometry,
conceptual hydrology and the like do **not** get merged into
`surtgis-core`. Core admits only types + I/O + streaming + primitives
with at least one concrete consumer.

## 4. Versioning

The ecosystem pins a **baseline**: `surtgis-core = "0.15"`. Within
0.15.x, the public API only grows — no breaking changes. Breaking
changes are announced one release ahead in the CHANGELOG under a
`Breaking` heading, and siblings migrate at their own pace.

A practical corollary: if your engine needs a capability that would
break the core API, open the discussion *before* the next minor
release, so the pre-announcement can ship in time.

## Smoke test

The minimum proof that an engine is wired correctly — read, process,
write through core:

```rust
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::Raster;

let r: Raster<f64> = read_geotiff("in.tif", None)?;
let (rows, cols) = r.shape();
let mut out = r.with_same_meta::<f64>(rows, cols);
out.set_nodata(Some(f64::NAN));
// ... your engine's processing ...
write_geotiff(&out, "out.tif", None)?;
```

If this compiles against the published `surtgis-core` (not a path
dependency), the contract holds.
