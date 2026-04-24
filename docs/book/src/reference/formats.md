# Supported formats

## Raster formats

| Format | Read | Write | Feature flag | Notes |
|---|---|---|---|---|
| GeoTIFF | ✓ | ✓ | (always) | Predictor 1/2/3, DEFLATE, LZW, GDAL_NODATA |
| Cloud Optimized GeoTIFF (COG) | ✓ | — | `cloud` | HTTP range reads; write not supported yet |
| Zarr (via `zarrs`) | ✓ | — | `zarr` | Azure Blob + Planetary Computer auth |
| NetCDF | ✓ | — | `netcdf` | Requires libnetcdf system library |
| GRIB2 | ✓ | — | `grib` | Requires libgribapi or eccodes |
| JP2000, HDF4, ECW, MrSID, ERDAS .img | — | — | — | Use GDAL |

## Vector formats

| Format | Read | Notes |
|---|---|---|
| GeoJSON (`.geojson`, `.json`) | ✓ | |
| Shapefile (`.shp` + sidecar files) | ✓ | Via `shapefile` crate |
| GeoPackage (`.gpkg`) | ✓ | Via `rusqlite` |

Vector writing is not supported in v0.7.0. Clip / rasterize accept vectors
as input; produce rasters as output.

## Floating-point value handling

For float element types (`f32`, `f64`), the `GDAL_NODATA` tag value is
replaced with `NaN` on read. Algorithms treat `NaN` as "invalid" and
propagate it as such — no sentinel values buried inside floats.

On write, the same convention: `NaN` pixels are written out as
`GDAL_NODATA` values if the output format supports the tag.

For integer types (`i16`, `u16`, `u8`), the original nodata sentinel is
preserved.

## Endianness and TIFF predictors

The native GeoTIFF reader handles both big-endian and little-endian
files. Predictor 1 (no prediction), 2 (horizontal differencing), and 3
(floating-point prediction) are all supported. Predictor 3 was added in
v0.6.21 specifically for Copernicus DEM GLO-30 which uses it.

## Compression support

Read: DEFLATE, LZW, packbits, none.
Write: DEFLATE (via `--compress` flag) or none.

Write support for LZW is not planned — DEFLATE compresses better on
geospatial rasters for comparable CPU cost.

## Size limits

The `tiff` crate enforces a default size limit to prevent malicious
inputs. For large DEMs (e.g. 20,000 × 20,000 or bigger), SurtGIS sets
`Limits::unlimited()` internally so uncompressed reads aren't rejected.

Files on disk can be arbitrarily large; streaming algorithms don't care
about total size.
