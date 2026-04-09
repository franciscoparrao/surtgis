# STAC Format Coverage Roadmap

> Audited: 2026-04-08
> Current coverage: ~100% of raster assets across Planetary Computer + Earth Search
> Updated: 2026-04-08 — GRIB2 reader implemented

## Current Support

| Format | Media Type | Reader | Feature Gate |
|--------|-----------|--------|-------------|
| COG (GeoTIFF) | `image/tiff; application=geotiff; profile=cloud-optimized` | `CogReader` | always |
| Zarr | `application/vnd+zarr` | `ZarrReader` | `zarr` |
| NetCDF | `application/x-netcdf`, `application/netcdf` | `NetCdfReader` | `netcdf` |
| GRIB2 | `application/wmo-GRIB2` | `GribReader` | `grib` |

## Planned Formats

### Priority 1: JPEG 2000 (`image/jp2`)

**Impact**: Sentinel-2 L1C on Earth Search — 14 bands, no COG alternative. One of the most used collections globally.

**Collections affected**: sentinel-2-l1c (Earth Search), some Sentinel-2 L2A dual-format assets

**Implementation approach**:
- Feature gate: `jp2`
- Crate options:
  - `jpeg2000` (pure Rust, immature)
  - `openjpeg-sys` / `openjp2` (bindings to OpenJPEG C library, mature)
  - GDAL feature-gated fallback (already exists in SurtGIS)
- Reader: `Jp2Reader` — HTTP Range-based like CogReader (JP2 supports tile-based access via codestreams)
- Note: `stac_introspect.rs` already detects `image/jp2` for band introspection (line 106), so only the pixel decoder is missing

**Estimated effort**: Medium — JP2 tile access is more complex than COG due to codestream structure

### Priority 2: HDF4 / HDF-EOS (`application/x-hdf`)

**Impact**: 19 MODIS collections on Planetary Computer — burned area, vegetation indices (NDVI/EVI), surface reflectance, snow cover, LST, GPP, NPP, LAI, evapotranspiration, fire detection.

**Collections affected** (Planetary Computer):
- modis-09A1-061, modis-09Q1-061 (surface reflectance)
- modis-11A1-061, modis-11A2-061 (land surface temperature)
- modis-13A1-061, modis-13Q1-061 (vegetation indices)
- modis-14A1-061, modis-14A2-061 (fire)
- modis-15A2H-061 (LAI/FPAR)
- modis-16A3GF-061 (evapotranspiration)
- modis-17A2HGF-061, modis-17A3HGF-061 (GPP/NPP)
- modis-43A4-061 (BRDF)
- modis-64A1-061 (burned area)
- modis-10A1-061, modis-10A2-061 (snow)

**Implementation approach**:
- Feature gate: `hdf4`
- Crate: `hdf4-sys` or custom bindings to libhdf4
- HDF-EOS2 is HDF4 + geospatial metadata convention (SDS arrays + geolocation fields)
- Reader: `Hdf4Reader` — download file, read SDS (Scientific Data Set) by name
- Must handle sinusoidal projection (MODIS native grid) → WGS84 reprojection
- Swath data has irregular geolocation (lat/lon arrays per pixel)

**Estimated effort**: High — HDF-EOS sinusoidal grid + swath handling is non-trivial

### Priority 3: GRIB2 (`application/wmo-GRIB2`)

**Impact**: 5 collections on Planetary Computer — NOAA HRRR (high-res weather), ECMWF forecasts, MRMS precipitation (3 products).

**Collections affected** (Planetary Computer):
- noaa-hrrr (High-Resolution Rapid Refresh, 3km, hourly)
- ecmwf-forecast (global weather, 0.25°, 6-hourly)
- noaa-mrms-qpe-1h, noaa-mrms-qpe-24h, noaa-mrms-qpe-pass2 (radar precipitation)

**Implementation approach**:
- Feature gate: `grib`
- Crate options:
  - `eccodes-rs` (bindings to ECMWF ecCodes C library, most complete)
  - `gribberish` (pure Rust GRIB2 reader, newer, less mature)
- Reader: `GribReader` — download file, read message by parameter/level/time
- GRIB2 structure: messages with parameter discipline/category/number, vertical levels, forecast times
- Must map GRIB parameter codes to human-readable names (e.g., 0/1/8 → "Total Precipitation")

**Estimated effort**: Medium — GRIB2 message structure is well-documented but verbose

### Low Priority (out of raster scope)

| Format | Type | Notes |
|--------|------|-------|
| GeoParquet | Vector/tabular | 6 collections on PC (biodiversity, buildings, census) |
| COPC | Point cloud | 1 collection (USGS 3DEP LiDAR) |
| FlatGeobuf | Vector | Some collections, alternative to GeoJSON |

## Coverage Projection

| Milestone | Formats | Coverage |
|-----------|---------|:--------:|
| **Current** | COG + Zarr + NetCDF | ~94% |
| **+JP2** | + JPEG 2000 | ~96% |
| **+HDF4** | + HDF-EOS | ~99% |
| **+GRIB2** | + GRIB2 | ~100% |

## Data Source

Audit based on:
- Planetary Computer STAC API: 135 collections, ~875 raster-relevant asset definitions
- Earth Search STAC API: 9 collections, ~133 asset definitions
- NASA CMR-STAC: HDF5/HDF-EOS2 dominant (not counted in totals above)
