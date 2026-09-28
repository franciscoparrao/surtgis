# Verify where a raster came from

Every GeoTIFF or COG that SurtGIS writes carries a provenance record inside
the file, as the `SURTGIS_PROVENANCE` item of the standard `GDAL_METADATA`
tag. `gdalinfo`, rasterio and QGIS show it as ordinary metadata, and it
survives copies, renames and object-store round trips. The pixels are never
touched by it.

The record holds:

- the engine name and version, the platform and the time of writing;
- the operation and the command line as typed, plus the working directory;
- the number of worker threads (and the seed, for stochastic algorithms);
- every input the command read, with its size and BLAKE3 digest when it
  was a local file (URLs are recorded without a digest);
- a BLAKE3 digest of the output pixel array as written, with its shape,
  band count and sample type.

## Show the record

```bash
surtgis provenance slope.tif
surtgis provenance --json slope.tif   # the raw record
```

`surtgis info` prints a one-line summary when a record is present.

## Verify a file

```bash
surtgis verify slope.tif
```

`verify` decodes the array, recomputes its digest and compares it with the
recorded one, then re-hashes each recorded input it can find (as given, or
relative to the recorded working directory). It exits non-zero on any
mismatch. Inputs that were not hashed at write time, or that are no longer
on disk, are reported as unverifiable but do not fail the check; pass
`--skip-inputs` to check only the output.

## Opt out

Hashing the inputs costs one sequential read of each file. To skip the
record entirely:

```bash
surtgis --no-provenance terrain slope dem.tif slope.tif
SURTGIS_NO_PROVENANCE=1 surtgis terrain slope dem.tif slope.tif
```

## From the library

`surtgis_core::provenance::Provenance` is the record;
`io::write_geotiff_with_provenance` and `io::write_cog_with_provenance`
embed one explicitly (the writer fills in the output digest), and
`io::read_provenance` reads it back. Tools that behave like the CLI can
install the process hooks (`set_input_observer`, `set_output_provider`)
so every native reader and writer participates without touching call
sites. Without hooks or an explicit record, nothing is written.

## What the digest means

`data_blake3` is BLAKE3 over a header (`surtgis-data/1`, sample type,
rows, columns, bands) followed by each band's samples in row-major order as
native little-endian bytes, band after band. Two files with the same
digest hold the same array. Because SurtGIS's results are bit-identical
across runs and thread counts (a contract checked in CI), re-running the
recorded command on the recorded inputs reproduces the digest.
