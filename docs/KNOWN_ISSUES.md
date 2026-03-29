
## Cross-UTM STAC Composite (Open)

**Status:** Partially fixed, needs deeper investigation.

**Problem:** When compositing regions spanning multiple UTM zones (e.g., Huasco basin crossing EPSG:32719/32720), most dates are silently lost. Only 2/9 dates pass through the pipeline, resulting in 16.5% spatial coverage instead of expected >90%.

**Attempted fixes (commits 426c801, c63d5c1, 706a249):**
- Added error logging for silent mosaic/cloud mask failures
- Added per-tile EPSG tracking (TileRef struct)
- Added utm_to_wgs84() inverse projection
- Added reproject_raster_utm() with bilinear interpolation
- Added strip_bb reprojection per tile's native CRS

**What's still failing:**
- Fix did not improve coverage (16.5% vs 21.1% before fix)
- Only 2 of 9 dates produce valid composites
- Root cause likely deeper in COG reader bbox handling or strip processor alignment

**Needs:**
- `--verbose` mode that traces per-date: tiles fetched, tiles read, mosaic result, cloud mask result, valid pixel count
- Test with single-zone region to confirm pipeline works normally
- Test reprojected tiles individually to verify geometry is correct
- Compare strip_bb in source CRS vs reprojected CRS for each tile

**Workaround:** Use topographic/hydrological features (100% coverage from DEM) as primary ML inputs. Spectral indices contribute where available with NaN handling.

**Affected regions:** Any basin crossing UTM zone boundaries (Huasco, Copiapó, and others in Chile).
