# Bug report archive

Historical bug reports and diagnostics from field use of SurtGIS, kept for the
record. **All of these are resolved** in the versions noted below; they are not
open issues. Each file captures the original symptom, diagnosis, and (where
applicable) the reproduction that drove the fix.

| Report | Area | Reported on | Resolved in |
|---|---|---|---|
| `SURTGIS_BUG_rasterize.md` | `rasterize`: geographic CRS dropped (`LOCAL_CS[metre]`) + O(1000×) slow fill | 0.15.4 | 0.16.3 (#62) |
| `BUG_TILE_DECODE_BPS15_STRIPING.md` | COG tile decode striping at bps=15 | 0.7.0 | v0.6.x/0.7 predictor fixes |
| `BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md` | STAC composite RAM budget vs real | 0.7.0 | v0.6.24 outer-band refactor |
| `BUG_RAM_V0624_LATE_GROWTH.md` | Late RAM growth (heap fragmentation) | 0.6.24 | v0.6.26 (mimalloc) |
| `BUG_RAM_V0623_STILL_OVER.md` | STAC composite RAM still over budget | 0.6.23 | v0.6.24 |
| `BUG_RAM_V0622_STILL_OVER.md` | STAC composite RAM still over budget | 0.6.22 | v0.6.23/24 |
| `BUG_RAM_COMPOSITE_MAULE.md` | STAC composite RAM spike (Maule) | 0.6.19 | v0.6.19 auto-cap → v0.6.24 |
| `BUG_RAM_SPIKE_ES_MAULE.md` | RAM spike, Earth Search Maule | 0.6.19 | v0.6.22–24 |
| `BUG_RAM_SPIKE.md` | STAC composite RAM spike | 0.6.x | v0.6.22–26 |
| `BUG_STAC_SEARCH_LIMIT.md` | STAC search result limit | 0.6.x | v0.6.x STAC pagination |
| `BUG_EARTH_SEARCH_LIMIT.md` | Earth Search result limit / raw `s3://` hrefs | 0.6.10 | v0.6.28 (`normalize_url`) |
| `BUG_PC_LIMIT_1000.md` | Planetary Computer 1000-item limit | 0.6.x | v0.6.x STAC pagination |
| `BUG_SAS_TOKEN_EXPIRY.md` | Azure SAS token expiry mid-read | 0.6.x | v0.6.2–12 SAS refresh |
| `BUG_MULTIBAND_STRIP3_GAP.md` | Multi-band composite strip-3 gap | 0.6.x | v0.6.2–12 multi-band fixes |
| `BUG_STRIP7_CUENCA03.md` | Strip-7 artefact, cuenca 03 | 0.6.4 | v0.6.x strip fixes |
| `BUG_STRIP7_OOB_RESAMPLE.md` | Strip-7 out-of-bounds resample | 0.6.x | v0.6.x |
| `DIAG_STRIP7_RESULTS.md` | Strip-7 diagnostics | 0.6.x | (diagnostic notes) |
| `BUG_FLUVIAL_GEOJSON_CRS.md` | Fluvial GeoJSON CRS handling | 0.10.0 | v0.1x fluvial fixes |
