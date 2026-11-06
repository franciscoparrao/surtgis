# Delineate lake catchments

`surtgis hydrology lake-catchment` computes, for many lakes at once, the
area that drains into each lake: catchment area, lake area and perimeter,
their ratio, lake surface and mean catchment elevation. It was built for
permafrost lake districts, where the usual `fill -> flow direction ->
watershed` chain fails twice. Filling erases the lake, which is itself a
depression. And a single pour pixel misses everything that drains into the
rest of the shoreline.

## How it works

For each lake:

1. A square window in the lake's UTM zone is centred on it. It starts at
   `--window-km`, or three times the lake's extent if that is larger.
2. Each `--dem` is warped onto that window: STAC items read by HTTP range,
   or local tiles.
3. The lake is rasterized from its polygon. A point lake is instead taken
   from `--water-mask` as the nearest water body, with a small
   morphological opening so the lake is not merged with its outflow river.
4. **The lake is the outlet.** Its cells drain the depression-filling step
   like the sea does, so a lake in a closed basin is not filled up to the
   basin's spill level. Every lake cell is a pour point, so the catchment
   is everything whose D8 path ends anywhere in the lake. Lakes upstream
   drain through and are part of it.
5. While the catchment or the lake touches the window edge (or a nodata
   cell), the window doubles, up to `--max-window-km`.

## Connected or local catchment

Lake districts on permafrost are full of closed basins, so "the catchment
of a lake" has two honest answers:

- **Connected** (`catchment_area_km2`, always reported): every depression
  is filled and spills onward. A closed basin whose overflow reaches the
  lake belongs to it. HydroSHEDS and HydroLAKES use this convention; it is
  the area that can contribute once storage is full.
- **Local** (`local_catchment_area_km2`, with `--sink-depth <m>`):
  depressions at least that deep keep their own drainage. Shallower ones
  are DEM noise and are filled. This is the area that drains to the lake
  directly. Pick the depth above the DEM's noise: about 1 m for
  Copernicus GLO-30.

The two can differ by one or two orders of magnitude on flat thermokarst
terrain. Reporting both is usually the right choice.

## Example

```bash
surtgis hydrology lake-catchment lakes.gpkg catchments.csv \
    --id-field lake_id \
    --dem stac:pc:cop-dem-glo-30 \
    --dem stac:pc:alos-dem \
    --dem /data/fabdem_tiles/ \
    --jobs 6 --masks-dir masks/
```

- `lakes.gpkg` holds polygons (preferred) or points, in any CRS with an
  EPSG code.
- A `--dem` is either `stac:<catalog>:<collection>[:asset]` or a local
  GeoTIFF/COG file or directory of tiles. FABDEM, for example, is
  distributed as 1-degree tiles.
- Point lakes need `--water-mask stac:pc:esa-worldcover:map` (water = 80).
- `--masks-dir` writes each catchment as a raster: 0 outside,
  1 catchment, 2 lake.

The CSV has one row per lake and DEM, with these columns:

- `lake_area_km2`, `lake_perimeter_km`, `lake_elev_m`;
- `catchment_area_km2`, `catchment_lake_ratio`, `catchment_mean_elev_m`;
- `window_km` and `status`, which is `ok`, `edge` (the catchment still
  reached the edge of the largest window, so the area is a lower bound) or
  `error`;
- `dem_spread`, present with two or more DEMs: the largest catchment area
  divided by the smallest. Values well above 1 flag flat terrain where the
  catchment is uncertain with any DEM;
- `water_trimmed`, for point lakes: the raw water component's size divided
  by the kept lake. Values well above 1 mean the opening cut the lake from
  a river, or split a narrow lake.

## Which DEM

- **Copernicus GLO-30** (`stac:pc:cop-dem-glo-30`) covers the whole globe
  and is a good default. It does not flatten small lakes, so lakes must come
  from polygons or a water mask, not from the DEM.
- **FABDEM** is GLO-30 with forests and buildings removed. Prefer it under
  boreal forest, where the canopy creates false divides. Its documented
  coverage ends at 80°N, and it is not on a STAC API: download the tiles.
- **ALOS AW3D30** (`stac:pc:alos-dem`) is an independent second opinion. It
  is a surface model and includes vegetation.

## Caveats

- In flat thermokarst terrain, the D8 catchment depends on decimetres of
  relief, and even on where the grid falls. Use two or more DEMs and read
  `dem_spread`.
- Lakes given as points depend on the water mask. When polygons exist
  (HydroLAKES, a national inventory), use them.
