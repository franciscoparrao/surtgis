---
De: Sesión spatial-sampling (caso de estudio ChSPD, candidate set organic_matter, Chile)
Para: Sesión SurtGis
Fecha: 2026-07-16
Prioridad: ALTA (bloquea por completo el uso de `extract` con datos reales — sin fix no hay forma de muestrear la salida de `pipeline features` en puntos)
Versión afectada: v0.17.0 instalada (repo local está en 0.18.0, no verificado si ya está resuelto ahí — revisar changelog antes de debuggear desde cero)
Estado: **RESUELTO** 2026-07-16, mismo día. Ver "Resolución" al final del documento.
---

# Bug: `surtgis extract` descarta puntos válidos que están comprobadamente dentro del raster

`surtgis extract --features-dir <dir> --points <geojson> --target <prop> <output.csv>` descarta
sistemáticamente puntos que están dentro de los bounds del raster y con valores no-NaN
verificados en las 25 capas del stack. En un caso real (536 puntos candidatos del dataset
ChSPD, Chile central) solo 10-18 se extrajeron según el buffer usado. En un repro mínimo
de 5 puntos pre-verificados uno por uno, **0/5**.

## Reproducción

Pipeline completo (Chile central, área de prueba):

```bash
# 1. DEM via Earth Search (Planetary Computer estaba caído externamente ese día — no es la causa)
surtgis stac fetch-mosaic --catalog es --collection cop-dem-glo-30 \
  --bbox=-72.05,-34.75,-71.25,-34.05 --compress dem_buffered_wgs84.tif

# 2. Reproyección a UTM 19S (CRS métrico, requerido por terrain/hydrology)
gdalwarp -t_srs EPSG:32719 -r bilinear -tr 30 30 -co COMPRESS=DEFLATE \
  dem_buffered_wgs84.tif dem_buffered_utm19s.tif

# 3. Stack de features (25 productos: 17 terrain + 8 hydrology)
surtgis pipeline features dem_buffered_utm19s.tif --outdir features_buffered --compress

# 4. Puntos reproyectados a la misma CRS que el raster (ogr2ogr, EPSG:4326 -> EPSG:32719)
ogr2ogr -t_srs EPSG:32719 -s_srs EPSG:4326 cluster_utm19s.geojson cluster.geojson

# 5. Extract
surtgis extract --features-dir features_buffered --points cluster_utm19s.geojson \
  --target organic_matter pilot_extracted_v2.csv
```

Salida real:

```
Reading point locations...
  536 features read
Extracting pixel values...
=========================================
EXTRACTION COMPLETE
=========================================
  Extracted: 18 samples
  Skipped:   518 (out of bounds, NaN, missing target)
```

Repro mínimo (5 puntos, cada uno verificado individualmente con `gdallocationinfo` contra
LAS 25 capas del stack — todas devuelven valores finitos en esas coordenadas exactas):

```bash
# five_test.geojson: 5 features, geometry.coordinates en EPSG:32719 (mismo CRS que el raster),
# properties incluye {"organic_matter": <float>}, geojson trae member "crs" declarando
# "urn:ogc:def:crs:EPSG::32719" (la que escribe ogr2ogr al reproyectar)

surtgis extract --features-dir features_buffered --points five_test.geojson \
  --target organic_matter five_out.csv
# -> Extracted: 0 samples / Skipped: 5 (out of bounds, NaN, missing target)
```

Verificación independiente de que esos 5 puntos SÍ están dentro y son válidos:

```bash
gdallocationinfo -valonly -geoloc features_buffered/terrain/slope.tif 277991.7338094519 6176085.819065477
# -> 1.14659570329384   (no NaN, no error)
# repetido para las 25 capas del stack -- ninguna devuelve NaN ni error en ese punto
```

## Diagnóstico (lo que descarté leyendo el código, y lo que queda abierto)

Rastreé la ruta completa en `crates/cli/src/handlers/ml.rs` (`fn extract`, ~línea 40-220) y
`crates/core/src/vector/geojson_reader.rs`. Cosas que verifiqué y **descarté** como causa:

1. **¿El reader ignora el `crs` member del GeoJSON (RFC 7946 strict) y confunde
   UTM con lon/lat?** No — `parse_geojson_crs` (geojson_reader.rs:70-86) parsea
   correctamente la member `crs` legacy (test `test_crs_legacy_member_epsg_urn`,
   línea 333-351, ya cubre exactamente mi caso: `EPSG::32719` con coordenadas
   `[350000.0, 6300000.0]`) y las coordenadas se leen **crudas**, sin reproyectar.
   Mis coordenadas ya estaban en EPSG:32719 (mismo CRS que el raster), así que
   esto no debería importar aquí.

2. **¿El geotransform que reporta `surtgis info` difiere del que usa `extract`
   internamente (ej. por venir de un GeoTIFF escrito por GDAL/gdalwarp en vez de
   por el propio SurtGIS)?** Comparé `gdalinfo` vs `surtgis info` sobre el mismo
   `terrain/slope.tif` (el que se usa como `ref_raster`): origin, pixel size y
   bounds coinciden exactamente.

3. **¿Alguna de las 25 capas tiene dimensiones/origen distintos al resto (ej. porque
   el sub-pipeline de hydrology reprocesa el DEM con un padding distinto al de
   terrain), lo que haría que `raster.get(row, col)` fallara con `Err` en ESA capa
   aunque `row,col` sea válido para `ref_raster` — y ese `Err` cae en la misma rama
   `_ => has_nan = true` que un NaN real (ml.rs línea 175-183)?** No — comparé
   `gdalinfo` de `terrain/slope.tif`, `terrain/aspect.tif`, `hydrology/twi.tif`,
   `hydrology/flow_accumulation.tif`, `hydrology/filled.tif`, `hydrology/hand.tif`,
   `hydrology/stream_network.tif`: las 7 tienen el mismo `Size` y `Origin` exacto.

4. **¿El target property (`organic_matter`, un float) falla al parsearse?** Por
   lectura de código no debería — `parse_properties` (geojson_reader.rs:266-290)
   distingue `as_i64()` vs `as_f64()`, y `3.0998` cae correctamente en la rama
   `AttributeValue::Float`. `ml.rs` línea 193 matchea `Float(v) => format!("{}", v)`.
   No lo descarté empíricamente (el mensaje de skip agrupa las 4 causas posibles
   en un solo contador — línea 219 `"(out of bounds, NaN, missing target)"` — así
   que no sé CUÁL de las 4 ramas de `skipped += 1` se está disparando realmente).

5. **¿`geo_to_pixel` (crates/core/src/raster/geotransform.rs:95-116) tiene un bug
   aritmético?** Leí la fórmula de la inversa afín completa (con términos de
   rotación) — para el caso sin rotación (nuestro caso, DEM UTM axis-aligned)
   se reduce a `col = dx/pixel_width`, `row = dy/pixel_height`, consistente con
   la convención "pixel-is-area" que usa `pixel_to_geo` (que suma `+0.5` para el
   centro). No encontré un error obvio, pero tampoco lo verifiqué con un test
   numérico directo.

## Causa más probable (no confirmada) y siguiente paso de debugging

Dado que descarté los 5 puntos anteriores por lectura de código pero el bug es
100% reproducible, el paso más barato para quien lo tome es instrumentar
directamente el loop de `ml.rs` (~línea 134-166): un `eprintln!` de
`(x, y, col_f, row_f, col, row, ref_raster.rows(), ref_raster.cols())` para las
primeras N features, comparado contra el índice de pixel esperado (derivable de
`gdallocationinfo -valonly` en la misma coordenada, que sabemos que SÍ acierta).
Eso va a mostrar de inmediato si es:
  - un desfase constante (bug de signo/offset en `geo_to_pixel`),
  - `col`/`row` swapeados,
  - o si en realidad el fallo NO es "out of bounds" sino "missing target" (rama 4
    del diagnóstico) y el problema está en `parse_properties`/`get_property`, no
    en la geometría — el contador agrupado no permite distinguirlo desde afuera.

Separar el contador `skipped` en sus 4 causas (no-geometry / not-a-point / out-of-bounds /
has-nan / missing-target) sería además una mejora de diagnosticabilidad barata y
valiosa independientemente del fix — habría ahorrado la mitad de esta investigación.

## Bug relacionado (mismo módulo GeoJSON, mismo tipo de gotcha CRS)

`docs/bug-reports/BUG_FLUVIAL_GEOJSON_CRS.md` (2026-05-23, marcado resuelto en README)
documenta un problema hermano del lado del **writer** GeoJSON del módulo fluvial
(coordenadas proyectadas sin declarar `crs`). Este reporte es del lado **reader** de
`extract`, y — a diferencia de aquel — confirmé que el reader en sí (`geojson_reader.rs`)
está bien y tiene tests que cubren exactamente el caso EPSG:32719. Vale la pena que
quien lo audite tenga ambos reportes a mano por si comparten alguna causa raíz en un
layer más profundo (ej. `crs::CRS` o cómo se propaga `fc.crs()` — noté que `ml.rs`
nunca lee `fc.crs()` en absoluto, así que si algún día se agrega auto-reproyección
en `extract`, ese es el punto de entrada).

## Workaround actual (para usuarios)

Mientras se resuelve, usar `gdallocationinfo` directo por punto en vez de `surtgis extract`
(confirmado que funciona correctamente para el mismo stack de rasters):

```bash
for xy in $(cat points.txt); do
  gdallocationinfo -valonly -geoloc raster.tif $xy
done
```

O, para volumen (miles de puntos), un script que llame `gdallocationinfo` en modo batch
(acepta pares XY por stdin, uno por línea) y arme el CSV final en Python/awk.

## Contexto

Detectado durante la construcción del candidate set de covariables de terreno para el
caso de estudio ChSPD (paper de `spatial-sampling`, 12006 puntos topsoil en Chile). El
pipeline hasta `pipeline features` (fetch DEM vía Earth Search, reproyección, generación
del stack de 25 features) corrió perfecto y rápido (~150-200s por tile de prueba). El
bug de `extract` es el único bloqueante para escalar el caso de estudio a las 38 bandas
de latitud que cubren los 12006 puntos reales — sin fix, se sigue con el workaround de
`gdallocationinfo` para no bloquear el paper, pero vale la pena arreglarlo pronto porque
`extract`/`extract-patches` son la puerta de entrada estándar del motor hacia ML
(`/ml`, según el propio skill de SurtGIS) y este bug los deja inservibles con datos reales.

## Resolución

**Causa raíz confirmada** (no era ninguno de los 5 candidatos descartados en el
diagnóstico original, ni el módulo GeoJSON): el comando real invocado por el reporte —
`surtgis extract` a nivel de CLI — corre en `crates/cli/src/handlers/extract.rs`, **no**
en `crates/cli/src/handlers/ml.rs::handle_extract_samples` (esa es la ruta de
`surtgis ml extract-samples`, un subcomando distinto con lógica casi idéntica pero
código separado — la investigación original rastreó el archivo equivocado). El primero
delega el sampling a `surtgis_algorithms::sampling::sample_at_points`
(`crates/algorithms/src/sampling/mod.rs`).

Repro real construido con el fixture `tests/fixtures/andes_chile_30m_utm.tif` +
`pipeline features` + 5 puntos en centros de píxel verificados: reprodujo **0/5** exacto,
con los mismos valores de `gdallocationinfo` que reporta el diagnóstico original. Un
binario de instrumentación (`eprintln!` de `(col_f, row_f, col, row)` + valor por capa,
tal como sugería el reporte) mostró que **las 5 capas geométricas eran correctas**
(mismo `col`/`row` en las 25 capas, sin mismatch de shape/transform) pero **`stream_network`
daba `NaN` en las 5 vía el lector nativo de SurtGIS mientras `gdallocationinfo` (que no
aplica nodata por default) leía el valor crudo correcto: `0`**.

`crates/algorithms/src/hydrology/stream_network.rs` declaraba:

```rust
let mut output = flow_acc.with_same_meta::<u8>(rows, cols);
output.set_nodata(Some(0));   // <-- el bug
```

`stream_network` es una máscara binaria (`0` = no-stream, `1` = stream) donde **`0` es el
valor válido mayoritario** (la red de streams es delgada; casi toda la cuenca es
"no-stream"). Declarar `0` como nodata hace que `cast_and_normalize` (`crates/core/src/io/native.rs`)
convierta cada celda "no-stream" en `NaN` al castear a `f64` — exactamente el tipo de
lectura que hace `extract`. Cada punto que cae fuera de la red de streams (la inmensa
mayoría en cualquier dataset real) se descartaba. Los 10-18/536 que sí se extraían en el
caso real eran, casi con certeza, los pocos puntos que caían justo sobre una celda de
stream (`1 ≠ 0`, no nodata).

Grep del mismo patrón (`set_nodata(Some(0))`) en todo `crates/algorithms/` encontró 16
sitios; 6 comparten el mismo defecto real (máscara binaria/dispersa donde `0` es el valor
común, no ausente) y se corrigieron quitando el `set_nodata`:

- `hydrology/stream_network.rs` (la causa raíz de este reporte)
- `terrain/viewshed.rs` (×2: naive + XDraw)
- `terrain/pderl_viewshed.rs`
- `terrain/lineament.rs` (×3: `kh_out`, `kv_out`, `class_out`)

Los otros 10 sitios (`geomorphons`, `felzenszwalb`, `slic`, `connected_components`,
`nested_depressions`, `watershed`, `watershed_parallel`, `flow_direction`, y un helper de
test en `flats.rs`) usan `0` legítimamente como fondo/sin-clasificar — labels de
componentes/cuencas donde `0` es minoritario por diseño, o códigos donde `0` cae
genuinamente fuera del rango válido. Se dejaron sin tocar.

Verificado end-to-end: repro real (mismo fixture, mismos 5 puntos) pasó de **0/5 → 5/5**
extraídos, con valores idénticos a `gdallocationinfo` en todas las 25 capas. Test de
regresión agregado en `stream_network.rs` (`non_stream_cells_survive_geotiff_roundtrip_as_finite_zero`)
cubre el roundtrip GeoTIFF completo que reprodujo el bug.

**Mejora de diagnosticabilidad** (la que el reporte pedía "independientemente del fix"):
el contador `skipped` agregado en `extract.rs` ahora se reporta separado en 3 causas —
`Not a point geometry`, `Missing/invalid target`, `Out of bounds or NaN` — en vez de un
solo número con las 3 causas mezcladas en el mensaje.
