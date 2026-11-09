---
De: Sesión spatial-sampling (caso de estudio ChSPD, agregando covariables de clima)
Para: Sesión SurtGis
Fecha: 2026-07-17
Prioridad: ALTA (los valores de datos parecen correctos, pero el georeferencing de
  salida queda en el hemisferio equivocado — cualquier join espacial posterior con
  puntos reales del hemisferio sur falla o, peor, matchea silenciosamente con la
  ubicación espejo equivocada)
Versión afectada: v0.18.0 (main, post-merge de BUG_EXTRACT_ZERO_MATCH / commit d875f66)
---

# Bug: `surtgis stac download-climate` graba el bbox del hemisferio opuesto en el GeoTIFF de salida (fuentes Zarr con latitud descendente, ej. ERA5)

Pedís un bbox en el hemisferio sur (Chile, latitudes negativas) y el GeoTIFF de salida
queda georreferenciado en el hemisferio norte con la MISMA magnitud de latitud
(espejado en el ecuador). Los valores de datos parecen correctos (fueron leídos de la
región real pedida) — es el `GeoTransform`/bounds grabado en el archivo el que está mal.

## Reproducción

```bash
# bbox en el hemisferio sur (Chile centro, -34.6 a -34.2)
surtgis stac download-climate --catalog pc --collection era5-pds \
  --variable precipitation_amount_1hour_Accumulation \
  --datetime 2020-01-01/2020-01-31 --aggregate monthly-mean \
  --bbox=-71.9,-34.6,-71.4,-34.2 out_sur

surtgis info out_sur/2020-01_mean.tif
# Bounds: (-72.125000, 34.125000) - (-71.375000, 34.625000)
#                        ^^^^^^^^                 ^^^^^^^^
#         latitud POSITIVA (hemisferio norte) para un bbox pedido en el sur
```

Confirmado simétrico con el bbox espejado (mismo valor absoluto, hemisferio norte):

```bash
surtgis stac download-climate --catalog pc --collection era5-pds \
  --variable precipitation_amount_1hour_Accumulation \
  --datetime 2020-01-01/2020-01-31 --aggregate monthly-mean \
  --bbox=-71.9,34.2,-71.4,34.6 out_norte

surtgis info out_norte/2020-01_mean.tif
# Bounds: (-72.125000, -34.625000) - (-71.375000, -34.125000)
#                       ^^^^^^^^^^                 ^^^^^^^^^^
#         exactamente el espejo del anterior -- confirma que es un flip de signo
#         sistemático, no un bug de datos puntual
```

`fetch-mosaic` (Copernicus DEM vía Earth Search, usado toda la sesión anterior sobre
Chile) NO tiene este problema — es específico de la ruta Zarr de `download-climate`.

## Diagnóstico (causa raíz localizada, no solo síntoma)

`crates/cloud/src/zarr_reader.rs`:

1. Al abrir el store, si la latitud cruda viene descendente (N→S, típico de ERA5:
   `raw_lat[0]=90, raw_lat[last]=-90`), `lat_descending=true` (línea 190) y
   `self.lat_coords` se guarda **ya invertido a ascendente** (líneas 191-194):
   `lat_coords[0]=-90 ... lat_coords[last]=90`.

2. `lat_range_for_bbox()` (líneas 404-416) calcula `start`/`end` con `find_nearest`
   sobre `self.lat_coords` (ascendente) — hasta ahí correcto, esos índices ya
   apuntan al lugar correcto **dentro de `self.lat_coords`**. Pero si
   `lat_descending`, la función los vuelve a invertir antes de retornar:
   `Ok((n - end, n - start))` (línea 412) — necesario para que la lectura del
   array Zarr CRUDO (línea 267-268, `starts[lat_dim] = lat_start`) apunte al
   índice correcto dentro del array original (que sigue descendente en disco).

3. **El bug**: en `read_bbox()`, línea 334:
   ```rust
   let sub_lat = &self.lat_coords[lat_start..lat_end];
   ```
   `lat_start`/`lat_end` son el par YA INVERTIDO (paso 2, pensado para indexar el
   array Zarr crudo), pero acá se usan para indexar `self.lat_coords` — que es el
   array **ascendente**, un espacio de índices DISTINTO. El resultado: `sub_lat`
   termina apuntando a la región espejada de `self.lat_coords` (hemisferio
   opuesto), y ese `sub_lat` es lo que arma el `GeoTransform` de salida
   (línea 344, `build_geotransform(sub_lat, &sub_lon)` → línea 347,
   `raster.set_transform(geo_transform)`).
4. La lectura de VALORES (líneas 264-306, `starts[lat_dim]`/`sizes[lat_dim]`) SÍ usa
   los índices en el espacio correcto (el array crudo), así que los datos en sí
   deberían venir de la región real pedida — el `flip_rows` de la línea 324-328
   solo reordena filas para que coincidan con la convención top-down del
   `GeoTransform`, no cambia qué región geográfica se leyó. **No verifiqué esto
   empíricamente con un DEM/fuente independiente que cubra la misma zona** (ej.
   comparar contra ERA5 leído con otra herramienta) — palabras textuales: "los
   valores PARECEN correctos" por argumento de código, no confirmado con un
   segundo dato de referencia. Vale la pena que quien lo arregle confirme esto
   también, no soy 100% quien deba darlo por sentado.

## Fix recomendado

`lat_range_for_bbox` está sirviendo dos propósitos con un solo par de índices que
viven en dos espacios distintos (raw-descendente vs `self.lat_coords`-ascendente).
Opción más simple: que la función siga retornando el par en espacio
**ascendente** (el que ya calcula antes del flip, líneas 405-406) SIEMPRE, y mover
el flip `(n - end, n - start)` a donde se usa específicamente para indexar el
array Zarr crudo (líneas 267-268), no como parte del valor de retorno de la
función. Así línea 334 (`self.lat_coords[lat_start..lat_end]`) queda consistente
con el espacio de índices real de `self.lat_coords`, y el flip solo se aplica
localmente donde hace falta (la lectura del array crudo).

Pseudocódigo del cambio:
```rust
// lat_range_for_bbox: SIEMPRE en espacio de self.lat_coords (ascendente)
fn lat_range_for_bbox(&self, bbox: &BBox) -> Result<(usize, usize)> {
    let start = find_nearest(&self.lat_coords, bbox.min_y);
    let end = (find_nearest(&self.lat_coords, bbox.max_y) + 1).min(self.lat_coords.len());
    if start >= end { return Err(CloudError::BBoxOutside); }
    Ok((start, end))  // sin el flip acá
}

// en read_bbox(), al armar starts/sizes para el array crudo:
let (lat_start, lat_end) = self.lat_range_for_bbox(bbox)?;  // espacio self.lat_coords
let (raw_lat_start, raw_lat_end) = if self.lat_descending {
    let n = self.lat_coords.len();
    (n - lat_end, n - lat_start)
} else {
    (lat_start, lat_end)
};
starts[lat_dim] = raw_lat_start as u64;
sizes[lat_dim] = (raw_lat_end - raw_lat_start) as u64;
// ... y línea 334 sigue usando (lat_start, lat_end) sin flip, correcto para self.lat_coords
let sub_lat = &self.lat_coords[lat_start..lat_end];
```

## Test de regresión sugerido

Análogo al que ya existe para `BUG_EXTRACT_ZERO_MATCH`
(`non_stream_cells_survive_geotiff_roundtrip_as_finite_zero`): un Zarr sintético
con latitud descendente, pedir un bbox en el hemisferio sur, y assertar que
`raster.bounds()` (o el `GeoTransform` resultante) tiene signo de latitud
NEGATIVO — hoy ese assert fallaría con el bug presente.

## Alcance

Reproducido con `era5-pds` (latitud descendente, típico ERA5/reanalysis). No
probé `terraclimate` (que además tiene su propio problema: está servido como un
único Zarr global sin ítems STAC por tile, `stac search`/`download-climate`
devuelve "0 items" para cualquier bbox — puede ser una limitación esperada del
enfoque item-based de `download-climate` frente a datasets servidos como cubo
único, no necesariamente un bug, pero vale la pena que alguien con más contexto
del diseño de `download-climate` lo confirme). No probé si el bug también afecta
`--aggregate none` (lectura sin agregación temporal) — el flujo de índices de
latitud es el mismo código, así que probablemente sí, pero no lo verifiqué.

## Contexto

Detectado al intentar agregar covariables climáticas (temperatura/precipitación)
al caso de estudio ChSPD (organic_matter, Chile) después de completar el de
terreno. `fetch-mosaic` (ruta COG, no Zarr) no tiene este problema — se usó sin
inconvenientes toda la sesión anterior para el DEM Copernicus. Mientras se
resuelve, el workaround es NO confiar en el `GeoTransform`/bounds de los outputs
de `download-climate` para fuentes con latitud descendente sin corregir
manualmente el signo, o extraer valores por índice de array directo en vez de
por coordenada geográfica.
