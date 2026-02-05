# Reproyeccion automatica de BBox en STAC fetch

## Problema

`stac fetch` con Sentinel-2 (Earth Search) fallaba porque:

- **STAC search** usa bbox en WGS84 (EPSG:4326) segun la especificacion
- Los **COGs de Sentinel-2** estan almacenados en UTM (EPSG:326xx/327xx)
- El **CogReader** esperaba el bbox en coordenadas nativas del COG, no en WGS84

Resultado: `BBoxOutside` error al intentar leer tiles de Sentinel-2.

## Solucion implementada

Reproyeccion automatica WGS84 -> UTM usando formulas de Snyder (1987, USGS Professional Paper 1395), sin dependencias externas. Compatible con WASM.

### Decisiones de diseno

| Decision | Eleccion | Razon |
|---|---|---|
| Dependencia | Sin deps externas (formulas Snyder) | `proj` requiere C libproj, no funciona en WASM |
| Alcance CRS | WGS84 <-> UTM (zonas 1-60 N/S) | Sentinel-2, Landsat y la mayoria de imagenes usan UTM |
| Donde reproyectar | stac_reader.rs + CLI stac fetch | Transparente para el usuario |
| Deteccion CRS | `proj:epsg` del item STAC, fallback a COG metadata | Evita HTTP request adicional |

### Cobertura CRS

| EPSG | Descripcion | Soporte |
|---|---|---|
| 4326 | WGS84 (geograficas) | No-op (ya es WGS84) |
| 32601-32660 | UTM zonas 1-60 Norte | Soportado |
| 32701-32760 | UTM zonas 1-60 Sur | Soportado |
| Otros | Web Mercator, Lambert, etc. | Devuelve bbox original (sin error) |

## Archivos modificados

### 1. NUEVO: `crates/cloud/src/reproject.rs`

Modulo de reproyeccion pura en Rust (~150 lineas de codigo, ~130 lineas de tests).

**API publica:**
- `reproject_bbox_to_cog(bbox, target_epsg) -> BBox` - Reproyecta un bbox WGS84 al CRS destino
- `is_wgs84(epsg) -> bool` - Verifica si un EPSG es WGS84
- `parse_utm_epsg(epsg) -> Option<(zone, is_north)>` - Parsea codigo EPSG UTM

**Funciones internas:**
- `wgs84_to_utm(lon, lat, zone, north) -> (easting, northing)` - Conversion Snyder
- `meridional_arc(lat) -> f64` - Arco meridional (eq. 3-21 de Snyder)

**Constantes WGS84:**
- Semi-eje mayor: 6,378,137 m
- Aplanamiento: 1/298.257223563
- Factor de escala UTM: 0.9996
- Falso Este: 500,000 m
- Falso Norte (sur): 10,000,000 m

### 2. MODIFICADO: `crates/cloud/src/stac_models.rs`

Nuevo metodo `StacItem::epsg()` que extrae `proj:epsg` de las propiedades del item STAC:

```rust
pub fn epsg(&self) -> Option<u32> {
    self.properties.extra.get("proj:epsg")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
}
```

### 3. MODIFICADO: `crates/cloud/src/stac_reader.rs`

Nueva funcion `resolve_read_bbox()` que determina el bbox efectivo:

1. Intenta `proj:epsg` del item STAC (evita round-trip HTTP extra)
2. Fallback: CRS de los metadatos del COG (`reader.metadata().crs`)
3. Si es WGS84 o CRS desconocido: usa bbox original

Aplicado en `read_stac_asset()` y `search_and_read()`.

### 4. MODIFICADO: `crates/cloud/src/lib.rs`

Registrado `pub mod reproject;`.

### 5. MODIFICADO: `crates/cli/src/main.rs`

En `StacCommands::Fetch`: reproyeccion del bbox despues de abrir el COG:
- Usa `item.epsg()` (STAC) o `reader.metadata().crs.epsg()` (COG) como fallback
- Imprime `"Reprojected bbox to EPSG:{epsg}"` cuando reproyecta

## Tests

### Tests unitarios (11 tests, todos pasan)

| Test | Que verifica |
|---|---|
| `parse_utm_north` | EPSG 32601/32630/32660 -> zona correcta, norte |
| `parse_utm_south` | EPSG 32701/32721/32760 -> zona correcta, sur |
| `parse_utm_invalid` | EPSG 4326/3857/32600/32661/32700 -> None |
| `is_wgs84_test` | 4326 = true, otros = false |
| `madrid_wgs84_to_utm30n` | (-3.7037, 40.4168) -> (440298.94, 4474257.31) +-1m vs pyproj |
| `buenos_aires_wgs84_to_utm21s` | (-58.3816, -34.6037) -> (373317.50, 6170036.17) +-1m vs pyproj |
| `equator_central_meridian` | (-3.0, 0.0) zona 30 -> (500000.0, 0.0) exacto |
| `reproject_bbox_wgs84_noop` | EPSG 4326 -> bbox sin cambios |
| `reproject_bbox_unknown_epsg_noop` | EPSG 3857 -> bbox sin cambios |
| `reproject_bbox_madrid_utm30n` | Bbox Madrid -> UTM30N, dimensiones razonables |
| `reproject_bbox_southern_hemisphere` | Bbox Buenos Aires -> UTM21S con offset sur |

### Test del modelo STAC (1 test nuevo)

| Test | Que verifica |
|---|---|
| `epsg_from_proj_extension` | `item.epsg()` extrae 32630 del fixture |

### Precision

Validado contra pyproj (PROJ 9.x) con tolerancia < 1 metro:

```
Madrid:       diff easting = 0.002m, diff northing = 0.004m
Buenos Aires: diff easting = 0.002m, diff northing = 0.001m
Equator/CM:   exacto (0.0m)
```

## Resultado de compilacion y tests

```
$ cargo test -p surtgis-cloud --features native
running 38 tests ... test result: ok. 38 passed; 0 failed

$ cargo build -p surtgis
Compiling surtgis v0.1.0 ... Finished
```

## Uso

El usuario no necesita hacer nada diferente. El bbox de entrada siempre es WGS84:

```bash
# Sentinel-2 (UTM) - ahora funciona automaticamente
surtgis stac fetch --catalog es \
  --bbox="-3.75,40.40,-3.70,40.45" \
  --collection sentinel-2-l2a --asset red \
  --datetime "2024-06-24T00:00:00Z/2024-06-24T23:59:59Z" \
  output.tif

# Copernicus DEM (WGS84) - sigue funcionando igual
surtgis stac fetch --catalog pc \
  --bbox="-3.75,40.38,-3.65,40.45" \
  --collection cop-dem-glo-30 --asset data \
  output.tif
```

El CLI imprimira `"Reprojected bbox to EPSG:32630"` cuando detecte que el COG esta en UTM.

## Limitaciones

- Solo soporta UTM (EPSG 326xx/327xx). Otros CRS (Web Mercator, Lambert, etc.) pasan el bbox original sin reproyectar.
- La reproyeccion es solo WGS84 -> UTM (forward). No hay inversa UTM -> WGS84 implementada.
- Para bboxes que cruzan zonas UTM, se usa la zona del item STAC (que es la correcta para ese COG).
