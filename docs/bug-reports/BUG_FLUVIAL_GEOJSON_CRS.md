---
De: Sesión Pangal (colaboración Quezada × Parra, frente Puerto Aysén)
Para: Sesión SurtGis
Fecha: 2026-05-23
Prioridad: MEDIA (no rompe el cálculo, sí rompe consumo downstream en cualquier cliente GIS estándar)
Versión afectada: v0.10.0 (módulo fluvial completo)
---

# Bug: outputs GeoJSON de `surtgis fluvial *` escriben coordenadas proyectadas sin declarar CRS

Los tres subcomandos del módulo fluvial que producen GeoJSON emiten las coordenadas en el CRS proyectado del raster de entrada (correcto), pero **omiten** la declaración de CRS en el header del GeoJSON. Por RFC 7946 §4 la ausencia de declaración significa **WGS84 (EPSG:4326)**. Cualquier cliente que respete el estándar (geopandas, QGIS via OGR moderna, MapLibre) interpreta las coordenadas como grados WGS84, generando lecturas absurdas.

## Reproducción

Pipeline mínimo sobre cualquier DEM proyectado (ej. FabDEM en UTM 18S):

```bash
surtgis hydrology fill-sinks            dem.tif filled.tif
surtgis hydrology flow-direction        filled.tif fdir.tif
surtgis hydrology flow-accumulation     fdir.tif facc.tif
surtgis hydrology stream-network --from-facc --threshold 1000 facc.tif streams.tif
surtgis hydrology watershed --pour-points "1429,778;1427,774;1613,466" fdir.tif basins.tif

surtgis fluvial chi      streams.tif fdir.tif facc.tif chi.tif
surtgis fluvial ksn      streams.tif fdir.tif facc.tif filled.tif ksn.tif \
                         --segments ksn_segments.geojson
surtgis fluvial knickpoints streams.tif fdir.tif facc.tif filled.tif knickpoints.geojson
surtgis fluvial divide-migration basins.tif filled.tif facc.tif divides.geojson --chi chi.tif
```

Inspección de los 3 outputs:

```python
import json
for f in ["ksn_segments.geojson", "knickpoints.geojson", "divides.geojson"]:
    d = json.load(open(f))
    g0 = d["features"][0]["geometry"]
    coord0 = g0["coordinates"][0] if g0["type"] == "LineString" else g0["coordinates"]
    print(f"{f:30s}  crs declared: {'crs' in d}  first coord: {coord0}")
```

Salida real (datos UTM 18S):

```
ksn_segments.geojson           crs declared: False  first coord: [687487.5, 5009287.5]
knickpoints.geojson            crs declared: False  first coord: [685112.5, 4979862.5]
divides.geojson                crs declared: False  first coord: [676625.0, 4991412.5]
```

Esos números son coordenadas UTM 18S en metros. Cualquier cliente estándar las lee como WGS84 y arroja `inf` al reproyectar, o las plotea fuera del planeta.

## Diagnóstico

Dos especificaciones GeoJSON conviven en la práctica GIS:

1. **GeoJSON 2008** (anterior) permitía la member `crs` para declarar el CRS no-WGS84.
2. **RFC 7946 (2016)** prohíbe la member `crs` y obliga a WGS84 (EPSG:4326). Si quieres otro CRS tienes que reproyectar antes de escribir.

El comportamiento actual de SurtGIS fluvial cae en una tierra de nadie: no declara CRS (consistente con RFC 7946) pero las coordenadas no están en WGS84 (incumple RFC 7946 y rompe los clientes).

Verificado con geopandas 1.x, lectura del archivo con `gpd.read_file()` retorna `crs=EPSG:4326`, y un `.to_crs("EPSG:32718")` posterior envía los puntos al infinito (sale del dominio válido del proj4).

## Fix recomendado

**Opción A — RFC 7946 strict (preferida)**: reproyectar las coordenadas a WGS84 antes de serializar. SurtGIS ya tiene PROJ disponible vía la dependencia `gdal` del workspace. En `crates/algorithms/src/fluvial/io.rs` (o donde viva el writer):

```rust
use proj::Proj;

fn write_geojson_wgs84<G: GeometryWriter>(
    geoms: &[G],
    source_crs: &str,        // ej. "EPSG:32718"
    output_path: &Path,
) -> Result<()> {
    let to_wgs84 = Proj::new_known_crs(source_crs, "EPSG:4326", None)?;
    // ... transform each coord pair before serializing
}
```

Coste: una llamada `Proj::transform` por vertex. Despreciable vs el cálculo morfométrico.

**Opción B — GeoJSON 2008 compat (menos limpia, no rompe nada)**: mantener coords en CRS original y declarar la member `crs` en el header:

```rust
// En el serializer:
let crs_member = json!({
    "type": "name",
    "properties": { "name": format!("urn:ogc:def:crs:EPSG::{epsg}") }
});
feature_collection.insert("crs", crs_member);
```

Esto es lo que SAGA y GRASS hacen por compatibilidad histórica. Funciona con todo cliente práctico pero no es estándar moderno.

**Recomiendo Opción A**. RFC 7946 es lo que asume el ecosistema moderno (geopandas, MapLibre, deck.gl, web mapping en general). Una flag opcional `--keep-crs` puede ofrecerse para usuarios que quieran preservar UTM nativo, en cuyo caso aplicar Opción B internamente con la declaración explícita.

## Workaround actual (para usuarios)

Mientras el fix se libera, al cargar cualquier GeoJSON de `surtgis fluvial *` hay que sobrescribir el CRS:

```python
import geopandas as gpd
gdf = gpd.read_file("ksn_segments.geojson")
gdf = gdf.set_crs("EPSG:32718", allow_override=True)  # o el CRS del raster fuente
```

Esto evita el problema pero exige que el usuario sepa cuál era el CRS del raster de entrada — información que SurtGIS conoce pero no propaga.

## Alcance

Confirmado en los 3 outputs vector del módulo fluvial:

| Subcomando | Output afectado |
|---|---|
| `surtgis fluvial ksn --segments` | LineString FeatureCollection |
| `surtgis fluvial knickpoints` | Point FeatureCollection |
| `surtgis fluvial divide-migration` | LineString FeatureCollection |

No verifiqué otros módulos. Si existen writers compartidos (`vector::geojson::write_*` u análogo), revisar todos los call sites para asegurar consistencia (probable que `surtgis hydrology stream-network --to-geojson` o equivalentes tengan el mismo patrón).

## Contexto

Detectado durante el primer análisis morfométrico completo del frente Puerto Aysén (proyecto colaboracion_paulo, mayo 2026). Los outputs son numéricamente correctos — el bug es solo en la metadata. Una vez aplicado el workaround, los resultados son confiables y la pipeline completa corre en <2 segundos sobre 896 km² a 25 m de resolución (notable, dicho sea de paso).

Sin fix, cualquier user que arranque el módulo fluvial sobre un CRS proyectado va a tropezar con esto en los primeros 5 minutos. Vale priorizarlo antes de hacer ruido público de v0.10.0.
