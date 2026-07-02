# Bug report — `surtgis rasterize`: CRS no preservado (output `LOCAL_CS` en metros) + rendimiento

**Versión:** surtgis 0.15.4
**Plataforma:** Linux x86-64, GeoTIFF de referencia en **EPSG:4326** (CRS geográfico, grados)
**Comando:** `surtgis rasterize <vector.gpkg> <out.tif> --reference <ref.tif> --attribute GLACIER --compress`

---

## Resumen

`rasterize` produce una máscara **geométricamente correcta** (idéntica pixel a pixel a la de `rasterio.features.rasterize`, 0 discrepancias) y **preserva el geotransform exacto**, pero:

1. **(Bug principal) No preserva el CRS de la referencia.** El output sale con
   `LOCAL_CS["unnamed", UNIT["metre",1,AUTHORITY["EPSG","9001"]], AXIS["Easting",EAST], AXIS["Northing",NORTH]]`
   en vez del `EPSG:4326` de la referencia. Peor aún: rotula como **metros** unas coordenadas que están en **grados**.
2. **(Secundario) Rendimiento.** 323 polígonos sobre una grilla 6912×6912 tardan **~10 min** (vs **0.5 s** de `rasterio` para el mismo resultado).

---

## Detalle 1 — CRS no propagado

### Esperado
El output GeoTIFF debe heredar el CRS de `--reference` (aquí `EPSG:4326`).

### Observado
```
CRS referencia : EPSG:4326
CRS output     : LOCAL_CS["unnamed", UNIT["metre",1,AUTHORITY["EPSG","9001"]], ...]
```

### Lo que SÍ está bien (acota el fix)
- **Geometría idéntica:** comparado contra `rasterio.features.rasterize` sobre la misma grilla → `np.array_equal == True`, **0 px de discrepancia** (28.925.461 px de glaciar en ambos).
- **Geotransform preservado exacto:** origen `(-73.80, -46.40)`, pixel `8.983e-05°`, idéntico al de la referencia (`transform.almost_equals == True`).

⇒ El defecto está **solo en la ruta de escritura del CRS** del GeoTIFF, no en el rasterizado ni en el georreferenciado. Probablemente el writer no copia el CRS de la referencia y, al no poder mapear/encodear un CRS **geográfico (EPSG:4326)**, cae a un `LOCAL_CS` genérico en metros por defecto.

### Fix sugerido
- Propagar el CRS del raster `--reference` directamente al output (copiar EPSG code / GeoKeys / cadena PROJ).
- Asegurar que el writer encodea CRS **geográficos** (no solo proyectados en metros). El fallback a `LOCAL_CS["metre"]` es incorrecto para datos en grados y puede causar errores silenciosos aguas abajo (cálculo de área, reproyección, units).

### Impacto
- Herramientas downstream (rasterio/GDAL/QGIS) leen el mask como "sin CRS / metros locales" → no alinea con otras capas, reproyecciones erróneas, áreas mal calculadas.
- Afecta cualquier flujo con tiles en EPSG:4326 (p. ej. exports de Google Earth Engine, que salen en 4326 por defecto).

---

## Detalle 2 — Rendimiento de rasterize

| Herramienta | Wall clock | RAM máx (RSS) |
|---|---|---|
| `surtgis rasterize` 0.15.4 | **10:11 min** | 948 MB |
| `rasterio.features.rasterize` | **0.5 s** | (bajo) |

323 polígonos, grilla 6912×6912 (≈47.8M celdas), output 1 banda. Mismo resultado geométrico. Una diferencia de ~1200× sugiere un algoritmo de relleno no-escanline (posible point-in-polygon por celda) o coste extra por manejo de coordenadas geográficas. Vale revisar el path de fill (scanline / edge-list por fila).

---

## Reproducción mínima

```bash
# ref.tif: GeoTIFF 1 banda, EPSG:4326, 6912x6912 (cualquier grilla geográfica sirve)
# glaciares.gpkg: polígonos en EPSG:4326 con columna entera GLACIER=1

surtgis rasterize glaciares.gpkg out.tif --reference ref.tif --attribute GLACIER --compress

# Verificar:
python -c "import rasterio; print(rasterio.open('out.tif').crs)"
# -> LOCAL_CS[...metre...]   (esperado: EPSG:4326)
```

### Verificación de equivalencia geométrica (prueba de que el rasterizado es correcto)
```python
import rasterio, numpy as np, geopandas as gpd
from rasterio.features import rasterize
sg  = rasterio.open('out.tif').read(1) > 0
ref = rasterio.open('ref.tif')
rio = rasterize(((g,1) for g in gpd.read_file('glaciares.gpkg').geometry),
                out_shape=(ref.height, ref.width), transform=ref.transform,
                fill=0, dtype='uint8') > 0
assert np.array_equal(sg, rio)   # pasa: geometría idéntica
```

---

## Contexto

Detectado al evaluar `surtgis rasterize` para rasterizar el Inventario Público de Glaciares (IPG 2022, DGA Chile; shapefile EPSG:5360) sobre la grilla de tiles de Sentinel-2 (EPSG:4326) en un proyecto de segmentación de glaciares con deep learning. El flujo: reproyectar IPG 5360→4326, recortar al tile, rasterizar a máscara binaria de glaciar.
