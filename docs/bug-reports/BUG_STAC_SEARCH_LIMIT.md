---
De: Sesión de procesamiento postdoc (15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-13
Prioridad: ALTA (cuencas grandes pierden tiles al oeste)
---

# Bug: search_limit=1000 insuficiente para cuencas grandes → tiles faltantes

## Síntoma

Cuenca 03 (Costeras Loa-Caracoles): triángulo de 0s en la zona costera sur (rows 4100+). El DEM tiene tierra pero el composite S2 no tiene datos. Afecta ~52km de ancho × ~120km N-S.

## Causa raíz

La cuenca tiene 6 tiles MGRS (KCR, KDR, KCQ, KDQ, KCP, KDP) × 152 fechas ≈ **~990 items** en STAC. El `search_limit=1000` captura justo 1000 items, pero el sort de Planetary Computer es por fecha descendente. Las 50 fechas seleccionadas pueden no tener **todos los 6 tiles** — los tiles de la columna C (oeste, que cubren la costa) pueden haberse cortado.

Evidencia:
- `Found 1000 items across 152 dates (using 50 dates)` → hit exacto del límite
- Tile 19KCR (x=[300000, 409800], y=[7490200, 7600000]) **existe y cubre el gap** verificado con rasterio
- El gap corresponde exactamente a la zona que cubriría la columna C de tiles pero no la columna D

## Cómo reproducir

```bash
surtgis stac composite --catalog=pc \
    --bbox="-70.6273,-23.6496,-69.7553,-21.4434" \
    --collection=sentinel-2-l2a --asset=B04 \
    --datetime="2022-01-01/2024-12-31" --max-scenes=50 \
    --align-to=factors/03_costeras_loa_caracoles/dem_30m.tif \
    test.tif
```

Row 4100+ tiene 0% de datos en la zona oeste donde debería haber territorio.

## Fix propuesto

### Opción A: Aumentar search_limit dinámicamente

La fórmula actual (L919):
```rust
let search_limit = ((estimated_spatial_tiles * max_scenes * 5).max(1000)) as u32;
```

Con 6 tiles × 50 escenas × 5 = 1500. Pero `estimated_spatial_tiles` se calcula con 60km spacing, que puede subestimar. Para cuenca 03 (92km × 245km):
- tiles_x = ceil(92/60) = 2
- tiles_y = ceil(245/60) = 5
- estimated = 10 tiles

10 × 50 × 5 = 2500 → debería ser suficiente. **¿Por qué se usó 1000?**

Posible issue: el bbox en grados se multiplica por 111 km/° uniformemente, pero a latitudes altas 1° de longitud es <111km. Para lat=-22°, 1°lon ≈ 103km. Eso no cambia mucho el cálculo.

**Verificar** que la fórmula está generando el search_limit correcto para este bbox. El log solo dice "Found 1000 items" — ¿cuál fue el search_limit calculado?

### Opción B: Post-filtro por cobertura

Después de seleccionar las 50 fechas, verificar que cada fecha tiene tiles cubriendo el bbox completo. Si una fecha tiene tiles solo en la mitad este, completarla buscando items adicionales de esa fecha para la zona oeste.

### Opción C: Búsqueda por tile MGRS

En vez de buscar todo el bbox de una vez, identificar los tiles MGRS que cubren el bbox y hacer una búsqueda por tile:

```
for each MGRS tile intersecting bbox:
    search STAC with tile's bbox + datetime
    take max_scenes dates
```

Luego unir las fechas comunes. Esto garantiza cobertura espacial completa.

## Datos del caso

- DEM: 3054×8168 px, EPSG:32719, x=[331358, 422978] y=[7383573, 7628613]
- Tiles MGRS: 19KCR, 19KDR (norte), 19KCQ, 19KDQ (centro), 19KCP, 19KDP (sur)
- Gap: tile 19KCR (columna C) faltante en strips 9-16, solo tiles columna D presentes
- Tile 19KCR bounds: x=[300000, 409800] y=[7490200, 7600000] — cubre el gap completamente
