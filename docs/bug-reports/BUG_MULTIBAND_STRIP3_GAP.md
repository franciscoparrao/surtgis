---
De: Sesión de procesamiento postdoc (15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-12
Prioridad: CRÍTICA (strip completo vacío en output multi-band)
---

# Bug: Strip 3 vacío (0% valid) en multi-band composite

## Síntoma

`stac composite --asset=B02,B03,...,B12` produce un gap completo en strip 3 (rows 1024-1536). Todas las 10 bandas tienen exactamente el mismo gap. Strips 1-2 y 4-7 están perfectos (100% valid).

```
Strip 1: rows 0-512    → 100.0% valid   ✓
Strip 2: rows 512-1024 → 100.0% valid   ✓
Strip 3: rows 1024-1536 → 0.0% valid    ✗ ← BUG
Strip 4: rows 1536-2048 → 100.0% valid  ✓  (SAS fix funciona!)
Strip 5: rows 2048-2560 → 100.0% valid  ✓
Strip 6: rows 2560-3072 → 100.0% valid  ✓
Strip 7: rows 3072-3348 → 100.0% valid  ✓
```

## Contexto espacial

Strip 3 cubre y=8006462-8021822 (UTM 19S). Los tiles MGRS:

| Tile | Y range | Cubre strip 3? |
|------|---------|----------------|
| 19KDA/19KCA (norte) | 7990240-8100040 | **SÍ** (overlap 15.4 km, strip completo) |
| 19KDV/19KCV (sur) | 7890220-8000020 | **NO** (gap 6.4 km, top=8000020 < strip bottom=8006462) |

Strip 3 debería funcionar solo con tiles del norte. Los tiles del sur correctamente retornan `BBoxOutside`.

## Causa probable: consistencia espacial multi-band

El spec de multi-band composite incluía:

> "si alguna banda falla para un tile MGRS, se descarta ese tile para TODAS las bandas"

**Hipótesis**: La lógica de consistencia ve que los tiles del sur (KCV/KDV) retornan `BBoxOutside` para sus 10 bandas en strip 3. Esto podría estar marcando esa **escena completa** como "inconsistente" y descartándola, arrastrando también los tiles del norte (KDA/KCA) que sí tenían datos.

Dicho de otra forma: para strip 3, cada escena tiene ~8 tiles. 4 tiles son del norte (datos OK) y 4 del sur (BBoxOutside). Si la lógica interpreta los BBoxOutside del sur como "fallo de banda" y descarta la escena entera, strip 3 queda sin datos.

## Fix propuesto

Un `BBoxOutside` no es un fallo de banda — es un tile que no intersecta este strip geográficamente. La lógica de consistencia debería:

1. **Ignorar tiles con BBoxOutside** al evaluar consistencia (no contarlos como fallo)
2. Solo descartar un tile si una banda retorna datos y otra no (para el mismo tile MGRS)
3. Tiles donde NINGUNA banda retorna datos (BBoxOutside para todas) simplemente se omiten sin afectar la decisión de consistencia

Pseudo-código:
```rust
// ANTES (probable implementación actual):
// Si algún tile falla para alguna banda → descartar escena
if any_band_failed_for_any_tile { discard_scene(); }

// DESPUÉS:
for tile in scene.tiles {
    let band_results: Vec<Option<Raster>> = download_all_bands(tile);
    let has_data: Vec<bool> = band_results.iter().map(|r| r.is_some()).collect();
    
    if has_data.iter().all(|&d| !d) {
        // BBoxOutside para todas las bandas → tile no cubre este strip, ignorar
        continue;
    }
    if has_data.iter().any(|&d| !d) {
        // Algunas bandas sí, otras no → inconsistencia real → descartar tile
        discard_tile_for_all_bands();
    }
    // Todas las bandas OK → usar tile
}
```

## Verificación

El test single-band (sin consistencia) con `--max-scenes=5` produjo strip 3 con datos correctos. El bug es específico del path multi-band.

## Datos de reproducción

```bash
surtgis --compress stac composite \
    --catalog=pc \
    --bbox="-70.3336,-18.5167,-69.354,-17.6135" \
    --collection=sentinel-2-l2a \
    --asset=B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 \
    --datetime="2022-01-01/2024-12-31" \
    --max-scenes=50 \
    --align-to=factors/01_rio_lluta/dem_30m.tif \
    output/composite.tif
```

## Archivos relevantes

- `crates/cli/src/handlers/stac.rs`: lógica multi-band en Fase 3 — buscar dónde se aplica la consistencia espacial entre bandas y cómo se manejan los BBoxOutside
