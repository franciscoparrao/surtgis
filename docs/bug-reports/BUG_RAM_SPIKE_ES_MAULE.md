---
De: Sesión postdoc (procesamiento 15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-21
Versión: v0.6.19 (commit 11ba36d "Fix stac composite RAM: auto-cap strip_rows + double-buffer spatial fill")
Prioridad: ALTA (congela máquina de 38GB RAM)
---

# Bug: RAM auto-cap subestima memoria con Earth Search (tiles 1024×1024)

## Síntoma

En cuenca Maule (~7274×5725 px) con Earth Search, la RAM subió abruptamente de ~30% a 100% congelando el computador (38GB). El fix de auto-cap reportó:

```
⚠ strip_rows capped: 512 → 175 (requested peak ~12.5 GB, using ~4.3 GB for 10 bands × 42 scenes × 7274 cols)
```

Peak estimado por el fix: **4.3 GB**. Peak real: **>30 GB** (hasta congelar).

## Raíz probable

Earth Search devuelve tiles de **1024×1024** (vs 512×512 en Planetary Computer):

```
[cog] pred=2 comp=8 bps=16 sf=1 raw_len=2097152 expected=2097152
[cog] assembled: 30 tiles (30+0 skip), output=9904x1861, tw=1024 th=1024
```

Comparado con PC en la misma cuenca:
```
[cog] pred=1 comp=8 bps=15 sf=1 raw_len=491520 expected=524288
[cog] tile(...) expected 262144 px, got 245760 (raw=491520 bytes, bps=15 tw=512 th=512)
```

**Tile raw ES**: 2,097,152 bytes = **4.27× más grande** que PC (491,520 bytes).

La fórmula actual del auto-cap considera `cols × bands × scenes × bytes_per_pixel` pero no ajusta por el tamaño de tile que el catálogo devuelve. Con tw=1024:
- Buffer por tile: 4× más grande
- Mask buffer: 4× más grande
- Número de tiles simultáneos en flight durante descarga concurrente se multiplica

## Dato concreto por escena

Una sola fecha genera cargas grandes:
```
ℹ 2024-11-03: tiles OK=45 outside=120 partial=45 mask=45
```

45 tiles × 10 bandas × (1024² × 2 bytes) = 922 MB solo para una fecha si se mantienen todos en memoria.

## Con PC (binario idéntico, misma cuenca)

Funciona sin congelar, estable en ~8-12 GB durante 20h (aunque lento por rate limits).

## Fix sugerido

El auto-cap debería:
1. Conocer (o estimar) el `tile_width` esperado del catálogo — ES=1024, PC=512
2. Ajustar `strip_rows` según: `strip_rows = RAM_target / (tile_width² × bytes × scenes × bands × concurrency)`
3. Idealmente detectar `tw` al descargar el primer tile y **re-ajustar strip_rows dinámicamente**

O más simple: cap agresivo cuando `catalog == EarthSearch`:
```rust
let ram_multiplier = match catalog {
    StacCatalog::EarthSearch => 4.0,  // tiles 1024 vs 512
    _ => 1.0,
};
let effective_target = ram_target / ram_multiplier;
```

## Contexto operacional

Nos vimos forzados a usar ES porque PC está rate-limiting la IP (2,756 × 503 y 324 × 429 errores en 20h sobre Maule). ES era la alternativa pero causó el freeze. Quedamos bloqueados para procesar las 5 cuencas restantes (Maule, Biobío, Bueno, Costeras Bueno-Puelo, Magallanes).

Workaround manual: `--strip-rows=64` pero no hemos probado si elimina el spike.

## Log del inicio

```
Multi-band composite: 10 bands [blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22]
📷 Collection: Sentinel-2 L2A (mask: Some("scl"))
  bbox: 2.4°×1.5° (~15 spatial tiles), search_limit=3750
Found 3750 items across 42 dates (using 42 dates)
  Resolved 42 scenes with 10 bands each
  Using reference grid from --align-to: 7274x5725 px=(30.0,-30.0)
Output grid: 7274 x 5725 (41.6M cells), 42 dates, 10 bands
⚠ strip_rows capped: 512 → 175 (requested peak ~12.5 GB, using ~4.3 GB for 10 bands × 42 scenes × 7274 cols)
```

Peak reportado por fix: 4.3 GB. Peak real: >30 GB (freeze).
