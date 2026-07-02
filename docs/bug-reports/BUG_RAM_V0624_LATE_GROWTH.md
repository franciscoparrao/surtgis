---
De: Sesión postdoc (procesamiento 15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-22
Versión: v0.6.24 (commit 242465b — "stac composite outer-band refactor for bounded per-band RAM")
Env: budget default 16 GB
Prioridad: ALTA (refactor estable 22 min, luego leak lineal)
---

# v0.6.24: estable 22 min, luego crecimiento lineal hasta crash

## Conclusión corta

**Refactor funciona en estado estacionario** (~5-6 GB durante 22 min).
**Pero algo se acumula después del minuto 22** y crece lineal hasta saturar. Watchdog lo mató a 14.1 GB.

Es un patrón **cualitativamente distinto** a v0.6.22/23 (que crecían desde el inicio). Ahora hay **un punto de inflexión** que sugiere transición de fase/strip.

## Timeline exacta (watchdog cada minuto)

```
10:32:04 RAM=5.9GB   ← arranque, budget predicho 10.1 GB
10:33:04 RAM=5.4GB
10:34:04 RAM=4.9GB
10:35:04 RAM=6.8GB   
10:36:05 RAM=6.2GB  ┐
10:37:05 RAM=5.7GB  │
10:38:05 RAM=5.1GB  │
10:39:05 RAM=7.0GB  │
10:40:05 RAM=6.2GB  │
10:41:05 RAM=5.6GB  │
10:42:05 RAM=5.1GB  │
10:43:06 RAM=6.6GB  │ ~22 min estables
10:44:06 RAM=5.3GB  │ oscilación 4.9-7.0 GB
10:45:06 RAM=5.5GB  │ HEALTHY
10:46:06 RAM=5.5GB  │
10:47:06 RAM=5.6GB  │
10:48:06 RAM=5.5GB  │
10:49:07 RAM=5.9GB  │
10:50:07 RAM=5.7GB  │
10:51:07 RAM=5.7GB  │
10:52:07 RAM=5.8GB  │
10:53:07 RAM=6.4GB  ┘
───────────────────── punto de inflexión
10:54:07 RAM=7.7GB    +1.3 GB
10:55:07 RAM=8.6GB    +0.9 GB
10:56:08 RAM=9.9GB    +1.3 GB
10:57:08 RAM=11.2GB   +1.3 GB
10:58:08 RAM=12.6GB   +1.4 GB
10:59:08 RAM=13.9GB   +1.3 GB
10:59:18 ⚠ 14.1GB > 14GB → KILL -9
```

**Ritmo del leak**: ~1.3 GB/min constante (idéntico al que tenía v0.6.23, pero empieza 22 min después).

## Output del inicio

```
Multi-band composite: 10 bands [blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22]
📷 Collection: Sentinel-2 L2A (mask: Some("scl"))
Found 3750 items across 42 dates (using 42 dates)
Output grid: 7274 x 5725 (41.6M cells), 42 dates, 10 bands
RAM budget (16.0 GB target) — output: 3.3 GB | mask cache: 5.0 GB | scene strips: 1.3 GB | band working: 0.4 GB | decode: 0.1 GB (strip_rows=512) → ~10.1 GB peak
```

Se murió con solo 1,515 tiles ensamblados (de >>50,000 esperados) — fase muy temprana. No alcanzó a terminar la primera banda probablemente.

## Hipótesis

Dado que:
1. 22 min estable con RAM ~5-6 GB (el refactor per-banda funciona)
2. Luego crecimiento limpio a 1.3 GB/min
3. La arquitectura nueva es: `for strip { for band { for scene { download + process }}}`

Posibilidades:
- **A. Transición strip 1 → strip 2**: el mask cache de la escena anterior no se libera al pasar a siguiente strip. 5 GB de mask cache × cada strip posterior acumula linealmente.
- **B. Transición banda 1 → banda 2**: el cache de tiles decodificados de la primera banda queda en memoria cuando arranca la segunda. 1.3 GB/min ≈ 10 scenes × ~130 MB por scene = plausible.
- **C. SCL cache acumulativo**: 5 GB de mask cache × 2 (strip 1 + strip 2) ≈ 10 GB adicionales → consistente con el salto.

**Pista**: 1.3 GB/min × 5 min = 6.5 GB. Muy cerca de los 5 GB del "mask cache" presupuestado. Sospecha fuerte en el **mask cache no liberándose en transiciones**.

## Pedido

Revisar el lifecycle del `mask_cache` en el band-outer:
- ¿Se drena al finalizar strip actual antes de arrancar siguiente strip?
- ¿Se drena al pasar de banda N a banda N+1?
- ¿Hay algún `Arc<>` o `HashMap` que retenga referencias más allá de su scope?

**Sugerencia rápida**: agregar un log `▶ strip 2/N starting: mask_cache.len()=..., RAM rss=...` en cada transición. Si el log muestra que el cache crece en cada strip, ya tenemos el culpable.

## Contexto operacional

Procesó 1,515 tiles en 27 min (55 tiles/min). Comparado con v0.6.19 que procesaba ~25 min por strip de 39 strips en PC, el band-outer es razonable en velocidad. **Si el mask cache se libera bien, debería terminar en ~3-5 h** — viable.

## Log del watchdog

```
/tmp/ram_watchdog.log
```

## Estado

Watchdog mató todo limpiamente. Sistema con 24 GB disponibles. No relanzamos hasta fix.
