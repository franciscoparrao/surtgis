---
De: Sesión postdoc (susceptibilidad Río Maule, paper 1)
Para: Sesión SurtGIS
Fecha: 2026-05-03
Versión: surtgis 0.7.0 (idéntica funcionalmente a v0.6.28)
Catálogo: Microsoft Planetary Computer
Prioridad: MEDIA (no bloquea — la corrida avanza estable, pero el budget predicho subestima la realidad y eso induce decisiones operativas erróneas en sistemas con poco headroom)
---

# v0.7.0: budget predicho 8.1 GB peak vs RSS real 10–12.6 GB en `stac composite` Maule (PC)

## Resumen

Corrida actualmente en curso (PID 2811009, 23+ h elapsed, strip 4/12). El refactor band-outer + mimalloc de v0.6.26 **funciona** — no hay leak progresivo como en v0.6.22–24, el RSS oscila acotado entre strips. Sin embargo:

1. El **peak logueado real (~12.6 GB) excede el peak predicho (8.1 GB) en +56%**, y el **peak real-time observado (13.1 GB)** lo excede en **+62%**.
2. El **floor inter-strip (~10 GB)** excede el **mask cache predicho (3.0 GB) en >3×**.
3. **Strips pares (2, 4) muestran peak transitorio en chunk [0..1]** (~12.3–13.1 GB) que **strips impares (1, 3) no muestran** (~10.1 GB peak). Patrón alternante reproducible.
4. El log "after Phase A (masks)" reporta RSS muy variable (8.8–11.4 GB), sugiriendo que Phase A además **libera** buffers del strip anterior (lo cual sería deseable pero no está documentado).
5. **El peak real está modulado por la presión externa de RAM/swap**. En este sistema, cuando había swap saturado y poco free, SurtGis se contuvo a 12.6 GB en strip 2; al liberar 2 GB de presión externa (cierre de Chrome + apps), SurtGis se expandió a **13.1 GB en strip 4 con el mismo evento del log**. Esto sugiere que el budget interno **no es estrictamente respetado**: el límite real lo impone el kernel + swap pressure, no la lógica de SurtGis.
6. **El log subreporta el peak real**. El evento `chunk [0..1] end` para strip 4 logueó 12,286 MB, pero el RSS real-time durante ese chunk llegó a **13,101 MB** (medido por `ps -o rss` ~10 s antes del log event). Diferencia +815 MB. Los snapshots `[ram]` no capturan el peak intra-chunk.

Estas discrepancias hacen que el budget impreso al inicio del job sea poco útil para planificar capacity en sistemas compartidos. Un usuario con 16 GB libres confiaría en "8.1 GB peak" y se llevaría sorpresas con OOM (de hecho, en mi sistema un primer intento OOMó porque tenía 11 GB libres de los 38 totales y SurtGis llegó a 12.6 GB).

## Comando

```bash
SURTGIS_RAM_BUDGET_GB=16 FORCE_CATALOG=pc \
surtgis --compress stac composite \
  --catalog=pc \
  --bbox=-72.687,-36.579,-70.319,-35.080 \
  --collection=sentinel-2-l2a \
  --asset=B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 \
  --datetime=2022-01-01/2024-12-31 \
  --max-scenes=50 \
  --naming=asset \
  --align-to=dem_30m.tif \
  composite.tif
```

- BBOX: 2.4° × 1.5° (~211 × 167 km, Río Maule, Chile).
- Output grid: 7274 × 5725 = 41.6 M cells, 10 bandas, 50 escenas.
- Tiles COG: 512 × 512 (PC y ES son ambos 512² en `sentinel-2-l2a`, contrario a una creencia común).

## Predicción del budget al arranque

```
RAM budget (16.0 GB target, band_chunk_size=1):
  output:        3.3 GB
  mask cache:    3.0 GB
  scene strips:  1.5 GB
  band working:  0.2 GB
  decode:        0.1 GB    (strip_rows=512)
  ──────────────────────
  → ~8.1 GB peak
```

`outer-band, 1-band chunks; HTTP requests per scene = 10`.

## RSS observado (ground truth)

Datos crudos del log de SurtGis para los primeros 4 strips. Cada fila es un evento `[ram] …`.

### Strip 1 (warm-up)

| Evento | RSS (MB) | Δ vs anterior |
|---|---:|---:|
| baseline before strip loop | 3,721 | — |
| strip 1/12 start | 3,560 | −161 |
| **strip 1/12 after Phase A (masks)** | **8,814** | **+5,254** |
| chunk [0..1] start | 8,814 | 0 |
| chunk [0..1] end | 10,114 | +1,300 |
| chunks [1..2] … [9..10] end | 9,330–10,115 | oscila |
| strip 1/12 chunk [9..10] end | 10,074 | — |

Peak strip 1: **10,115 MB** (chunk [5..6] end / [6..7] start).

### Strip 2 (peak transitorio anómalo)

| Evento | RSS (MB) | Δ |
|---|---:|---:|
| strip 2/12 start | 10,074 | 0 vs strip 1 end |
| **strip 2/12 after Phase A (masks)** | **11,376** | **+1,302** |
| chunk [0..1] start | 11,376 | 0 |
| **chunk [0..1] end** | **12,639** | **+1,263** ← peak |
| chunk [1..2] end | 12,636 | −3 |
| chunks [2..3]…[9..10] end | 12,259–12,632 | oscila estable |
| strip 2/12 chunk [9..10] end | 12,315 | — |

Peak strip 2: **12,639 MB** (chunk [0..1] end). **+2.5 GB vs peak strip 1 sin razón aparente**.

### Strip 3 (drop inesperado en Phase A)

| Evento | RSS (MB) | Δ |
|---|---:|---:|
| strip 3/12 start | 12,315 | 0 vs strip 2 end |
| **strip 3/12 after Phase A (masks)** | **8,878** | **−3,437** ← drop |
| chunk [0..1] start | 8,878 | 0 |
| chunk [0..1] end | 10,065 | +1,187 |
| chunks [1..2]…[9..10] | 10,065–10,117 | oscila estable |
| strip 3/12 chunk [9..10] end | 10,083 | — |

Peak strip 3: **10,117 MB**. Phase A bajó el RSS en 3.4 GB, en lugar de subirlo (Phase A en strips 1, 2, 4 sube +1.3 a +5.3 GB).

### Strip 4 (replica el patrón de strip 2 — peak transitorio recurrente)

| Evento | RSS logueado (MB) | RSS real-time observado (MB) | Δ |
|---|---:|---:|---:|
| strip 4/12 start | 10,083 | — | 0 vs strip 3 end |
| **strip 4/12 after Phase A (masks)** | **11,382** | — | **+1,299** |
| chunk [0..1] start | 11,382 | — | 0 |
| (durante chunk [0..1], medido en vivo) | — | **13,101** | **+1,719** sobre log inicial |
| **chunk [0..1] end** | **12,286** | — | +904 (sobre start, pero ya bajó del peak intra-chunk) |
| chunk [1..2] start | 12,286 | — | 0 |
| (post chunk [1..2] start, vivo) | — | 11,571 | −715 ya empezó a liberar |

**Observación clave**: Strip 4 replica el patrón de Strip 2 (peak transitorio en chunk [0..1]), confirmando que el patrón es **alternante par/impar**, no anomalía única de strip 2.

**Peak real-time NO logueado**: 13,101 MB en strip 4 chunk [0..1]. El log reporta 12,286 MB en `chunk [0..1] end`, **subreportando 815 MB del peak real**.

**Diferencia con strip 2**: el sistema tenía menos headroom durante strip 2 (swap saturado + Chrome 21 GB) y SurtGis se contuvo en 12.6 GB. Para strip 4 el usuario había cerrado apps (swap drenándose, +2 GB free), y SurtGis se expandió a 13.1 GB. Sugiere fuertemente que **el peak es modulable por presión kernel** y no estrictamente por el budget interno.

## Análisis

### 1. Discrepancia budget vs realidad

| Componente | Predicho | Observado (estimado) |
|---|---:|---:|
| output buffer | 3.3 GB | ~3.3 GB ✓ |
| mask cache | 3.0 GB | **~5–6 GB** (estimado por delta Phase A strip 1: +5.3 GB) |
| scene strips | 1.5 GB | ~1.0–1.5 GB ✓ |
| band working | 0.2 GB | ✓ |
| decode | 0.1 GB | ✓ |
| **allocator overhead (mimalloc retained pages)** | **no contabilizado** | **+1.5–3 GB estimado** |
| **Peak total** | **8.1 GB** | **10.1–12.6 GB** |

El `mask cache` real parece ser ~2× el predicho, y el budget no contabiliza el overhead típico de mimalloc (que retiene páginas para reuso).

### 2. Peak transitorio par/impar (strips 2 y 4 confirmados)

Por qué strips pares (2, 4) llegan a 12.3–13.1 GB en chunk [0..1] y strips impares (1, 3) no superan 10.1 GB es **el dato más interesante** de este reporte. Con strip 4 confirmado, ya no es anomalía única; es **patrón reproducible**.

Hipótesis:

- (a) **Overlap temporal**: durante el primer chunk de strips pares, las masks o buffers de scene strips del strip impar anterior todavía no se liberaron mientras se cargan los nuevos (Phase A reporta +1.3 GB en strips 2/4, vs +5.3 GB en strip 1 partiendo de baseline limpia).
- (b) **Doble buffering implícito**: SurtGis quizá usa dos buffers de scene strips alternándolos por strip (par/impar), y el segundo buffer se aloca recién en el primer chunk del strip 2, sumándose al primero antes de que mimalloc libere.
- (c) **Allocator (mimalloc)**: puede retener páginas al final de strip 1 que se acumulan en strip 2; al entrar a strip 3 el allocator finalmente las devuelve. Esto explicaría el drop de 3.4 GB en Phase A del strip 3.

La hipótesis (b) tiene fuerza adicional con el dato de strip 4: si fuera solo allocator, el drop debería haber ocurrido también entre strip 2→3 lo suficiente como para que strip 4 no replicara el peak. Pero strip 4 sí replica → hay un patrón estructural par/impar, no solo un artefacto de allocator.

### 2b. Peak real modulado por presión externa

Dato nuevo (no en versión 1 del reporte):

- **Strip 2 chunk [0..1] end (logueado)**: 12,639 MB. Sistema tenía swap 100% + Chrome 21 GB → free 11 GB.
- **Strip 4 chunk [0..1] (vivo)**: **13,101 MB**. Sistema tenía swap drenándose + Chrome cerrado → free 6 GB.
- **Strip 4 chunk [0..1] end (logueado)**: 12,286 MB. Cuando el log capturó el evento, el peak ya había pasado.

**Lectura**: SurtGis tiene un peak interno **fluctuante** que depende de cuánto puede allocar antes de que el kernel le ponga back-pressure. En sistemas con poco free + swap saturado, el kernel induce throttling de page allocs (forzando page-ins lentos), y SurtGis se "queda" en ~12.6 GB. En sistemas con headroom, SurtGis se expande hasta ~13.1 GB+.

**Implicancia para usuarios**: el peak observado en una corrida exitosa NO es un techo confiable para planificar capacity en otro sistema. Si SurtGis corre cómodo con 16 GB libres, podría estar consumiendo más cerca de 13–14 GB que de los 12.6 GB del log.

### 3. NO hay leak progresivo (✓)

A diferencia de v0.6.22–24, el RSS NO crece linealmente con strips. Confirmado de strip 1 → 4:

```
strip 1 end:  10,074 MB
strip 2 end:  12,315 MB  (+2.2)   ← peak transitorio (strip par)
strip 3 end:  10,083 MB  (−2.2)   ← regresa al floor (strip impar)
strip 4 start: 10,083 MB
strip 4 chunk[1..2] start: 12,286 MB  ← otra vez peak (strip par, patrón confirmado)
```

El refactor band-outer + mimalloc fix funciona. No reabro un BUG por leak, este reporte es solo sobre la calibración del budget impreso y la modulación por presión externa.

## Pregunta principal: ¿es factible bajar el peak a poca RAM?

Análisis de los componentes irreducibles para `stac composite` con median compositing:

| Componente | Mínimo teórico | Reducible cómo |
|---|---|---|
| output buffer | 7274×5725×10 bandas × 4 B = **1.66 GB** float32 | Stream a disco banda-por-banda → ~0.17 GB (1 banda en RAM); requiere reescribir el output como banda-outer. |
| mask cache | Si masks son u8 (SCL 0–255): 50 × 7274 × 5725 × 1 B = **2.0 GB** | Bitset (1 bit por píxel válido por escena) → 256 MB. O recomputar masks por strip → 0 MB pero +50 fetches/strip. |
| scene strips | 50 escenas × 512 rows × 7274 × 2 B = **374 MB** por band | OK como está. Es el componente más eficiente. |
| band working + decode | <0.5 GB | OK. |
| allocator overhead | +20–40% del activo | Cambiar mimalloc → jemalloc con `MALLOC_CONF=background_thread:true,dirty_decay_ms:1000` libera más agresivamente. |

**Mínimo teórico realizable**: ~3–4 GB peak (con bitset masks + streaming output + jemalloc + strip_rows=512). Pero requiere refactor no trivial de la representación de mask cache y del output writer.

**Mínimo realista con la arquitectura actual**: ~6–8 GB peak. El budget predicho (8.1 GB) ya está cerca de eso. El problema es la **subestimación**, no la imposibilidad.

**Si el usuario fuerza budget bajo (ej. 4 GB)**: SurtGis cap-ea `strip_rows` (en el caso de Maule, de 512 a 65), produciendo **89 strips en lugar de 12**. Esto multiplica por ~7 los HTTP requests (50 escenas × 10 bandas × 89 strips = 44,500 GETs, vs 6,000 con strip_rows=512), y la corrida pasa de "lenta" a "inviable" (~día y medio adicional solo por overhead de red).

**Conclusión**: la corrida con poca RAM **no es imposible**, pero la curva costo-beneficio es muy desfavorable. La banda 4–8 GB es razonable; bajar de 4 GB es masoquismo.

## Recomendaciones para SurtGis

### Para el budget impreso (alta prioridad, baja complejidad)

1. **Calibrar el coeficiente de mask cache**. El `3.0 GB` parece subestimar por ~2×. Si la representación interna de mask es `Vec<u8>` (no bitset), el cálculo debería ser `n_scenes × output_w × output_h × 1`, no algo más comprimido. Para Maule: 50 × 7274 × 5725 × 1 B = 2.0 GB ya es cercano a lo predicho, pero el delta Phase A (+5.3 GB) sugiere que hay una segunda copia o un staging buffer no contabilizado.

2. **Sumar un término de allocator overhead** al peak predicho. Algo como `predicted_peak * 1.3` con mimalloc, o incluso tomar el dato real del primer strip y reportar "predicted peak = 8.1 GB | observed first-strip peak = 10.1 GB" en el log para que el usuario calibre.

3. **Reportar el peak real-time**, no solo el RSS al cierre de cada chunk. Sugerencia: spawn un thread watchdog interno que muestree `getrusage(RUSAGE_SELF).ru_maxrss` cada 5 s y emita una línea `[ram] peak_intra_chunk: NNNN MB` al cerrar el chunk. Esto cerraría la brecha entre log (12.3 GB) y observación externa (13.1 GB).

4. **Documentar el comportamiento "Phase A libera"** si es intencional. Hoy `after Phase A (masks)` da deltas tan distintos como −3.4 GB y +5.3 GB; si la fase realmente hace teardown de masks anteriores antes de cargar las nuevas, el log debería decirlo (ej. dos eventos: `phase A teardown: RSS=X` y `phase A masks loaded: RSS=Y`).

5. **Advertir sobre modulación por presión externa**. Una nota al final del bloque budget: `actual peak may be 5–10% higher than predicted on systems with abundant free RAM (mimalloc may retain more pages); 5–10% lower on memory-pressured systems (kernel throttles allocations)`. Esto desactiva sorpresas en deploys diversos.

### Para el peak transitorio par/impar (media prioridad, alta complejidad)

6. **Investigar por qué strips pares (2, 4) llegan a 12.3–13.1 GB y strips impares (1, 3) no superan 10.1 GB**. Confirmado patrón alternante reproducible. Si es doble-tenencia de masks o doble buffering implícito de scene strips, tear-down explícito al cierre de cada strip eliminaría 2–3 GB de peak. Alternativas: `mimalloc::collect(true)` o `malloc_trim` al cierre de cada strip; o bien forzar liberación de buffers de scene strips antes de allocar los nuevos.

### Para usuarios con poca RAM (baja prioridad)

7. **Implementar streaming de output** banda-por-banda al disco (bajaría peak ~1.5 GB) detrás de un flag opcional `--stream-output`. Útil para sistemas <8 GB.

8. **Evaluar bitset masks** (1 bit por píxel válido): bajaría 2 GB de peak. Solo viable si todas las máscaras se reducen a binario (válido/inválido), lo cual es el uso típico para median compositing.

### Para diagnóstico (baja prioridad, baja complejidad)

9. **Emitir métrica de presión externa** al final del job. Algo como: `system free at end: X GB | swap used: Y GB | warning: actual peak may have been throttled by kernel pressure`. Permite distinguir corridas "limpias" de corridas "presionadas" al post-mortem.

## Anexo: timeline elapsed vs throughput

- 22:50 (2 mayo) → arranque
- 09:26 (3 mayo, +10h36m) → strip 2/12 chunk [5..6]
- 11:03 (3 mayo, +12h13m) → strip 2/12 chunk [8..9]
- 15:40 (3 mayo, +16h50m) → strip 3/12 chunk [2..3]
- 21:27 (3 mayo, +22h37m) → strip 4/12 chunk [0..1]

**Promedio: ~6 h por strip. Proyección total: ~72 h** para 12 strips. Compárese con la corrida con Earth Search del 25 abril: ~12 h por strip (PC ~2× más rápido en la misma máquina).

## Archivos relacionados

- Log SurtGis: `papers/paper1_susceptibilidad/factors/11_rio_maule/spectral/download.log` (5.4 MB en este momento, modificado en tiempo real).
- Wrapper: `papers/paper1_susceptibilidad/download_multiband.sh` (modificado para soportar `FORCE_CATALOG=pc|es`, backup en `download_multiband.sh.bak_pre_force_pc`).
- Reportes RAM previos relacionados:
  - `BUG_RAM_COMPOSITE_MAULE.md` (v0.6.19, 24 GB sin terminar — ya resuelto)
  - `BUG_RAM_V0622_STILL_OVER.md`, `BUG_RAM_V0623_STILL_OVER.md` (leaks lineales — ya resueltos)
  - `BUG_RAM_V0624_LATE_GROWTH.md` (leak con punto de inflexión — ya resuelto)
- Decisión clave registrada en contexto persistente postdoc: "Forzar Planetary Computer (no ES)" — confirmado empíricamente en esta corrida.
