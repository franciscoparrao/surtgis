---
De: Sesión postdoc (procesamiento 15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-22
Versión: v0.6.22 (commit 31efa35 — "fix stac composite RAM spike on Earth Search")
Prioridad: ALTA (sigue saturando RAM)
---

# v0.6.22: el fix redujo el pico pero aún excede el budget ~1.55×

## Resumen

Con v0.6.22 recién compilada, Maule + Earth Search, el auto-cap reportó presupuesto de **12.9 GB peak**. La RAM real alcanzó **20 GB** antes de que matáramos el proceso (~10 min de corrida). El sistema de 38 GB quedó con solo 5 GB disponibles, forzando el kill antes de freeze.

## Output del inicio (log)

```
⚠ strip_rows capped: 512 → 127 (Earth Search (1024² COG tiles)): accumulator 12.5 GB → 3.1 GB
  RAM budget — output buffers: 3.3 GB | download reserve: 6.4 GB | accumulator: 3.1 GB (strip_rows=127) → ~12.9 GB peak
```

## Peak real medido

`ps aux --sort=-%mem` al momento del kill:
```
surtgis: RAM=20009MB (50.3%)
```

- Budget reportado: **12.9 GB**
- Pico observado: **20.0 GB**
- Exceso: **+7.1 GB (~55%)**

El proceso llevaba ~10 min corriendo, aún en fase temprana de descargas. Posible que siga subiendo.

## Contexto

- Máquina: 38 GB RAM total
- Cuenca: Maule (7274×5725 px)
- Catálogo: Earth Search
- Binario: compilado desde source (commit 31efa35), verificado con `surtgis --version` = 0.6.22

## Hipótesis

El chunking de 4 outer tiles concurrentes ayudó (antes >30 GB, ahora 20 GB) pero algo más está persistiendo:

1. ¿Output buffers realmente 3.3 GB? Para 7274×5725×10 bandas × 8 bytes = 3.33 GB efectivo ✓. Esto no explica el exceso.
2. ¿`download reserve: 6.4 GB` es un límite duro o estimado? Si es estimado, tal vez el working set supere eso en picos.
3. ¿Hay alguna estructura intermedia no contabilizada? Ej: SCL mask buffers × 10 bandas, buffers de reproyección strip-a-strip, caches de predictor pred=2.
4. ¿Múltiples fechas en vuelo simultáneo? Si el outer iteration over dates no es estrictamente secuencial (1 fecha → next), entonces múltiples accumulators pueden apilarse.

## Pregunta

¿El `download reserve` se aplica como límite duro (semáforo/backpressure) o es solo una reserva del budget? Si es reserva pasiva, el consumo real puede superarla libremente.

## Pedido

Si el budget estimado es 12.9 GB pero el real es 20 GB, el factor de corrección es ~1.55. Dos opciones:

A. Agregar factor conservador (`BUDGET_SAFETY_FACTOR = 1.6`) al total para que el auto-cap sea más agresivo.
B. Implementar backpressure duro: semáforo sobre download reserve con tamaño real decodificado, bloqueando nuevas descargas hasta que la cuota libere.

## Workaround temporal

Probaremos manual con `--max-scenes=25` (la mitad) para bajar el budget estimado a ~7 GB. Si aun así supera ~10 GB real, reportamos de nuevo.

## Log completo guardado

`/tmp/maule_es_v0622_bug.log` (sobrescrito por retry PC del script, pero la línea del budget está confirmada arriba).
