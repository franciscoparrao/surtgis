---
De: Sesión postdoc (procesamiento 15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-22
Versión: v0.6.23 (commit 0d94d17 — "stac composite RAM — per-scene tile cache accounting")
Env: SURTGIS_RAM_BUDGET_GB=12 (target conservador)
Prioridad: ALTA (misma magnitud de error que v0.6.22)
---

# v0.6.23: presupuesto analítico sigue subestimando ~53% (idéntico patrón que v0.6.22)

## Resumen

Configuramos `SURTGIS_RAM_BUDGET_GB=12`. El auto-cap reportó **11.9 GB peak**. Pusimos un watchdog externo con umbral 18 GB. **A los 6 minutos el proceso cruzó 18 GB y el watchdog lo mató.** Misma cuenca Maule + Earth Search.

## Timeline RAM (watchdog cada minuto)

```
02:06:59 watchdog arranca: threshold=18GB poll=10s
02:11:00 PID=2335190 RAM=14.9GB OK
02:12:00 PID=2335190 RAM=16.0GB OK  (+1.1 GB/min)
02:13:00 PID=2335190 RAM=17.4GB OK  (+1.4 GB/min)
02:13:10 ⚠ PID=2335190 RAM=18.2GB > 18GB → KILL -9
```

Crecimiento **lineal sostenido de ~1.1-1.4 GB/min** — no un spike, acumulación progresiva.

## Output del inicio (fix v0.6.23)

```
⚠ strip_rows capped: 512 → 83 (Earth Search (1024² COG tiles, heavy reprojection)): accumulator 12.5 GB → 2.0 GB
  RAM budget (12.0 GB target) — output: 3.3 GB | concurrent decode: 0.3 GB | per-scene tile cache: 6.3 GB | accumulator: 2.0 GB (strip_rows=83) → ~11.9 GB peak
```

Budget predicho: **11.9 GB**
Real observado (al kill): **18.2 GB**
**Exceso: +53%** — idéntico al +55% de v0.6.22.

## Patrón histórico

| Versión | Budget | Real observado | Factor |
|---------|--------|----------------|--------|
| v0.6.19 | 4.3 GB  | >30 GB (freeze) | ~7× |
| v0.6.22 | 12.9 GB | 20 GB          | +55% |
| **v0.6.23** | **11.9 GB** | **18.2 GB**    | **+53%** |

El fix de v0.6.23 agregó el cache per-scene al modelo, que explica la diferencia v0.6.19→v0.6.22 (de 7× a +55%). Pero **entre v0.6.22 y v0.6.23 el ratio NO mejoró** — ~53% ≈ ~55%. Algo sistemático sigue sin contabilizarse.

## Pistas del log

El output muestra alternancia entre tiles 1024² y tiles 512² en la misma pasada:
```
[cog] assembled: 30 tiles (30+0 skip), output=4944x793, tw=512 th=512
[cog] assembled: 9 tiles (9+0 skip), output=8207x270, tw=1024 th=1024
```

Sospecha: ¿están conviviendo buffers de ambos tamaños simultáneamente (ambos catálogos/bandas mezclando)? O ¿buffers de predictor `pred=2 bps=16 tw=1024 raw_len=2097152` que son 2 MB cada uno y se apilan?

Además hay muchas líneas `[pred] undo horiz-diff: bps=2 tw=1024 len=2097152` — cada una es un buffer de decodificación de 2 MB que si no se libera inmediatamente se acumula.

## Hipótesis estructurales

1. **Múltiples escenas en vuelo**: quizás el outer loop inicia descarga de la fecha N+1 antes de que la fecha N termine de drenar todos los tiles decodificados. El chunking de 4 tiles ayudó intra-escena, pero inter-escena no.

2. **Buffers predictor no liberados**: los `[pred] undo horiz-diff` pueden estar acumulándose si se clonan antes de ser consumidos.

3. **Cache de mosaico por banda duplicado**: 10 bandas × buffers intermedios × strips en vuelo = explosión.

## Sugerencia pragmática

Dado el patrón lineal de crecimiento (~1.2 GB/min), no parece salvable con constantes calibradas. Las opciones reales:

**A. Refactor banda-por-banda** (lo mencionaron): procesar 1 banda a la vez, una fecha a la vez. Elimina el cache multi-banda intermedio. Trade: 10× más HTTP requests pero bounded RAM.

**B. Hard cap con backpressure real**: `Arc<Semaphore>` sobre bytes decodificados totales. Cualquier `decode()` que haría superar `target` × 1.0 se bloquea hasta que otro libere. No depende del modelo analítico.

**C. Flag experimental `--ram-mode=strict`** que usa B, y deja el modo actual como fallback optimista.

Preferimos B o C.

## Workaround usado

Watchdog externo mata surtgis + script padre si cruza 18 GB, previene freeze. `/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad/ram_watchdog.sh`. Pero solo aborta, no permite completar.

## Pedido

Si la opción B (semáforo sobre bytes decodificados) es factible en ~horas, lo esperamos. Si requiere refactor grande (días), consideremos cerrar las 5 cuencas restantes con PC + paciencia con rate limits (v0.6.19 antes del freeze procesaba PC estable a ~8-12 GB, solo era LENTO por 503/429, no explotaba).
