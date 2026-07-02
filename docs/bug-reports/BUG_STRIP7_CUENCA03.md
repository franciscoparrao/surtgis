---
De: Sesión de procesamiento postdoc
Para: Sesión SurtGis
Fecha: 2026-04-13
Prioridad: ALTA (mismo bug que v0.6.4 recurrente)
---

# Bug: Strip 7 vacío (0%) en cuenca 03 — regresión del fix v0.6.4

## Síntoma

Cuenca 03 (v0.6.10): strip 7 (rows 3072-3584, y=[7521093, 7536453]) tiene 0% de datos. Strips adyacentes (6 y 8) tienen datos.

## Es el mismo bug de v0.6.4

Strip 7 está completamente dentro de tile 19KCR (y=[7490200, 7600000], overlap=15.4km). No intersecta 19KCQ (y=[7390240, 7500040]). Los tiles KCQ retornan BBoxOutside para strip 7, y la lógica de consistencia multi-band descarta la escena completa.

Este bug fue corregido en v0.6.4 para Lluta. Parece que la refactorización de v0.6.6+ reintrodujo el problema.

## Verificar

Buscar en `handle_multiband_composite` la lógica que evalúa si un tile tiene "todas las bandas None". La regla debería ser:

- Todas las bandas None para un tile → tile fuera del strip, **omitir silenciosamente**
- Mezcla Some/None → inconsistencia real, descartar tile
- Todas Some → incluir

Si esta lógica opera a nivel de **escena** (todos los tiles juntos) en vez de **por tile individual**, un tile KCQ con BBoxOutside contamina la decisión para tiles KCR que sí tienen datos.

## Datos

- Strip 7: rows 3072-3584, y=[7521093, 7536453]
- 19KCR: y=[7490200, 7600000] → cubre strip 7 completamente
- 19KCQ: y=[7390240, 7500040] → NO cubre strip 7 (gap=21km)
- 19KDR: x=[399960, 509760] → cubre parte este
- 19KDQ: no cubre
