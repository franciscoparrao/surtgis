---
De: Sesión postdoc
Para: Sesión SurtGis
Fecha: 2026-04-14
Prioridad: ALTA
---

# Bug: Strip 7 cuenca 03 — resample OOB, tile 19KCR no se descarga

## Hallazgo definitivo

Strip 7 (y=[7521093, 7536453]) tiene resamples OOB:

```
[resample] out(0,0) geo=(331372.9,7521078.0) src=(-6859.2,11.7) OOB (src_bounds=0..2312,0..1557)
```

El output empieza en x=331372 pero el mosaic descargado empieza en x≈399960 (tile 19KDR). La columna src=-6859 es (331372-399960)/10 = -6859. El tile **19KCR** (x=[300000, 409800]) cubriría x=331372 pero **no se está incluyendo en el mosaic de strip 7**.

## Diagnóstico

- Tile 19KCR: x=[300000, 409800] y=[7490200, 7600000] → **cubre strip 7** (y=[7521093, 7536453]) ✓
- Tile 19KDR: x=[399960, 509760] y=[7490200, 7600000] → cubre strip 7 pero solo la parte este
- Output grid empieza en x=331358 → solo cubierto por 19KCR, no por 19KDR

Los items de 19KCR existen en la búsqueda (2500 items, 527 tiles). Pero para strip 7, solo se descargan tiles de la columna D (19KDR), no de la columna C (19KCR).

## Hipótesis

Puede que el tile 19KCR no esté presente en todas las 50 fechas seleccionadas. Si las fechas se seleccionan por orden cronológico y 19KCR no tiene imágenes en esas fechas (diferente órbita), ese tile no aparece en `scene.tiles` para esas fechas.

Para verificar: logear qué tiles tiene cada escena para strip 7, y si 19KCR aparece en alguna.

## Alternativa más probable

La búsqueda STAC retorna items de diferentes fechas con diferentes tiles. Al agrupar por fecha y tomar las primeras 50, algunas fechas pueden no tener tile 19KCR. Si TODAS las 50 fechas seleccionadas carecen de 19KCR para strip 7, queda vacío.

Esto pasaría si Sentinel-2 pasa por la columna C y D en órbitas diferentes (que es el caso — S2A y S2B tienen diferentes ground tracks).

## Fix propuesto

Al seleccionar las 50 fechas, priorizar fechas que tengan **cobertura espacial completa** (todos los tiles MGRS del bbox). Descartar fechas que solo cubren parte del bbox si hay suficientes fechas con cobertura completa.

Alternativamente: no agrupar por fecha. Agrupar por tile MGRS y tomar las mejores 50 escenas por tile independientemente. La mediana se calcula por pixel, así que no importa si diferentes pixels tienen diferentes fechas.
