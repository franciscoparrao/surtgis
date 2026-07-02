---
De: Sesión postdoc
Para: Sesión SurtGis
Fecha: 2026-04-14
---

# Diagnóstico strip 7: errores HTTP transitorios + consistencia = strip vacío

## Hallazgos

1. **Strip 7 NO está vacío con pocas escenas**: test con 3 escenas → strip 7 = 38% valid. El problema solo aparece con 50 escenas.

2. **Errores HTTP reales**: `[cog] async read_bbox FAILED: HTTP error: error decoding response body`. Estos son errores transitorios de la API de Planetary Computer, no BBoxOutside.

3. **La consistencia amplifica los errores**: `⚠ 2023-06-05: tile has 2/3 bands — skipping for consistency`. Un error HTTP en 1 de 10 bandas descarta las 9 bandas que sí funcionaron.

4. **Con 50 escenas, la probabilidad de que TODAS fallen en algún strip es alta**: si cada escena tiene ~5% de probabilidad de perder una banda por HTTP error, con 50 escenas la probabilidad de que al menos una escena sobreviva es buena, pero si la consistencia descarta escenas con errores parciales, se necesita que las 10 bandas funcionen sin error en al menos 1 escena por strip.

## Fix propuesto: retry antes de consistencia

Antes de declarar una banda como fallida y activar la consistencia, reintentar el download 1-2 veces con backoff:

```rust
let data = match read_cog_tile_async(&href, &bb).await {
    Ok(r) => Some(r),
    Err(_) => {
        // Retry once after 1s
        tokio::time::sleep(Duration::from_secs(1)).await;
        read_cog_tile_async(&href, &bb).await.ok()
    }
};
```

Esto debería resolver la mayoría de los `error decoding response body` que son transitorios (network glitch, server hiccup).

## Alternativa: relajar consistencia

En vez de descartar el tile entero cuando una banda falla, descartar solo esa banda-tile pero mantener las demás. Cada banda acumularía sus propios scene_strips independientemente. La consistencia pixel-a-pixel se mantiene porque los pixels sin datos son NaN y la mediana los ignora.

Esto es más permisivo pero produce resultados correctos: si B08 falla en una escena pero B04 funciona, el NDVI para esos pixels se calcula con el B04 de esa escena y el B08 de otras escenas. La mediana temporal ya maneja diferentes cantidades de observaciones por pixel.
