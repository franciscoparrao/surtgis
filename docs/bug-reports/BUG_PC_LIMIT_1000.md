---
De: Sesión de procesamiento postdoc (15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-13
Prioridad: CRÍTICA (cuenca 03 no procesa)
---

# Bug: PC API rechaza search_limit > 1000

Planetary Computer devuelve HTTP 400 cuando `limit > 1000`:

```
STAC search returned HTTP 400: Input should be less than or equal to 1000
```

## Fix

El parámetro `limit` en la request STAC debe ser `min(search_limit, 1000)`. Para obtener más de 1000 items, usar **paginación** (el campo `next` en la respuesta STAC). `search_all()` probablemente ya maneja paginación — el problema es que `limit` se pasa como parámetro de la request individual, no como el total deseado.

```rust
// El limit en StacSearchParams debe ser el page size (≤1000)
// El max_items en StacClientOptions debe ser el total deseado
let params = StacSearchParams::new()
    .bbox(...)
    .limit(1000.min(search_limit))  // page size ≤ 1000
    ...;

let client_opts = StacClientOptions {
    max_items: search_limit as usize,  // total deseado (puede ser >1000)
    ..
};
```

Verificar que `search_all()` pagina automáticamente cuando `max_items > limit`.
