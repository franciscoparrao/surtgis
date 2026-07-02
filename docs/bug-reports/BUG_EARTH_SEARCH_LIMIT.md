---
De: Sesión postdoc
Para: Sesión SurtGis
Fecha: 2026-04-15
Prioridad: ALTA (bloquea cambio a Earth Search mientras PC está rate-limited)
---

# Bug: Earth Search 502 con page size > 250

Earth Search retorna HTTP 502 (Internal Server Error) cuando `limit` × response size excede un umbral. Probado:

- limit=100: ✓ 200 OK
- limit=250: ✓ 200 OK
- limit=350: ✗ 502
- limit=500: ✗ 502
- limit=1000: ✗ 502

El fix de v0.6.10 capea el page size a 1000 (para PC). Earth Search necesita un cap más bajo.

## Fix

Cap el page size por catálogo:

```rust
let page_size = match catalog {
    StacCatalog::PlanetaryComputer => search_limit.min(1000),
    StacCatalog::EarthSearch => search_limit.min(250),
    _ => search_limit.min(500),  // conservative default
};
```

Con `search_limit=1500` y `page_size=250`, el cliente haría 6 páginas de 250 items via next links. `search_all()` ya maneja la paginación.

## Contexto

PC nos tiene rate-limited (403 en toda la IP). Earth Search es la alternativa inmediata pero no funciona con page sizes grandes. Necesitamos este fix para poder procesar las 12 cuencas restantes.
