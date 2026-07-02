---
De: Sesión de procesamiento postdoc (15 cuencas)
Para: Sesión SurtGis
Fecha: 2026-04-11
Prioridad: CRÍTICA (bloquea procesamiento de 15 cuencas)
---

# Bug: SAS tokens expiran durante stac composite con muchas escenas

## Síntoma

`surtgis stac composite` con `--max-scenes=50` produce rasters con solo ~46km de cobertura N-S en cuencas de ~100km. Los strips del sur (4-7 de 7) reportan `scenes=0` — todos los píxeles quedan NaN.

## Causa raíz

Los SAS tokens de Planetary Computer se firman **una vez** en la Fase 1 (resolución de assets, línea ~1038 de `stac.rs`) y se embeben como strings estáticos en `TileRef.data_href` / `TileRef.scl_href`. Durante la Fase 3 (strip-by-strip), las URLs firmadas se pasan directamente a `read_cog_tile()`.

Con 50 escenas × ~4 tiles/escena × 7 strips, el procesamiento toma ~57 minutos. Los tokens de PC tienen validez de ~1 hora (campo `se=` en la URL). Para el momento de procesar strips 4-7 (~30-50 min después de firmar), **los tokens ya expiraron** y las lecturas HTTP fallan silenciosamente:

```rust
// stac.rs:2309 — el .ok()? suprime el error de token expirado
let mut r: surtgis_core::Raster<f64> = dr.read_bbox(bb, None).ok()?;
```

El `CogReaderBlocking::open()` o las range requests fallan con HTTP 403, pero `.ok()?` convierte eso en `None`, que se interpreta como "tile sin datos".

## Evidencia

1. **Con `--max-scenes=5`** (1 fecha, 3 min): cobertura 100% (100.4 km N-S, 95.5% valid pixels) ✓
2. **Con `--max-scenes=50`** (50 fechas, 57 min): cobertura 46km (45.9% valid, solo strips 1-3) ✗
3. **No hay errores ni warnings** en los logs — las fallas son silenciosas
4. Las tiles del sur (MGRS 19KCV, 19KDV) existen y son legibles (verificado con rasterio)
5. `tiles_for_bbox` calcula intersecciones correctas para todos los strips
6. El SAS token cache (`SasTokenCache`, max_age=45min) solo se consulta durante `sign_asset_href` en Fase 1 — nunca durante Fase 3

## Datos del caso de prueba

Cuenca: 01_rio_lluta (Arica, -17.6° a -18.5° S, 100.5 km N-S)
Todas las tiles en EPSG:32719 (sin cruce UTM):

| Tile MGRS | X range | Y range | Cubre strips |
|-----------|---------|---------|-------------|
| 19KCA | 300000-409800 | 7990240-8100040 | 1-3 (norte) |
| 19KDA | 399960-509760 | 7990240-8100040 | 1-3 (norte) |
| 19KCV | 300000-409800 | 7890220-8000020 | 4-7 (sur) |
| 19KDV | 399960-509760 | 7890220-8000020 | 4-7 (sur) |

Output grid: x=[358490, 462680] y=[7952102, 8052542], 3473×3348 px @ 30m

## Fix propuesto

### Opción A: Re-firmar por strip (recomendado)

Antes de procesar cada strip, verificar si los tokens embebidos están próximos a expirar y re-firmarlos. Requiere:

1. Almacenar los **hrefs originales** (sin firmar) en `TileRef` además de los firmados
2. Guardar el `Instant` de cuándo se firmaron
3. Al inicio de cada strip, si han pasado >30 min desde la firma, re-firmar todas las URLs

```rust
struct TileRef {
    data_href: String,        // firmada
    scl_href: String,         // firmada
    original_data_href: String,  // sin firmar (NUEVO)
    original_scl_href: String,   // sin firmar (NUEVO)
    epsg: Option<u32>,
    signed_at: Instant,       // NUEVO
}
```

En la Fase 3, al inicio de cada strip:
```rust
// Pseudo-código
if scene.tiles[0].signed_at.elapsed() > Duration::from_secs(30 * 60) {
    for tile in &mut scene.tiles {
        tile.data_href = client.sign_asset_href(&tile.original_data_href, collection)?;
        tile.scl_href = if !tile.original_scl_href.is_empty() {
            client.sign_asset_href(&tile.original_scl_href, collection)?
        } else { String::new() };
        tile.signed_at = Instant::now();
    }
}
```

### Opción B: Re-firmar inline (más simple, menos eficiente)

En `read_cog_tile`, si la lectura falla, intentar re-firmar y reintentar:

```rust
fn read_cog_tile(href: &str, bb: &BBox, log_meta: bool) -> Option<Raster<f64>> {
    match try_read_cog_tile(href, bb, log_meta) {
        Some(r) => Some(r),
        None => {
            // Retry with fresh token
            let fresh_href = re_sign_pc_href(href)?;
            try_read_cog_tile(&fresh_href, bb, false)
        }
    }
}
```

Desventaja: requiere acceso al signing API desde dentro del thread, y no distingue "token expirado" de "tile fuera de bbox".

### Opción C: Firmar justo antes de usar (más limpio)

Mover la firma al momento de uso en vez de hacerla en Fase 1. Almacenar solo hrefs originales en `TileRef` y firmar en el thread de descarga:

```rust
handles.push(std::thread::spawn(move || {
    let signed_data = sign_href_blocking(&data_href);  // firma just-in-time
    let data = read_cog_tile(&signed_data, &bb, first);
    // ...
}));
```

Ventaja: tokens siempre frescos. Desventaja: más requests al API de signing (pero el cache por container lo mitiga).

## Bonus: logging de errores

Independiente del fix, cambiar los `.ok()?` por logging explícito para que los errores de token no sean silenciosos:

```rust
// En read_cog_tile, línea 2300:
let mut dr = match CogReaderBlocking::open(href, CogReaderOptions::default()) {
    Ok(r) => r,
    Err(e) => {
        eprintln!("    [cog] open FAILED: {} (href={}...)", e, &href[..80.min(href.len())]);
        return None;
    }
};

// Línea 2309:
let mut r: Raster<f64> = match dr.read_bbox(bb, None) {
    Ok(r) => r,
    Err(e) => {
        eprintln!("    [cog] read_bbox FAILED: {}", e);
        return None;
    }
};
```

## Archivos relevantes

- `crates/cli/src/handlers/stac.rs`: Fase 1 (signing ~L1038), Fase 3 (strips ~L1182-1632), read_cog_tile (~L2299)
- `crates/cloud/src/stac_client.rs`: SasTokenCache (~L17-84), sign_asset_href (~L251-269)
- `crates/cloud/src/cog_reader.rs`: read_bbox → tiles_for_bbox → assemble_raster
