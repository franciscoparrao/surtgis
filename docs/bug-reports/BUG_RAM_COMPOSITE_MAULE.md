---
De: Sesión postdoc (susceptibilidad Río Maule)
Para: Sesión SurtGIS
Fecha: 2026-04-18
Prioridad: ALTA (bloquea procesamiento de escenas grandes)
Versión: surtgis 0.6.19 (commit b05b7fd)
---

# `stac composite` consume >24 GB RAM sin terminar en 4 horas (bbox ~211×167 km, 10 bandas, 50 escenas)

## Problema

Invocación real para el paper de susceptibilidad (Río Maule, Chile):

```bash
surtgis --compress stac composite \
  --catalog pc \
  --bbox -72.687,-36.579,-70.319,-35.080 \
  --collection sentinel-2-l2a \
  --asset B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 \
  --datetime 2022-01-01/2024-12-31 \
  --max-scenes 50 \
  --naming asset \
  --align-to dem_30m.tif \
  composite.tif
```

**Resultado observado:**

| Métrica | Valor |
|---------|-------|
| Tiempo transcurrido | 3h 58min (aún corriendo, no completó) |
| RSS (memoria residente) | **24.5 GB** |
| VmPeak (pico histórico) | **34.9 GB** |
| VmSize (virtual) | 32.5 GB |
| CPU | 57% |
| Threads | 27 |
| Estado | Rl (Running) |

**Sistema**: 38 GB RAM total. El proceso llegó a consumir el 92% del RAM disponible, forzando uso de swap y dejando al sistema con 2.1 GB libres. Terminé matándolo porque bloqueó otras tareas.

## Escala del trabajo

- **Raster target (`--align-to`)**: 7274 × 5725 pixels, 30m (≈41.6 M pixels)
- **Bbox**: 2.37° × 1.50° ≈ 211 × 167 km (latitud ~-36°)
- **Bandas**: 10 (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
- **Escenas**: 50 (3 años × filtro de coverage)
- **Salida esperada**: 10 × 7274 × 5725 × 4 bytes (f32) ≈ **1.6 GB** en disco

## Hipótesis

El picco de 34.9 GB sugiere que **todas las bandas × todas las escenas están acumulándose en memoria** en algún punto, probablemente durante la agregación mediana/mean por pixel. El tamaño esperado en memoria si fuera todo f64 en RAM:

- 41.6M pixels × 10 bandas × 50 escenas × 8 bytes = **166 GB** (obviamente no cabe)
- Incluso a f32 serían 83 GB

Pero el proceso llegó a 34 GB, lo que sugiere que se está intentando cargar **parte significativa del stack** en memoria en lugar de:
1. Procesar strip por strip (cada strip: 7274 × 512 × 10 × 50 = 1.3 GB en f32 = manejable)
2. Procesar tile por tile (COG tiles de 512×512 a través de las escenas)
3. Liberar escenas antiguas después de integrar su contribución al mediador

## Relación con bugs previos

Veo en el repo `BUG_RAM_SPIKE.md` (2026-04-15) donde ya reporté que `--strip-rows=1024` con 10 bandas causaba OOM. Esa vez `--strip-rows=512` (default) funcionaba para un bbox más chico. Ahora con este bbox de 211×167 km ni el default alcanza.

También hay `BUG_MULTIBAND_STRIP3_GAP.md` (2026-04-12) sobre gaps en strip 3 del multi-band. Posiblemente la lógica de strip_reader no escala bien a áreas grandes cuando hay muchas bandas/escenas.

## Comportamiento esperado

Para un composite con 10 bandas × 50 escenas × 41.6M pixels, el uso de RAM debería ser aproximadamente:

```
RAM ≈ (strip_rows × cols × n_bands × n_scenes × bytes_per_value) + constante
    ≈ (512 × 7274 × 10 × 50 × 4 bytes) ≈ 7.4 GB  (máximo, si se paraleliza por strip)

O mejor aún, si se integra escena por escena:

    ≈ (strip_rows × cols × n_bands × bytes_per_value × 2) + acumuladores
    ≈ (512 × 7274 × 10 × 4 × 2) ≈ 300 MB + acumuladores
```

En cualquier caso, **no debería exceder 4-8 GB** para esta carga.

## Sugerencias de fix

### 1. Pre-estimación y warning

Al inicio del comando, estimar el RAM peak y warnear/abortar si excede el 70% disponible:

```
Estimando RAM para composite:
  Raster: 7274 × 5725 × 10 bands
  Scenes: 50
  strip_rows: 512

  ⚠ Estimated peak RAM: 7.4 GB
     Available: 38 GB, free: 30 GB
     OK to proceed (19% of available)
```

Si excede:
```
  ✗ Estimated peak RAM: 34 GB exceeds 70% of available (26 GB)
     Try: --max-scenes 20, or smaller --bbox, or reduce --asset
     Use --force to proceed anyway
```

### 2. Integración incremental por escena

El algoritmo de mediana requiere tener todas las escenas en memoria, pero para **mean** o **max** se puede hacer incremental:

```rust
// Escena por escena (no todas en RAM)
for scene in scenes {
    let strip_data = scene.read_strip(...)?;  // solo 1 strip de 1 escena
    accumulator += strip_data;
    count += 1;
}
mean = accumulator / count;
```

Para mediana, usar aproximación con histograma (p-square algorithm, t-digest) o quick-select por tiles.

### 3. Cap automático basado en RAM disponible

```rust
let available = get_available_ram();
let max_batch_scenes = available / (n_bands * cols * strip_rows * 8 * safety_factor);
// Procesar escenas en batches de este tamaño
```

### 4. Documentación CLI

Agregar a `stac composite --help`:
```
NOTE: RAM usage scales with bbox × n_bands × n_scenes.
Rule of thumb: reduce --max-scenes or split bbox if > ~1000 km² × 10 bands × 50 scenes.
```

## Qué haré como workaround

Voy a relanzar Maule dividiendo:

```bash
# Opción A: Menos escenas (mediana con 20 es aceptable)
--max-scenes 20

# Opción B: Dividir bbox en tiles y mosaicar
# Norte
--bbox -72.687,-35.830,-70.319,-35.080

# Sur
--bbox -72.687,-36.579,-70.319,-35.830

# Luego mosaic de los dos
```

Si alguna de las dos opciones funciona con <8 GB RAM, registro los resultados.

## Contexto

Paper: "Modelo de susceptibilidad a deslizamientos en Chile central (Río Maule)"
Zona: 11 cuencas de los factors del paper de postdoc
Hardware: ThinkPad T14 Gen 3, 38 GB RAM, Ryzen 7 PRO

Este era el quinto raster de 11 cuencas que procesaba. Las anteriores más pequeñas sí terminaron. El bbox del Maule es notablemente más grande (~3.5× el área media de las otras).
