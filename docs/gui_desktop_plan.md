# Plan: GUI Desktop estilo SAGA (Enfoque Híbrido)

**Branch**: `feature/gui-desktop`
**Enfoque**: egui desktop nativo + web demo Svelte independiente (mantener ambos)

## Decisiones de arquitectura

| Decisión | Elección | Razón |
|---|---|---|
| Framework GUI | egui + eframe (wgpu) | Puro Rust, sin riesgo WebKitGTK en Linux, rendimiento directo |
| Docking | egui_dock | Soporta tabs, splits binarios, floating windows |
| Mapa basemap | walkers (fase 3) | Slippy map WebMercator con plugin system |
| File dialog | rfd (async) | Diálogos nativos del SO, no bloquea UI |
| Colormaps | Nuevo crate surtgis-colormap | Compartido entre GUI, CLI, WASM. Port directo del JS |
| Ejecución | crossbeam-channel + std::thread | Algoritmos en hilo separado, mensajes al UI |
| Web demo | Mantener Svelte tal cual | Independiente, no se modifica |

## Nuevos crates

### 1. `crates/colormap/` (reusable)

Port de `web/src/lib/colormap.js` a Rust. 8 esquemas predefinidos + motor de interpolación multi-stop.

### 2. `crates/gui/` (aplicación desktop)

Aplicación egui con docking estilo SAGA: MapCanvas, AlgoTree, Console, Properties.

## Fases

- **Fase 0**: Foundation - colormap crate + ventana eframe skeleton (~500-700 LOC)
- **Fase 1**: MVP Desktop - abrir GeoTIFF, visualizar, ejecutar algoritmo (~2000-2800 LOC)
- **Fase 2**: Workspace SAGA-like - multi-dataset, layers, properties (~2500-3500 LOC)
- **Fase 3**: Advanced - basemap tiles, rasters grandes, COG/STAC, 3D (~2000-3000 LOC)

## Grafo de dependencias

```
surtgis-gui (bin)
    ├── surtgis-core
    ├── surtgis-algorithms
    │   ├── surtgis-core
    │   └── surtgis-parallel
    ├── surtgis-parallel
    ├── surtgis-colormap (NUEVO)
    │   └── surtgis-core
    ├── surtgis-cloud (optional)
    ├── eframe + egui + egui_dock
    ├── rfd
    └── crossbeam-channel
```

## Exploración del codebase (referencia)

### Raster<T> API
- `Raster<T>` con `data: Array2<T>`, `transform: GeoTransform`, `crs: Option<CRS>`, `nodata: Option<T>`
- Métodos: `rows()`, `cols()`, `data()`, `statistics()`, `pixel_to_geo()`, `geo_to_pixel()`, `bounds()`
- `RasterElement` trait implementado para i8-i64, u8-u64, f32, f64

### IO
- `read_geotiff<T, P>(path, band)` / `write_geotiff<T, P>(raster, path, opts)`
- `read_geotiff_from_buffer<T>(data, band)` / `write_geotiff_to_buffer<T>(raster, opts)`

### Algoritmos (200+)
- terrain: slope, aspect, hillshade, curvature, tpi, tri, twi, geomorphons, viewshed, etc.
- hydrology: flow_direction, flow_accumulation, fill_sinks, watershed, hand, etc.
- imagery: ndvi, ndwi, savi, evi, band_math, etc.
- morphology: erode, dilate, opening, closing
- statistics: focal_mean, focal_std, focal_range, moran_i, getis_ord

### CLI dispatch pattern
```rust
// Read → Process → Write
let dem = read_dem(&input)?;
let result = slope(&dem, SlopeParams { units, z_factor })?;
write_result(&result, &output)?;
```

### Colormap schemes (from web/src/lib/colormap.js)
- terrain: green→yellow→brown→white
- divergent: blue→white→red
- grayscale: black→white
- ndvi: brown→yellow→green
- blue_white_red: blue→white→red
- geomorphons: 10 distinct classes
- water: white→cyan→blue
- accumulation: yellow→orange→red→purple
