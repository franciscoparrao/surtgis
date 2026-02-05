# Fase 3 — GUI Desktop Avanzado: Plan y Resultados

## Estado

**Completado.** Todos los componentes implementados y compilando correctamente.

## Resumen

Fase 3 agrega al GUI desktop:
- Upgrade de egui 0.31 a 0.33 (+ eframe 0.33, egui_dock 0.18)
- Basemap tiles OSM via walkers 0.52
- Tiled renderer con LRU cache para rasters grandes (>16M pixels)
- Navegador STAC para buscar y descargar COGs desde Planetary Computer / Earth Search
- Vista 3D wireframe del DEM activo con proyeccion isometrica

## Dependencias actualizadas

| Dependencia | Antes | Despues |
|-------------|-------|---------|
| egui | 0.31 | 0.33 |
| eframe | 0.31 | 0.33 |
| egui_dock | 0.16 | 0.18 |
| walkers | — | 0.52 (nueva) |

## Archivos nuevos (6)

| Archivo | LOC | Descripcion |
|---------|-----|-------------|
| `crates/gui/src/render/mod.rs` | 17 | Modulo render, enum MapMode (Simple/Basemap) |
| `crates/gui/src/render/map_tiles.rs` | 85 | Basemap OSM via walkers con RasterOverlay plugin |
| `crates/gui/src/render/tiled_renderer.rs` | 190 | LRU tile cache 512x512 para rasters >4096x4096 |
| `crates/gui/src/panels/stac_browser.rs` | 270 | Panel STAC: catalogo, bbox, datetime, resultados, download |
| `crates/gui/src/panels/view_3d.rs` | 195 | Vista wireframe 3D con proyeccion isometrica via Painter |
| **Total nuevos** | **~757** | |

## Archivos modificados (7)

| Archivo | Cambio |
|---------|--------|
| `Cargo.toml` (workspace) | egui 0.33, eframe 0.33, egui_dock 0.18, +walkers 0.52 |
| `crates/gui/Cargo.toml` | +walkers.workspace, cloud dep con features=["native"] |
| `crates/gui/src/main.rs` | +mod render |
| `crates/gui/src/dock.rs` | +PanelId::View3D, +PanelId::StacBrowser, layout actualizado |
| `crates/gui/src/app.rs` | +estados (map_mode, basemap, tiled_renderer, stac_browser, view_3d), +message handling, +tab dispatch, +STAC search/download |
| `crates/gui/src/panels/mod.rs` | +pub mod stac_browser, +pub mod view_3d |
| `crates/gui/src/state/messages.rs` | +StacSearchComplete, +StacAssetLoaded, +StacError |
| `crates/gui/src/menu.rs` | +Map Mode submenu (Simple/Basemap), migrado a egui 0.33 API (MenuBar::new, ui.close) |

## Componentes implementados

### 1. Basemap (walkers)

- `BasemapState`: almacena `HttpTiles` (OSM), `MapMemory`, `Position` centro
- `RasterOverlay`: implementa `Plugin` de walkers, dibuja textura raster sobre tiles
- `show_basemap()`: renderiza mapa + overlay
- Toggle via menu View > Map Mode > Basemap (OSM)
- Lazy-init: basemap se crea al primer uso, centrado en dataset activo o Madrid por defecto
- CRS: walkers usa WebMercator; bounds del raster en WGS84

### 2. Tiled Renderer

- `TiledRenderer`: LRU cache de 256 tiles de 512x512
- `RasterTileKey`: (generation, col, row) para invalidacion por generacion
- `visible_tiles()`: calcula tiles visibles dado viewport + scale + offset
- `get_or_render()`: obtiene o genera tile bajo demanda
- Activacion: `should_use(rows, cols)` retorna true cuando rows*cols > 16M
- Eviction LRU cuando cache alcanza 256 tiles

### 3. STAC Browser

- Feature gate: `cloud` feature requerido
- Selector de catalogo: Planetary Computer / Earth Search / Custom URL
- Filtros: bbox, datetime range, collection, max cloud cover (slider)
- Boton "Use Map Extent" para rellenar bbox desde dataset activo
- Resultados en tabla scrollable con columnas: fecha, cloud%, platform, GSD, collection
- Boton Download por item con signed URLs (Planetary Computer)
- Ejecucion en background via `std::thread::spawn` + crossbeam channels
- Patron tx/rx identico al de algoritmos

### 4. Vista 3D Wireframe

- Proyeccion 3D→2D con egui Painter (sin pipeline wgpu)
- Grid subsampled del raster con step configurable (1-32)
- Lineas coloreadas por elevacion usando colormap activo
- Controles: sliders para azimuth (0-360), elevation (5-85), z_exaggeration (0.1-20)
- Drag para rotar (azimuth + elevation)
- Solo rasters f64 (DEM)

### 5. Layout actualizado

```
+----------------------------+---------------------+
|  Map (tab)                 |  Algorithms (tab)   |
|  3D View (tab)             |  Properties (tab)   |
|                            |  STAC (tab)         |
|                            +---------------------+
|                            |  Data (tab)         |
|                            |  Layers (tab)       |
+----------------------------+---------------------+
|              Console                              |
+---------------------------------------------------+
```

## Verificacion

```
cargo build -p surtgis-gui                    # OK (0 errores, warnings de dead_code esperados)
cargo build -p surtgis-gui --features cloud   # OK (0 errores)
cargo test --workspace                         # OK (585 tests, 0 failures)
```

## Migracion egui 0.31 → 0.33

Cambios necesarios para la migracion:
- `egui::menu::bar()` → `egui::MenuBar::new().ui()`
- `ui.close_menu()` → `ui.close()`
- `Shadow::NONE` sigue funcionando sin cambios
- `ComboBox::from_id_salt` sigue funcionando
- `Projector::project()` retorna `Vec2` (no convierte a `Pos2` directamente)
- Rust 2024 edition: `ref mut` en pattern matching no permitido bajo default binding mode `ref mut`

## Proximos pasos sugeridos

- Integrar tiled renderer en map_canvas cuando raster > threshold
- Agregar mas tile providers (Mapbox, Esri) ademas de OSM
- Implementar reproyeccion automatica de rasters a WebMercator para overlay correcto
- Anadir cache persistente de tiles OSM a disco
- Mejorar vista 3D con iluminacion y textura
