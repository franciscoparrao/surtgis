# SurtGIS Algorithm Expansion Roadmap

**Fecha**: 2026-02-03
**Estado**: Planificado
**Baseline**: 44 algoritmos en GUI, ~114 funciones en `surtgis-algorithms`

## Resumen ejecutivo

La GUI expone 44 algoritmos de los ~114 disponibles en el crate nativo. El MCP Gateway
tiene 2,728 herramientas. Esta hoja de ruta propone expandir de 44 a ~90 algoritmos
nativos en 3 fases, priorizando lo que ya esta implementado en Rust pero no expuesto
en la GUI, y luego algoritmos nuevos.

---

## Inventario actual

### GUI (44 algoritmos en 5 categorias)

| Categoria   | # | Algoritmos |
|-------------|---|------------|
| Terrain     | 20 | slope, aspect, hillshade, multidirectional_hillshade, curvature, tpi, tri, twi, geomorphons, dev, landform, sky_view_factor, positive_openness, negative_openness, convergence_index, vrm, shape_index, curvedness, northness, eastness |
| Hydrology   | 5 | fill_sinks, priority_flood, flow_direction, flow_accumulation, hand |
| Imagery     | 8 | ndvi, ndwi, mndwi, nbr, savi, evi, bsi, band_math |
| Morphology  | 7 | erode, dilate, opening, closing, gradient, top_hat, black_hat |
| Statistics  | 4 | focal_mean, focal_std, focal_range, focal_median |

### Crate `surtgis-algorithms` (funciones implementadas NO expuestas en GUI)

| Modulo         | Funciones en crate | En GUI | Sin exponer |
|----------------|-------------------|--------|-------------|
| terrain        | ~50               | 20     | ~30         |
| hydrology      | 14                | 5      | 9           |
| imagery        | 15                | 8      | 7           |
| interpolation  | 12                | 0      | 12          |
| statistics     | 4 + zonal + autocorr | 4  | 3+          |
| morphology     | 7                 | 7      | 0           |
| vector         | 12                | 0      | 12          |

### MCP Gateway (2,728 herramientas por categoria)

| Categoria          | # herramientas | Backends |
|--------------------|---------------|----------|
| Imagery            | 957           | OTB, QGIS, GRASS, SAGA, GEE, Planetary |
| Vectors            | 698           | QGIS, OTB, GRASS, SAGA, GeoPandas |
| Terrain            | 520           | GRASS, SAGA, GDAL, GEE, Whitebox |
| Data Access        | 167           | Planetary, GRASS, SAGA |
| Raster Ops         | 135           | GDAL, QGIS, GRASS, SAGA |
| Hydrology          | 117           | GRASS, SAGA, Whitebox |
| Visualization      | 88            | OTB, QGIS |
| Classification/ML  | 32            | OTB, SAGA, GRASS, QGIS, MLPy |
| Climate            | 2             | CDS |
| Feature Extraction | 0             | (vacio) |

---

## Fase A — Quick wins: exponer lo que ya existe en Rust

**Objetivo**: De 44 a ~70 algoritmos. Sin escribir algoritmos nuevos, solo wiring
en registry.rs + executor.rs + (opcionalmente GUI categories nuevas).

### A.1 Hydrology (+9 algoritmos)

Funciones ya en `surtgis_algorithms::hydrology` sin exponer:

- [ ] `breach_depressions` — Breach depressions (alternativa a fill)
- [ ] `flow_direction_dinf` — D-infinity flow direction
- [ ] `flow_accumulation_mfd` — Multiple Flow Direction accumulation
- [ ] `flow_accumulation_mfd_adaptive` — Adaptive MFD
- [ ] `flow_accumulation_tfga` — TFGA flow accumulation
- [ ] `watershed` — Watershed delineation (basin IDs)
- [ ] `stream_network` — Stream network extraction
- [ ] `nested_depressions` — Nested depression analysis
- [ ] `flow_dinf` — D-infinity flow computation

### A.2 Imagery (+7 algoritmos)

Funciones ya en `surtgis_algorithms::imagery` sin exponer:

- [ ] `ndre` — Normalized Difference Red Edge
- [ ] `gndvi` — Green NDVI
- [ ] `ngrdi` — Normalized Green Red Difference Index
- [ ] `reci` — Red Edge Chlorophyll Index
- [ ] `normalized_difference` — Generic two-band normalized difference
- [ ] `reclassify` — Value-based reclassification
- [ ] `band_math` (expression) — String-based raster algebra expressions

### A.3 Terrain (+8 algoritmos seleccionados)

Funciones ya en `surtgis_algorithms::terrain` sin exponer:

- [ ] `spi` — Stream Power Index
- [ ] `sti` — Sediment Transport Index
- [ ] `viewshed` — Viewshed (cuenca visual desde un punto)
- [ ] `mrvbf` — Multi-resolution Valley Bottom Flatness
- [ ] `wind_exposure` — Wind shelter/exposure index
- [ ] `solar_radiation` — Solar beam + diffuse radiation
- [ ] `lineament_detection` — Linear feature detection
- [ ] `advanced_curvatures` — 12 Florinsky curvatures

### A.4 Statistics (+6 algoritmos)

Funciones ya en `surtgis_algorithms::statistics` sin exponer:

- [ ] `focal_min` — Focal minimum (FocalStatistic::Min, solo wiring)
- [ ] `focal_max` — Focal maximum (FocalStatistic::Max, solo wiring)
- [ ] `focal_sum` — Focal sum (FocalStatistic::Sum, solo wiring)
- [ ] `zonal_statistics` — Statistics by zones
- [ ] `global_morans_i` — Global Moran's I spatial autocorrelation
- [ ] `local_getis_ord` — Local Getis-Ord Gi* hot/cold spots

### A.5 Interpolation (categoria nueva, +6 algoritmos)

Modulo completo en `surtgis_algorithms::interpolation` sin exponer:

- [ ] `idw` — Inverse Distance Weighting
- [ ] `nearest_neighbor` — Nearest neighbor interpolation
- [ ] `natural_neighbor` — Natural neighbor (Voronoi)
- [ ] `tin_interpolation` — TIN (Triangulated Irregular Network)
- [ ] `tps_interpolation` — Thin Plate Spline
- [ ] `ordinary_kriging` — Ordinary Kriging

### Subtotal Fase A: +36 algoritmos (44 -> 80)

### Cambios necesarios Fase A

**Archivos a modificar**:

| Archivo | Cambio |
|---------|--------|
| `crates/gui/src/registry.rs` | +33 AlgorithmEntry, +categoria Interpolation |
| `crates/gui/src/executor.rs` | +33 ramas en match dispatch |
| `crates/gui/src/dock.rs` | (sin cambios) |
| `crates/gui/src/app.rs` | (sin cambios) |

**Nuevo AlgoCategory**: `Interpolation` (requiere punto de muestreo, UX distinta)

---

## Fase B — Algoritmos nuevos (nativos en Rust)

**Objetivo**: De ~77 a ~90. Algoritmos que NO existen en el crate pero son
fundamentales para una plataforma GIS.

### B.1 Imagery - indices nuevos (+5)

Implementar en `crates/algorithms/src/imagery/`:

- [ ] `ndsi` — Normalized Difference Snow Index (Green - SWIR) / (Green + SWIR)
- [ ] `ndbi` — Normalized Difference Built-up Index (SWIR - NIR) / (SWIR + NIR)
- [ ] `ndmi` — Normalized Difference Moisture Index (NIR - SWIR) / (NIR + SWIR)
- [ ] `msavi` — Modified SAVI (auto-adjusting L factor)
- [ ] `evi2` — Two-band EVI (sin banda azul)

### B.2 Statistics - focal majority (+1)

`FocalStatistic` ya soporta Min, Max, Sum, Count, Percentile (movidos a Fase A).
Solo queda Majority que requiere implementacion nueva.

- [ ] `focal_majority` — Focal majority/moda (**nuevo, implementar**)

### B.3 Terrain - utilidades faltantes (+2)

- [ ] `contour_lines` — Generar isolineas (output: Vec de LineString + valor)
- [ ] `cost_distance` — Distancia de costo acumulado

### B.4 Landscape Ecology (categoria nueva, +3)

Implementar en `crates/algorithms/src/landscape/`:

- [ ] `shannon_diversity` — Indice de diversidad de Shannon (ventana movil)
- [ ] `simpson_diversity` — Indice de diversidad de Simpson
- [ ] `patch_density` — Densidad de parches por ventana

### Subtotal Fase B: +11 algoritmos (80 -> 91)

### Cambios necesarios Fase B

**Archivos nuevos**:

| Archivo | Contenido |
|---------|-----------|
| `crates/algorithms/src/landscape/mod.rs` | Shannon, Simpson, patch density |
| `crates/algorithms/src/landscape/diversity.rs` | Implementaciones |

**Archivos a modificar**:

| Archivo | Cambio |
|---------|--------|
| `crates/algorithms/src/lib.rs` | +pub mod landscape |
| `crates/algorithms/src/imagery/mod.rs` | +ndsi, ndbi, ndmi, msavi, evi2 |
| `crates/algorithms/src/statistics/mod.rs` | +FocalStatistic variants (si no existen) |
| `crates/gui/src/registry.rs` | +14 AlgorithmEntry, +categoria Landscape |
| `crates/gui/src/executor.rs` | +14 ramas en match dispatch |

---

## Fase C — Avanzados y ML (futuro)

**Objetivo**: De ~90 a ~110+. Mas especializados.

### C.1 Classification/ML (categoria nueva)

- [ ] `pca` — Principal Component Analysis (eigen de covarianza)
- [ ] `kmeans_raster` — K-means clustering en espacio de bandas
- [ ] `isodata` — Iterative Self-Organizing Data Analysis
- [ ] `minimum_distance` — Clasificacion supervisada (distancia minima)
- [ ] `maximum_likelihood` — Clasificacion supervisada (maxima verosimilitud)

### C.2 Change Detection

- [ ] `raster_difference` — Diferencia entre dos rasters (con categorias de cambio)
- [ ] `change_vector_analysis` — CVA multi-banda

### C.3 Texture / Features

- [ ] `haralick_glcm` — Texturas de Haralick (energia, contraste, homogeneidad, etc.)
- [ ] `sobel_edge` — Deteccion de bordes Sobel
- [ ] `laplacian` — Filtro Laplaciano

### C.4 Hydrology avanzada

- [ ] `strahler_order` — Orden de Strahler en red de drenaje
- [ ] `flow_path_length` — Longitud del camino de flujo
- [ ] `isobasins` — Cuencas de area aproximadamente igual
- [ ] `flood_fill_simulation` — Simulacion de inundacion por nivel de agua

### Subtotal Fase C: +14 algoritmos (91 -> 105)

---

## Comparativa con plataformas de referencia

| Plataforma | Algoritmos nativos | Enfoque |
|------------|-------------------|---------|
| QGIS Processing | ~1,000+ | Generalista, plugins |
| SAGA GIS | ~800 | Terreno + hidrologia |
| GRASS GIS | ~500 | Cientifico |
| WhiteboxTools | ~500 | Hidrologia + terreno |
| Google Earth Engine | ~300 | Satelital + temporal |
| **SurtGIS actual** | **44** | Terreno + hidrologia |
| **SurtGIS post-A** | **~80** | +imagery +interpolation +statistics |
| **SurtGIS post-B** | **~91** | +landscape +nuevos indices |
| **SurtGIS post-C** | **~105** | +ML +change detection +textura |

Con ~90 algoritmos nativos en Rust, SurtGIS cubre el 80% del uso real
de analistas GIS, con rendimiento superior a las plataformas Python/C.

---

## Prioridad de implementacion (Fase A detallada)

### Batch 1 — Hydrology (alto impacto, muchos dependen de estos)

| # | Algoritmo | Funcion existente | Params necesarios | Output |
|---|-----------|-------------------|-------------------|--------|
| 1 | Breach Depressions | `breach_depressions()` | mode, max_length, max_depth | f64 |
| 2 | D-inf Flow Direction | `flow_direction_dinf()` | (ninguno) | f64 (angulo) |
| 3 | MFD Accumulation | `flow_accumulation_mfd()` | convergence | f64 |
| 4 | Watershed | `watershed()` | threshold | i32 (IDs) |
| 5 | Stream Network | `stream_network()` | threshold | u8 (1=stream) |
| 6 | Nested Depressions | `nested_depressions()` | (verificar params) | f64 |

### Batch 2 — Imagery indices (faciles, solo wiring)

| # | Algoritmo | Funcion existente | Bandas requeridas | Output |
|---|-----------|-------------------|-------------------|--------|
| 7 | NDRE | `ndre()` | NIR + RedEdge | f64 |
| 8 | GNDVI | `gndvi()` | NIR + Green | f64 |
| 9 | NGRDI | `ngrdi()` | Green + Red | f64 |
| 10 | RECI | `reci()` | NIR + RedEdge | f64 |
| 11 | Normalized Difference | `normalized_difference()` | Band1 + Band2 | f64 |
| 12 | Reclassify | `reclassify()` | ranges (especial) | f64 |

### Batch 3 — Terrain avanzado (alto valor cientifico)

| # | Algoritmo | Funcion existente | Params clave |
|---|-----------|-------------------|-------------|
| 13 | SPI | `spi()` | (auto fill+flow+acc) |
| 14 | STI | `sti()` | (auto fill+flow+acc) |
| 15 | Viewshed | `viewshed()` | observer_x, observer_y, height, radius |
| 16 | MRVBF | `mrvbf()` | t_slope, t_pctl, ... |
| 17 | Solar Radiation | `solar_radiation()` | day, lat, interval |
| 18 | Wind Exposure | `wind_exposure()` | directions, radius |

### Batch 4 — Interpolation (categoria nueva)

| # | Algoritmo | Funcion existente | Params clave |
|---|-----------|-------------------|-------------|
| 19 | IDW | `idw()` | power, search_radius, min_points |
| 20 | Nearest Neighbor | `nearest_neighbor()` | (ninguno) |
| 21 | Natural Neighbor | `natural_neighbor()` | (ninguno) |
| 22 | TIN | `tin_interpolation()` | (ninguno) |
| 23 | Thin Plate Spline | `tps_interpolation()` | regularization |
| 24 | Ordinary Kriging | `ordinary_kriging()` | variogram_model, range, sill |

### Batch 5 — Statistics + resto

| # | Algoritmo | Funcion existente | Params clave |
|---|-----------|-------------------|-------------|
| 25 | Focal Min | `focal_statistics()` | radius (FocalStatistic::Min) |
| 26 | Focal Max | `focal_statistics()` | radius (FocalStatistic::Max) |
| 27 | Focal Sum | `focal_statistics()` | radius (FocalStatistic::Sum) |
| 28 | Zonal Statistics | `zonal_statistics()` | zone_raster, statistic |
| 29 | Moran's I | `global_morans_i()` | (ninguno) |
| 30 | Getis-Ord Gi* | `local_getis_ord()` | radius |
| 31 | Lineament Detection | `lineament_detection()` | min_length, threshold |
| 32 | Advanced Curvatures | `advanced_curvatures()` | curvature_type |
| 33 | Band Math (expr) | `band_math()` | expression string |

---

## Notas tecnicas

### Interpolacion — UX especial

Los algoritmos de interpolacion requieren **puntos de muestreo** (SamplePoint),
no un raster de entrada. Opciones:

1. **Desde CSV**: Cargar archivo de puntos (x, y, value)
2. **Desde raster con NoData**: Interpolar los huecos (fill nodata)
3. **Desde seleccion manual**: Click en mapa para definir puntos

Para Fase A: opcion 1 (CSV) es la mas simple. Requiere un loader en `io.rs`
y un tipo de datos en `Workspace` para puntos.

### Viewshed — UX especial

Requiere coordenadas del observador. Opciones:
1. Entrada manual (x, y, height)
2. Click en mapa (futuro)

Para Fase A: entrada manual via ParamKind::Float.

### Watershed — Output i32

Ya existe dispatch_i32 (AppMessage::AlgoCompleteI32). Funciona.

### Stream Network — Output u8

Usa dispatch_u8 (AppMessage::AlgoCompleteU8). Funciona.

---

## Checklist de verificacion por algoritmo

Para cada algoritmo nuevo:

- [ ] Entry en `registry.rs` (id, name, category, description, params)
- [ ] Branch en `executor.rs` (dispatch + param extraction)
- [ ] Import de la funcion/types en `executor.rs`
- [ ] Test manual: ejecutar desde GUI y verificar resultado
- [ ] Colormap sugerido en `suggest_colormap()` si aplica
