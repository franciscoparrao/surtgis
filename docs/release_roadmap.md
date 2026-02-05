# SurtGIS — Hoja de Ruta hacia la Release

> De "proyecto con 200 algoritmos" a "herramienta que otros usan"
> Fecha: 2026-02-04
> Baseline: 105 GUI algorithms, 9 crates, 637 tests, 0 usuarios externos

---

## Contexto

SurtGIS tiene una base técnica madura: 200+ funciones nativas en Rust, 3 interfaces
(CLI, desktop, WASM), I/O cloud-native, rendimiento 2-21x superior a GDAL/SAGA, y
un nicho inexplorado en la literatura (procesamiento raster client-side vía WASM).

Lo que falta es distribución, validación externa y publicación. Esta hoja de ruta
prioriza las acciones que maximizan impacto con mínimo esfuerzo, en orden estricto
de dependencia.

---

## Dashboard

| Fase | Nombre | Items | Estado | Impacto |
|------|--------|-------|--------|---------|
| 0 | Merge y limpieza | 5 | Pendiente | Prerequisito |
| 1 | Publicar en registros | 8 | Pendiente | Distribución |
| 2 | Paper académico | 6 | Pendiente | Credibilidad |
| 3 | Python usable | 5 | Pendiente | Adopción masiva |
| 4 | Documentación pública | 4 | Pendiente | Onboarding |
| 5 | LiDAR pipeline | 4 | Pendiente | Diferenciación |
| 6 | GPU selectivo | 3 | Pendiente | Rendimiento |

---

## Fase 0: Merge y Limpieza Pre-Release

**Objetivo**: Tener `main` en estado publicable.

**Dependencias**: Ninguna.

- [ ] **0.1** Merge `feature/gui-desktop` → `main`
  - Squash o merge commit (preferir merge para preservar historia de 8+ commits)
  - Resolver conflictos si los hay en `Cargo.toml` o `Cargo.lock`
  - Verificar `cargo test --workspace` en main post-merge

- [ ] **0.2** Limpiar warnings restantes
  - `cargo clippy --workspace -- -D warnings` debe pasar limpio
  - Eliminar `#[allow(dead_code)]` innecesarios
  - Eliminar constante `D8_REVERSE` no usada en `hydrology/advanced.rs`

- [ ] **0.3** Actualizar `README.md` principal
  - Reflejar 105 algoritmos, 9 categorías, 9 crates
  - Agregar badges: tests, crates.io (placeholder), license
  - Tabla de benchmarks actualizada
  - Quick start: CLI, Rust, WASM, Python (placeholders para los 2 últimos)

- [ ] **0.4** Crear `CHANGELOG.md`
  - Formato Keep a Changelog
  - Versión `0.1.0` con inventario de lo que incluye
  - Secciones: Added, Changed, Fixed

- [ ] **0.5** Verificar licencia en todos los crates
  - `LICENSE-MIT` y `LICENSE-APACHE` en raíz
  - `license = "MIT OR Apache-2.0"` en cada `Cargo.toml`
  - `cargo deny check licenses` (instalar si no existe)

---

## Fase 1: Publicar en Registros de Paquetes

**Objetivo**: `cargo add surtgis-core`, `npm install surtgis`, `pip install surtgis`.

**Dependencias**: Fase 0 completa.

**Justificación**: Sin distribución estándar, el proyecto no existe para el 99% de
los desarrolladores. Es la barrera más importante a eliminar.

### 1.1 crates.io (Rust)

- [ ] **1.1.1** Metadata en cada `Cargo.toml` del workspace
  ```toml
  description = "..."
  keywords = ["gis", "geospatial", "raster", "terrain", "wasm"]
  categories = ["science::geo", "command-line-utilities"]
  repository = "https://github.com/franciscoparrao/surtgis"
  documentation = "https://docs.rs/surtgis-core"
  readme = "README.md"
  exclude = ["tests/fixtures/*", "benches/*"]
  ```

- [ ] **1.1.2** Publicar en orden de dependencias
  1. `surtgis-core` (base, sin deps internas)
  2. `surtgis-colormap` (solo depende de core)
  3. `surtgis-parallel` (depende de core)
  4. `surtgis-algorithms` (depende de core + parallel)
  5. `surtgis-cloud` (depende de core, optional)
  6. `surtgis` (CLI, depende de todo)
  - Usar `cargo publish --dry-run` antes de cada uno
  - Verificar `cargo install surtgis` funciona post-publicación

- [ ] **1.1.3** Verificar docs.rs
  - `cargo doc --workspace --no-deps --open` localmente
  - Verificar que docs.rs renderiza correctamente post-publish
  - Agregar `#![doc = include_str!("../README.md")]` en lib.rs de crates clave

### 1.2 npm (WASM/JavaScript)

- [ ] **1.2.1** Publicar `surtgis` en npmjs.com
  - `wasm-pack build --target bundler --release`
  - `npm publish` con 2FA
  - Verificar: `npm install surtgis` + import en Vite/webpack funciona

- [ ] **1.2.2** Verificar en frameworks frontend
  - Test mínimo en Vite + vanilla TS
  - Test mínimo en React/Next.js
  - Documentar incompatibilidades si las hay (SSR, Node vs browser)

### 1.3 PyPI (Python)

- [ ] **1.3.1** Configurar maturin + pyproject.toml
  - Python 3.9-3.13
  - Wheels: linux-x86_64, macos-arm64, windows-x86_64
  - `maturin publish` a TestPyPI primero
  - CI con `maturin-action` para builds multiplataforma

- [ ] **1.3.2** Publicar en PyPI
  - `pip install surtgis` funcional
  - Sin dependencia de GDAL en runtime
  - `import surtgis; print(surtgis.__version__)` funciona

---

## Fase 2: Paper Académico

**Objetivo**: Publicar en revista indexada (Computers & Geosciences, SoftwareX, JOSS, o ISPRS).

**Dependencias**: Fase 1.1 (crates.io) para poder citar con DOI/versión.

**Justificación**: El `sota_review.md` confirma un vacío en WOS: no existe ningún paper
sobre procesamiento geoespacial raster client-side vía WebAssembly. Esto es la
contribución más valiosa del proyecto, más allá del código.

### Target principal: SoftwareX o JOSS

SoftwareX y JOSS evalúan software científico por su calidad, documentación y utilidad.
No requieren novedad algorítmica, sino software bien construido. SurtGIS califica.

Alternativa: Computers & Geosciences si se enfoca en los benchmarks y la arquitectura
WASM como contribución técnica.

### Estructura del paper

- [ ] **2.1** Redactar borrador
  - Título: *"SurtGIS: High-performance geospatial raster analysis in Rust
    with native WebAssembly support"*
  - Secciones:
    1. Introduction (gap: no client-side raster processing in WASM)
    2. Architecture (9 crates, Raster<T>, maybe_rayon, WASM target)
    3. Algorithm coverage (105 GUI, 200+ total, 9 categories)
    4. Performance evaluation (vs GDAL, SAGA, WhiteboxTools, R/terra)
    5. WebAssembly deployment (572 KB, 33 functions, Web Worker)
    6. Cloud-native I/O (COG, STAC, HTTP Range)
    7. Case study (terrain analysis workflow: DEM → slope/TWI/viewshed)
    8. Conclusions

- [ ] **2.2** Benchmarks formales
  - Repetir benchmarks con metodología publicable:
    - 3 DEMs de distinto tamaño (1K, 4K, 8K pixels por lado)
    - 5 algoritmos representativos (slope, hillshade, focal_mean, flow_acc, TWI)
    - 5 herramientas de referencia (GDAL 3.11, SAGA 9, WhiteboxTools 2.4, GRASS 8.3, R/terra)
    - 10 repeticiones, reportar media y desviación estándar
    - Medir: tiempo de ejecución, uso de memoria peak, precisión numérica
    - Incluir WASM benchmark (browser Chrome/Firefox, single-thread)
  - Publicar datos de benchmark como CSV en repositorio

- [ ] **2.3** Benchmarks WASM vs servidor
  - Comparar: SurtGIS-WASM (browser) vs SurtGIS-nativo (server) vs GDAL-Python (server)
  - Métricas: tiempo, transferencia de datos, latencia end-to-end
  - Caso: usuario sube GeoTIFF 10 MB → slope → visualiza
  - Incluir overhead de red (upload a servidor vs procesamiento local)

- [ ] **2.4** Cross-validation extendida
  - Ampliar la cross-validation existente (slope, aspect) a más algoritmos:
    - hillshade, TPI, TRI, curvature, TWI, flow_accumulation
  - Comparar contra outputs de referencia GDAL + GRASS
  - Reportar RMSE, MAE, max error, correlación
  - Incluir edge cases: nodata handling, flat areas, steep slopes

- [ ] **2.5** Caso de estudio reproducible
  - DEM público (Copernicus 30m, ya descargado)
  - Notebook Jupyter (o script Rust/Python) reproducible
  - Workflow completo: descarga STAC → fill sinks → TWI → viewshed → mapa
  - Publicar como material suplementario

- [ ] **2.6** Submission
  - Elegir revista (JOSS para impacto rápido, SoftwareX para mayor visibilidad)
  - JOSS requiere: documentación, tests, contribución guidelines, ejemplo
  - SoftwareX requiere: artículo corto (3-6 páginas) + repositorio archivado (Zenodo)
  - Obtener DOI via Zenodo para la release v0.1.0

---

## Fase 3: Python Usable

**Objetivo**: Que un usuario GIS pueda hacer `pip install surtgis` y reemplazar
`rasterio` + `richdem` para análisis de terreno básico.

**Dependencias**: Fase 1.3 (PyPI publish).

**Justificación**: El 80% de los usuarios GIS trabajan en Python. Sin bindings usables,
SurtGIS compite en un mercado de nicho (desarrolladores Rust + usuarios CLI).

- [ ] **3.1** API Pythonic sobre PyO3
  - `surtgis.Raster` con `__repr__`, `shape`, `dtype`, `crs`, `bounds`
  - `surtgis.Raster.from_file(path)` y `.save(path)`
  - `surtgis.Raster.to_numpy()` (zero-copy si es posible) y `.from_numpy()`
  - `surtgis.Raster.plot()` con matplotlib (dependencia opcional)

- [ ] **3.2** Funciones de alto nivel
  ```python
  import surtgis
  dem = surtgis.read("dem.tif")
  slope = surtgis.slope(dem, units="degrees")
  twi = surtgis.twi(dem)  # auto fill + flow + slope
  slope.save("slope.tif")
  ```
  - Funciones que matcheen 1:1 con los nombres del CLI
  - Defaults sensatos (mismos que la GUI)
  - Documentación con docstrings + type hints

- [ ] **3.3** Integración con xarray/rioxarray
  - `Raster.to_xarray()` → `xarray.DataArray` con coordenadas espaciales
  - `surtgis.from_xarray(da)` → `Raster`
  - CRS y geotransform preservados en attrs
  - Compatibilidad con `rioxarray` para I/O roundtrip

- [ ] **3.4** Notebooks de ejemplo
  - `01_terrain_analysis.ipynb`: slope, aspect, hillshade, curvature
  - `02_hydrology.ipynb`: fill sinks, flow, TWI, watershed
  - `03_spectral_indices.ipynb`: NDVI, NDWI desde Sentinel-2 via STAC
  - `04_classification.ipynb`: K-means, ISODATA sobre imagen satelital
  - Ejecutables en Google Colab con `!pip install surtgis`

- [ ] **3.5** Benchmark Python: surtgis vs rasterio+richdem
  - Mismo DEM, mismos algoritmos
  - Comparar tiempo de ejecución y uso de memoria
  - Publicar como notebook reproducible

---

## Fase 4: Documentación Pública

**Objetivo**: Que un nuevo usuario pueda empezar a usar SurtGIS sin leer el código fuente.

**Dependencias**: Fase 1 (al menos un registro publicado).

- [ ] **4.1** docs.rs completo para crates Rust
  - Módulos documentados con `//!` headers
  - Ejemplos en docstrings de funciones principales (al menos 20)
  - `#[doc(hidden)]` para internals que no son API pública
  - Links entre tipos (e.g., `SlopeParams` linkeado desde `slope()`)

- [ ] **4.2** Landing page / GitHub Pages
  - Página estática con:
    - Qué es SurtGIS (1 párrafo)
    - Instalación: Rust / npm / pip (3 tabs)
    - Quick start para cada interfaz
    - Demo interactiva WASM (ya existe, deployar)
    - Link a docs.rs, npm, PyPI, paper
  - Framework: mdBook o VitePress (mínimo, no SPA compleja)

- [ ] **4.3** Documentación JavaScript/TypeScript
  - Getting started con Vite
  - API reference (auto-generado desde `.d.ts`)
  - Ejemplo: cargar GeoTIFF + slope + visualizar en canvas
  - Ejemplo: Web Worker para no bloquear UI
  - Guía de migración desde `geotiff.js`

- [ ] **4.4** CONTRIBUTING.md + issue templates
  - Guía de contribución: cómo compilar, correr tests, estructura de crates
  - Templates: bug report, feature request, algorithm request
  - Code of conduct (Contributor Covenant)

---

## Fase 5: LiDAR Pipeline

**Objetivo**: Pipeline completo LAS/LAZ → DEM → análisis, sin dependencias externas.

**Dependencias**: Ninguna técnica, pero requiere que Fases 0-2 estén al menos en progreso
para que el esfuerzo tenga audiencia.

**Justificación**: LiDAR es el sector de mayor crecimiento en datos geoespaciales
(USGS 3DEP, España PNOA, programas nacionales en Francia, UK, Noruega). No existe
un lector LAS/LAZ en Rust puro con pipeline a DEM integrado.

- [ ] **5.1** Lector LAS/LAZ nativo
  - Parser LAS 1.2-1.4 (header, VLRs, point records)
  - Descompresión LAZ via `laz-rs` (crate Rust existente)
  - Iterador lazy sobre puntos (sin cargar todo en memoria)
  - Filtros: por clasificación, return number, bounds, intensidad

- [ ] **5.2** Clasificación de suelo (PMF)
  - Progressive Morphological Filter (Zhang et al. 2003)
  - Separar ground returns de vegetation/buildings
  - Parámetros: cell size, slope threshold, max window size

- [ ] **5.3** Rasterización punto → DEM
  - Métodos: min, max, mean, IDW, TIN
  - Resolución configurable
  - Manejo de celdas vacías (interpolación o nodata)
  - Salida: `Raster<f64>` compatible con todos los algoritmos existentes

- [ ] **5.4** Pipeline integrado CLI + GUI
  - CLI: `surtgis lidar info|classify|rasterize|pipeline`
  - GUI: Panel de carga LAS con preview 3D (reusar vista wireframe)
  - Pipeline: `file.laz → classify → rasterize → slope/TWI/viewshed`

---

## Fase 6: GPU Selectivo

**Objetivo**: Acelerar los 3-4 algoritmos más pesados con wgpu compute shaders.

**Dependencias**: Ninguna técnica, pero menor prioridad que Fases 0-4.

**Justificación**: Solo tiene sentido para rasters grandes (>4096x4096) y algoritmos
con paralelismo de datos masivo. No reinventar pipeline GPU completo.

- [ ] **6.1** Crate `surtgis-gpu` con slope/hillshade en WGSL
  - Shader 3x3 neighborhood (Horn's method)
  - Buffer management: Raster → GPU → Raster
  - Feature flag `gpu` (no activado por defecto)
  - Fallback a CPU si GPU no disponible

- [ ] **6.2** Focal statistics genérico en GPU
  - Shader NxN moving window (mean, min, max, sum)
  - Shared memory optimization para ventanas solapadas
  - Benchmark: CPU vs GPU por tamaño de raster y radio

- [ ] **6.3** WebGPU en browser
  - Verificar que shaders WGSL funcionan en Chrome/Firefox WebGPU
  - WASM + wgpu (backend WebGPU)
  - Demo: terrain analysis GPU-accelerated en browser

---

## Orden de Ejecución Recomendado

```
Fase 0 ─────────────────────────────────┐
  Merge + limpieza + README              │
                                         ▼
Fase 1.1 ──────────► Fase 2 ──────────► Fase 4
  crates.io            Paper              Docs

Fase 1.2
  npm publish ─────────────────────────► Fase 4.3
                                          JS docs
Fase 1.3 ──────────► Fase 3
  PyPI                 Python usable ──► Fase 4.2
                                          Landing page

                     Fase 5              Fase 6
                      LiDAR              GPU
                     (paralelo)          (paralelo)
```

Las Fases 5 y 6 son independientes y pueden ejecutarse en cualquier momento,
pero su impacto se multiplica después de que Fases 0-4 estén completas
(porque ya habría usuarios que se beneficien).

---

## Métricas de Éxito

| Fase | Métrica | Target |
|------|---------|--------|
| 0 | `main` compilable y limpio | 0 warnings, 637+ tests |
| 1.1 | Descargas crates.io (30 días) | >100 |
| 1.2 | Descargas npm (30 días) | >50 |
| 1.3 | Descargas PyPI (30 días) | >200 |
| 2 | Paper aceptado | 1 publicación indexada |
| 3 | Issues/PRs de usuarios Python | >5 |
| 4 | Estrellas GitHub | >100 |
| 5 | Pipeline LiDAR funcional | LAS → DEM → slope end-to-end |
| 6 | Speedup GPU | >5x sobre CPU en 4096x4096 |

---

## Qué NO hacer

1. **No agregar más algoritmos al GUI**: 105 son suficientes para v0.1.0. Más algoritmos
   sin usuarios es acumulación sin validación.

2. **No competir con QGIS como desktop GIS**: El valor de SurtGIS es como library
   (Rust/Python/WASM), no como aplicación desktop. La GUI es una demo, no el producto.

3. **No invertir en Zarr/NetCDF todavía**: COG + STAC cubren >90% de los casos de uso
   de datos raster en la nube. Zarr es relevante para datos climáticos, un nicho
   diferente.

4. **No hacer GPU pipeline completo**: Solo slope, hillshade, y focal statistics
   justifican el esfuerzo. Flow accumulation tiene dependencias de datos que limitan
   la paralelización GPU.

5. **No reescribir PyO3 bindings**: Priorizar API mínima funcional (20 funciones)
   sobre cobertura completa. Iterar basándose en feedback real.

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Bus factor = 1 | Alta | Crítico | Publicar (crates.io, paper) para preservar el trabajo. CONTRIBUTING.md para atraer contribuidores |
| WASM sin usuarios | Alta | Alto | Publicar en npm + demo desplegada. Sin usuarios no hay feedback |
| Python bindings inestables | Media | Alto | Empezar con API mínima (10 funciones). Tests de integración |
| Paper rechazado | Media | Medio | Elegir JOSS (tasa de aceptación alta para software bien documentado). Tener SoftwareX como backup |
| egui limitaciones UX | Baja | Bajo | No invertir más en GUI desktop. Mantener como demo funcional |
| Dependencias Rust rotas | Baja | Medio | `Cargo.lock` versionado. Dependencias estables (ndarray, rayon, egui) |

---

## Resumen

El proyecto tiene fundamentos técnicos excepcionales. La prioridad ahora es
**distribución y validación**: publicar en registros estándar, escribir el paper
que documenta la contribución única (WASM geoespacial), y hacer que Python sea
usable. Todo lo demás (LiDAR, GPU, más algoritmos) se multiplica en impacto
después de que existan usuarios.

La secuencia crítica es: **merge → crates.io → paper → PyPI → documentación**.
