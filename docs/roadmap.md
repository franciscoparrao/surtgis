# SurtGis Roadmap

## Estado actual

- **45+ algoritmos** en 7 módulos (terrain, hydrology, imagery, interpolation, vector, morphology, statistics)
- **231 tests**, 0 warnings
- **CLI**: 26+ comandos (terrain, hydrology, imagery, morphology)
- **Benchmarks**: 2-21x más rápido que GDAL, SAGA, WhiteboxTools, R/terra
- **I/O**: GeoTIFF nativo + GDAL opcional via feature flag

## Líneas de trabajo

### 1. CLI para algoritmos nuevos

Agregar ~15 subcomandos para los algoritmos implementados que aún no tienen interfaz CLI:

- `terrain twi`, `terrain spi`, `terrain sti`
- `terrain geomorphons`, `terrain viewshed`, `terrain svf`
- `terrain openness`, `terrain convergence`, `terrain smoothing`
- `terrain wind-exposure`, `terrain solar-radiation`, `terrain mrvbf`
- `terrain multiscale-curvature`
- `statistics focal`, `statistics zonal`

**Impacto**: Usabilidad directa desde terminal sin escribir código Rust.

### 2. WASM support

Compilar a WebAssembly para ejecutar análisis geoespacial en el navegador.

- Feature flags para desactivar Rayon e I/O del filesystem
- API expuesta via `wasm-bindgen`
- Paquete npm publicable
- Demo web interactiva

**Impacto**: Análisis client-side sin servidor, demos interactivas, integración con mapas web.

### 3. Python bindings (PyO3)

Módulo Python nativo (`pip install surtgis`) que expone todos los algoritmos.

- Conversión numpy ↔ Raster<T> zero-copy
- API Pythonic con type hints
- Publicación en PyPI

**Impacto**: Acceso al ecosistema Python/Jupyter, adopción masiva.

### 4. Benchmarks para nuevos algoritmos

- Criterion micro-benchmarks para los 15 algoritmos nuevos
- Comparación wall-clock vs SAGA/WhiteboxTools/GRASS para geomorphons, viewshed, SVF, solar radiation
- Benchmark de focal/zonal statistics vs R/terra y GDAL

**Impacto**: Datos concretos de rendimiento para marketing y papers.

### 5. Más algoritmos (Tier 2/3 del state-of-the-art review)

- Terrain texture (GLCM - Haralick)
- Landscape metrics (fragmentation, connectivity, patch analysis)
- Topographic Ruggedness Vector (Sappington 2007)
- Hydrological connectivity index
- Terrain surface classification (Wood 1996)
- Multi-flow direction (MFD/DINF)
- Channel network extraction

**Impacto**: Cerrar gaps con WhiteboxTools y SAGA.

### 6. Calidad y CI/CD

- GitHub Actions: build + test en Linux/macOS/Windows
- Doc-tests para todos los módulos públicos
- Examples con datos reales (DEM descargable)
- `cargo doc` publicado en GitHub Pages
- Badges (tests, coverage, docs)

**Impacto**: Profesionalización del proyecto, confianza de usuarios.
