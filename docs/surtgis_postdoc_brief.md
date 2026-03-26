# SurtGIS: Infraestructura Computacional para el Postdoctorado

**Para**: Dr. Mauricio Marín, Profesor Titular, Departamento de Ingeniería Informática, USACH
**De**: Francisco Parra
**Fecha**: Marzo 2026

---

## Resumen ejecutivo

SurtGIS es una librería geoespacial de alto rendimiento en Rust que he desarrollado como preparación para el postdoctorado. Compila desde un solo codebase a tres targets (nativo con paralelismo Rayon, WebAssembly para browser, Python via PyO3), implementa 136 algoritmos de análisis de terreno, hidrología e imágenes satelitales, y demuestra speedups de 1.8× sobre GDAL y hasta 23× sobre GRASS GIS en benchmarks de pipeline completo GeoTIFF.

El paper describiendo la librería está actualmente en revisión en *Environmental Modelling & Software* (Elsevier, IF 4.9).

**Argumento central**: SurtGIS no es solo un producto del doctorado — es la plataforma que permitirá ejecutar el postdoctorado de forma acelerada, porque los pipelines de procesamiento geoespacial que el postdoc requiere ya están resueltos como comandos de una línea.

---

## Relevancia para la línea de investigación

### Conexión con computación paralela y de alto rendimiento

SurtGIS implementa patrones directamente alineados con la experticia del laboratorio:

| Concepto HPC | Implementación en SurtGIS |
|-------------|--------------------------|
| **Paralelismo de datos** | Rayon work-stealing sobre grillas raster (partición por filas, parallel iterators) |
| **Streaming I/O** | Procesamiento strip-by-strip con memoria acotada (~200MB para DEMs de cualquier tamaño) |
| **Procesamiento distribuido de datos** | STAC client nativo con HTTP range requests para lectura parcial de Cloud Optimized GeoTIFFs |
| **Pipeline paralelo** | Composite streaming: por cada strip del output, lee ventanas parciales de N escenas remotas en paralelo |
| **Compilación cross-platform** | Un codebase → native (Rayon), WebAssembly (browser), Python (PyO3) |

### Resultados de rendimiento (publicados en el paper)

| Algoritmo | SurtGIS vs GDAL | SurtGIS vs GRASS | SurtGIS vs WhiteboxTools |
|-----------|-----------------|-------------------|--------------------------|
| Slope | 1.7–1.8× más rápido | 4.5–4.9× | — |
| Aspect | 2.0× | 3.5–4.8× | 4.2–7.9× |
| Flow accumulation D8 | — | 7.5–23.1× | 7.5–7.7× |
| Depression filling | — | — | 1.6–2.6× |

Benchmarks en pipeline completo (read GeoTIFF → compute → write GeoTIFF), no solo cómputo in-memory. DEMs sintéticos de 1K² a 20K² (400M celdas).

---

## Capacidad actual (v0.2.0)

### 136 algoritmos en 8 categorías

| Categoría | Algoritmos | Destacados |
|-----------|-----------|------------|
| Terreno | 25 | Florinsky 14 curvaturas, MRVBF, geomorphons, LS-Factor, Valley Depth |
| Hidrología | 16 | Priority-Flood, D-infinity, HAND, TWI, Sediment Connectivity Index |
| Imagery | 22 | 16 índices espectrales + calc con expresiones arbitrarias + composite + cloud-mask |
| Landscape | 5 | Connected components, PARA, FRAC, AI, COHESION, SHDI, SIDI |
| Morfología | 7 | Erode, dilate, opening, closing, gradient, top-hat, black-hat |
| Clasificación | 5 | K-means, ISODATA, PCA, min-distance, max-likelihood |
| Estadística | 4 | Focal (mediana, percentil), zonal, Moran's I, Getis-Ord Gi* |
| Interpolación | 5 | IDW, kriging, natural neighbor, thin-plate spline |

### 88 subcomandos CLI

Todo accesible desde la línea de comandos sin programar:

```bash
# Compute all terrain factors from a DEM in one pass
surtgis terrain all dem.tif --outdir ./factors/

# Full hydrology pipeline from DEM
surtgis hydrology all dem.tif --outdir ./hydro/ --threshold 1000

# Cloud-free satellite composite from Planetary Computer
surtgis stac composite \
  --bbox -71.5,-28.0,-69.0,-26.0 \
  --collection sentinel-2-l2a \
  --asset red \
  --datetime 2024-01-01/2024-12-31 \
  --max-scenes 12 \
  composite_red.tif

# Custom spectral index with expression
surtgis imagery calc \
  --expression "(NIR - Red) / (NIR + Red)" \
  --band NIR=nir.tif --band Red=red.tif \
  ndvi.tif

# Landscape analysis
surtgis landscape analyze classification.tif \
  --output-labels labels.tif --output-csv metrics.csv
```

### Streaming I/O

12 algoritmos de terreno procesan DEMs de cualquier tamaño con ~200MB de RAM:

| DEM | In-memory | Streaming |
|-----|-----------|-----------|
| 10K² (100M celdas) | 800MB | 200MB |
| 20K² (400M celdas) | 3.2GB | 200MB |
| 50K² (2.5B celdas) | **20GB — imposible** | **200MB** |

### STAC Composite (diferenciador único)

Un solo comando que ejecuta el pipeline completo de composición satelital:
1. Busca escenas en Planetary Computer/Earth Search (paginación automática)
2. Agrupa por fecha de adquisición
3. Mosaic espacial por fecha (une tiles de la misma pasada)
4. Cloud-mask con SCL (resampleo automático 20m→10m)
5. Mediana temporal para composite cloud-free

Validado en producción: 150M celdas, 14 tiles, 5 fechas, resultado en 10 minutos. **Ninguna otra herramienta CLI ofrece esto sin Python.**

---

## Impacto en el postdoctorado

### Sin SurtGIS (enfoque tradicional)

Para cada cuenca del estudio:
1. Descargar DEM desde Copernicus (script Python, Planetary Computer SDK)
2. Instalar GDAL, GRASS, WhiteboxTools (3 herramientas distintas)
3. Calcular slope con GDAL, flow accumulation con GRASS, TWI con WhiteboxTools
4. Integrar outputs con scripts Python (CRS, formatos, nodata incompatibles)
5. Descargar imágenes Sentinel-2 (script Python, cloud masking manual)
6. Composite temporal (numpy/scipy, ~50 líneas de orquestación)
7. Calcular índices espectrales (rasterio, banda por banda)
8. Landscape metrics (scipy.ndimage + código custom)

**Tiempo estimado por cuenca**: 2–3 días de desarrollo + depuración
**15 cuencas**: 30–45 días solo en procesamiento

### Con SurtGIS

```bash
# DEM: descargar y generar todos los factores
surtgis stac fetch-mosaic --collection cop-dem-glo-30 --bbox ... dem.tif
surtgis terrain all dem.tif --outdir factors/ --compress
surtgis hydrology all dem.tif --outdir hydro/ --compress

# Sentinel-2: composite cloud-free por banda
for band in red green blue nir swir16; do
  surtgis stac composite --collection sentinel-2-l2a \
    --asset $band --datetime 2024-01-01/2024-12-31 \
    --bbox ... --max-scenes 12 ${band}_composite.tif
done

# Índices espectrales
surtgis imagery calc --expression "(NIR-Red)/(NIR+Red)" \
  --band NIR=nir_composite.tif --band Red=red_composite.tif ndvi.tif

# Landscape metrics
surtgis landscape analyze landcover.tif --output-csv metrics.csv
```

**Tiempo por cuenca**: ~1 hora (dominado por descargas de red, no por procesamiento)
**15 cuencas**: 2–3 días, automatizable con un loop bash

### Reducción de tiempo: 10–15×

El postdoctorado se enfoca en modelamiento de susceptibilidad, no en procesamiento de datos. SurtGIS elimina la etapa de procesamiento como bottleneck.

---

## Contribución científica dual

El postdoctorado produce dos tipos de output:

1. **Papers de dominio** (susceptibilidad de humedales, erosión, riesgo) — donde SurtGIS es la herramienta que genera los datos de entrada
2. **Paper de software** (EMS, en revisión) — donde SurtGIS es la contribución principal

Esto permite publicar en dos frentes simultáneamente:
- Los papers de dominio citan a SurtGIS, generando tracción para el paper de software
- El paper de software valida la herramienta con los casos de uso del postdoctorado

---

## Estado de madurez

| Dimensión | Estado |
|-----------|--------|
| Paper | En revisión en EMS (Elsevier, IF 4.9) |
| Código | 56,000+ líneas Rust, 602 tests passing |
| Publicación | crates.io (Rust), PyPI (Python), npm (WASM) |
| CI/CD | 9 GitHub Actions jobs (check, clippy, test, WASM, Python, format, benchmarks) |
| Seguridad | cargo-audit limpio (1 advisory pendiente en pyo3, no crítico) |
| Licencia | MIT/Apache-2.0 dual license |
| Validación | Slope RMSE < 10⁻⁴° vs GDAL, curvaturas R² = 1.000000 vs analítico |
| Producción | Validado en pipeline real: 150M celdas Sentinel-2, 14 tiles, Planetary Computer |

---

## Conclusión

SurtGIS transforma el procesamiento geoespacial de un bottleneck de semanas en una operación de horas. Para el postdoctorado, esto significa que puedo dedicar el tiempo al modelamiento y análisis — que es donde está la contribución científica — en vez de a la ingeniería de datos que cada investigador resuelve ad-hoc con scripts frágiles.

La herramienta ya existe, está validada, y tiene un paper en revisión. El postdoctorado la usa, la extiende, y la cita. Es un ciclo virtuoso.
