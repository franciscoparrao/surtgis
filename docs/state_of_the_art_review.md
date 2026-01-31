# SurtGis — Revisión del Estado del Arte: Algoritmos GIS en Herramientas Open Source

**Fecha:** 2026-01-30
**Objetivo:** Identificar la combinación ideal de algoritmos para que SurtGis marque diferencia en el ecosistema GIS open source.

---

## 1. Panorama Actual de Competidores

| Herramienta | Algoritmos | Lenguaje | Fortaleza principal |
|---|---|---|---|
| **SAGA GIS** | ~800 | C++ | Morfometría, análisis geocientífico |
| **WhiteboxTools** | ~550 | Rust | Geomorfometría multiescala, LiDAR, hidrología |
| **GRASS GIS** | ~350 | C | Hidrología, radiación solar, ecología |
| **GDAL** | ~50 | C/C++ | I/O universal, operaciones básicas de terreno |
| **SurtGis** (actual) | ~30 | Rust | Rendimiento puro en terreno/imagen/morfología |

### WhiteboxTools — Competidor directo en Rust

WhiteboxTools (Prof. John Lindsay, U. of Guelph) es el competidor más cercano:
- ~550 herramientas, escrito en Rust, sin dependencias externas (no GDAL)
- 99 herramientas solo en geomorfometría (vs 7 en SurtGis)
- CLI-first (no ofrece API de librería Rust pública)
- No tiene soporte cloud-native (COG/STAC)
- Incluye algoritmos únicos: Multiscale LSPs, Florinsky curvatures, Feature-Preserving Smoothing, Geomorphons, SVF, Openness, etc.

**Ref:** [WhiteboxTools Geomorphometric Analysis](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html)

### SAGA GIS — El especialista en geocientífico

SAGA (U. Göttingen) tiene >800 módulos con algoritmos únicos:
- Morphometric Protection Index, SAGA Wetness Index, Convergence Index
- Effective Air Flow Heights, Mass Balance Index, Diurnal Anisotropic Heating
- MRVBF (Multi-resolution Valley Bottom Flatness)
- Fuzzy Landform Element Classification

**Ref:** [SAGA Morphometry Module Library](https://saga-gis.sourceforge.io/saga_tool_doc/2.2.4/ta_morphometry.html)

### GRASS GIS — Referencia en hidrología y radiación

GRASS destaca por:
- r.sun: modelo de radiación solar completo (beam, diffuse, reflected)
- r.watershed: hidrología robusta a gran escala
- r.survey: análisis de visibilidad con orientación de pixel
- 350+ herramientas con >40 años de desarrollo

**Ref:** [GRASS r.sun Manual](https://grass.osgeo.org/grass-stable/manuals/r.sun.html)

---

## 2. Algoritmos Ausentes en SurtGis — Análisis por Impacto

### Tier 1 — Alto impacto, alta demanda ("must have")

| Algoritmo | Disponible en | Justificación |
|---|---|---|
| **Focal Statistics** (generic kernel) | Todos | Base para decenas de índices derivados. Operaciones de ventana deslizante: mean, std, min, max, range, sum, count, median, percentile. |
| **Zonal Statistics** | Todos | Estadísticas por zonas/polígonos sobre rasters. Herramienta esencial de análisis espacial. |
| **TWI** (Topographic Wetness Index) | SAGA, GRASS, WBT | El índice hidrológico más citado. Proxy de humedad del suelo. TWI = ln(a / tan(β)). Requiere flow accumulation + slope. |
| **Geomorphons** (Jasiewicz & Stepinski 2013) | GRASS, WBT | Clasificación auto-adaptativa de geoformas usando patrones ternarios. 10 clases: peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit, flat. ~1000 citas. |
| **SPI** (Stream Power Index) | SAGA, WBT | SPI = A × tan(β). Potencia erosiva del flujo. Fundamental en geomorfología fluvial. |
| **STI** (Sediment Transport Index) | SAGA, WBT | STI basado en USLE. Capacidad de transporte de sedimentos. |
| **Viewshed** | GRASS, WBT, ArcGIS | Análisis de visibilidad. Aplicaciones: planificación urbana, telecomunicaciones, energía eólica. |

### Tier 2 — Diferenciadores

| Algoritmo | Disponible en | Justificación |
|---|---|---|
| **Multiscale Curvatures** (Florinsky 2016) | WBT | Ajuste polinomial 3er orden en 5×5. Más robusto que métodos 3×3. Curvaturas en projected + geographic CRS. |
| **Sky View Factor (SVF)** | WBT, SAGA | Fracción visible del cielo. Aplicaciones: microclima, arqueología, urbanismo. |
| **Openness** (Yokoyama 2002) | WBT | Apertura angular positiva/negativa del terreno. Complementa SVF. |
| **Convergence Index** | SAGA | Similar a curvatura horizontal pero más suave. Indica convergencia/divergencia de flujo. |
| **Feature-Preserving Smoothing** | WBT | Suavizado de DEM preservando quiebres de pendiente. Preprocesamiento esencial. |
| **Wind Exposure (Topex)** | WBT, ArcGIS | Exposición al viento basada en topografía. Riesgo de viento, redistribución de nieve. |
| **Solar Radiation** (Hofierka & Šúri) | GRASS (r.sun) | Radiación directa, difusa y reflejada con sombras topográficas. Gran impacto en energía y ecología. |
| **MRVBF/MRRTF** (Gallant & Dowling 2003) | SAGA | Multi-resolution Valley Bottom / Ridge Top Flatness. Mapeo geomorfológico multiescala. |

### Tier 3 — Análisis avanzado y tendencias emergentes

| Algoritmo | Disponible en | Justificación |
|---|---|---|
| **Spatial Autocorrelation** (Moran's I, Getis-Ord Gi*) | ArcGIS, R, QGIS plugin | Hotspot analysis sobre rasters. Tendencia fuerte en análisis espacial. |
| **Landscape Metrics** (FRAGSTATS-style) | R (landscapemetrics) | Ecología del paisaje sobre rasters categóricos. Nicho poco cubierto. |
| **DEM Super-Resolution** (deep learning) | Ninguno integrado | Tendencia 2024-2025. CNN/transformers para mejorar resolución de DEMs. |
| **Hypsometric Analysis** | WBT, SAGA | Curvas hipsométricas para análisis de cuencas. |
| **Breakline Mapping** | WBT | Detección de quiebres de pendiente. |

---

## 3. Nicho Estratégico de SurtGis

Lo que diferencia a SurtGis de WhiteboxTools (competidor directo en Rust):

1. **Rendimiento demostrado** — 2-21x más rápido que alternativas en terreno básico
2. **API de librería** — WBT es CLI-first; SurtGis ofrece API Rust nativa ergonómica
3. **Cloud-native** (futuro) — Soporte para COG/STAC directo
4. **WASM** (futuro) — Análisis geoespacial en el navegador
5. **Estadística espacial** — Focal/Zonal stats + autocorrelación (nicho poco cubierto)
6. **Multiscale nativo** — Con paralelismo Rayon automático

---

## 4. Plan de Implementación: 15 Algoritmos Prioritarios

| # | Algoritmo | Módulo | Complejidad |
|---|---|---|---|
| 1 | Focal Statistics | `statistics/focal` | Media |
| 2 | Zonal Statistics | `statistics/zonal` | Media |
| 3 | TWI | `terrain/twi` | Baja |
| 4 | Geomorphons | `terrain/geomorphons` | Media-Alta |
| 5 | SPI + STI | `terrain/spi_sti` | Baja |
| 6 | Viewshed | `terrain/viewshed` | Media-Alta |
| 7 | Sky View Factor | `terrain/sky_view_factor` | Alta |
| 8 | Multiscale Curvatures | `terrain/multiscale_curvatures` | Media |
| 9 | Convergence Index | `terrain/convergence` | Baja-Media |
| 10 | Solar Radiation | `terrain/solar_radiation` | Alta |
| 11 | Wind Exposure (Topex) | `terrain/wind_exposure` | Media |
| 12 | Feature-Preserving Smoothing | `terrain/smoothing` | Media |
| 13 | Openness | `terrain/openness` | Media |
| 14 | Spatial Autocorrelation | `statistics/autocorrelation` | Media |
| 15 | MRVBF/MRRTF | `terrain/mrvbf` | Alta |

---

## 5. Queries para Web of Science

### Geomorfometría y Análisis de Terreno

```
TS=("geomorphometry" AND ("algorithm*" OR "method*") AND ("DEM" OR "digital elevation model") AND ("open source" OR "software"))
```

```
TS=("geomorphons" AND "landform classification" AND "DEM")
```

```
TS=("multiscale" AND ("curvature" OR "land surface parameter*") AND "digital elevation model" AND ("Florinsky" OR "Gaussian" OR "scale space"))
```

```
TS=("topographic position index" OR "TPI") AND TS=("landform classification" OR "geomorphometry") AND TS=("multiscale" OR "multi-scale")
```

### Índices Hidrológicos

```
TS=("topographic wetness index" OR "TWI") AND TS=("algorithm*" OR "flow accumulation" OR "D8" OR "FD8" OR "MFD") AND TS=("comparison" OR "evaluation" OR "accuracy")
```

```
TS=("stream power index" OR "sediment transport index") AND TS=("DEM" OR "terrain analysis" OR "erosion")
```

```
TS=("SAGA wetness index" OR "MRVBF" OR "valley bottom flatness") AND TS=("geomorphometry" OR "terrain" OR "landform")
```

### Visibilidad y Radiación Solar

```
TS=("viewshed" AND "algorithm*" AND ("parallel" OR "GPU" OR "high performance" OR "optimization")) AND TS=("DEM" OR "digital elevation model")
```

```
TS=("sky view factor" OR "openness") AND TS=("DEM" OR "terrain") AND TS=("algorithm*" OR "computation" OR "method*")
```

```
TS=("solar radiation" AND "DEM" AND ("r.sun" OR "terrain" OR "topographic shadow*") AND ("model*" OR "algorithm*"))
```

### Estadística Espacial en Rasters

```
TS=("focal statistics" OR "moving window" OR "kernel statistics") AND TS=("raster" OR "grid") AND TS=("GIS" OR "geospatial" OR "spatial analysis")
```

```
TS=("zonal statistics") AND TS=("raster" OR "GIS") AND TS=("algorithm*" OR "implementation" OR "method*")
```

```
TS=("spatial autocorrelation" OR "Moran" OR "Getis-Ord") AND TS=("raster" OR "grid" OR "continuous") AND TS=("hotspot" OR "cluster" OR "pattern")
```

### Exposición al Viento y Nieve

```
TS=("wind exposure" OR "topex" OR "wind shelter") AND TS=("DEM" OR "terrain" OR "topograph*") AND TS=("algorithm*" OR "GIS" OR "model*")
```

```
TS=("snow redistribution" AND "terrain" AND ("wind" OR "topograph*") AND ("model*" OR "parameter*"))
```

### Rendimiento y Computación

```
TS=("terrain analysis" OR "geomorphometry") AND TS=("parallel" OR "GPU" OR "high performance" OR "Rust" OR "WebAssembly") AND TS=("algorithm*" OR "implementation")
```

```
TS=("digital terrain analysis" AND "benchmark*" AND ("SAGA" OR "GRASS" OR "WhiteboxTools" OR "GDAL"))
```

### Deep Learning + DEM

```
TS=("deep learning" OR "CNN" OR "transformer") AND TS=("DEM" OR "digital elevation model") AND TS=("super resolution" OR "landform" OR "terrain classification")
```

```
TS=("GeoAI" OR "geospatial artificial intelligence") AND TS=("terrain" OR "geomorphometry" OR "landform") AND TS=(2024 OR 2025)
```

---

## 6. Referencias Principales

- Jasiewicz, J. & Stepinski, T. (2013). Geomorphons — a pattern recognition approach to classification and mapping of landforms. *Geomorphology*, 182, 147-156.
- Florinsky, I.V. (2016). *Digital Terrain Analysis in Soil Science and Geology*. Academic Press.
- Gallant, J.C. & Dowling, T.I. (2003). A multiresolution index of valley bottom flatness for mapping depositional areas. *Water Resources Research*, 39(12).
- Hofierka, J. & Šúri, M. (2002). The solar radiation model for Open source GIS: implementation and applications. *Open source GIS-GRASS users conference*.
- Yokoyama, R., Shirasawa, M. & Pike, R.J. (2002). Visualizing topography by openness: A new application of image processing to digital elevation models. *Photogrammetric Engineering & Remote Sensing*, 68(3), 257-265.
- Lindsay, J.B. (2016). Whitebox GAT: A case study in geomorphometric analysis. *Computers & Geosciences*, 95, 75-84.
- Böhner, J. & Antonić, O. (2009). Land-Surface Parameters Specific to Topo-Climatology. In *Geomorphometry*, Elsevier, 195-226.
- Winstral, A. & Marks, D. (2002). Simulating wind fields and snow redistribution using terrain-based parameters. *Hydrological Processes*, 16(18), 3585-3603.
- Riley, S.J., DeGloria, S.D. & Elliot, R. (1999). A terrain ruggedness index. *Intermountain Journal of Sciences*, 5, 23-27.
- Amatulli, G. et al. (2020). Geomorpho90m, empirical evaluation and accuracy assessment of global high-resolution geomorphometric layers. *Scientific Data*, 7, 162.

---

## 7. Ecosistema Rust Geoespacial (contexto)

| Proyecto | Descripción | Estado |
|---|---|---|
| GeoRust (geo, geo-types) | Primitivos y algoritmos geométricos | Maduro |
| GeoArrow-RS | GeoArrow + WASM vectorizado | Activo |
| stac-rs / rustac | STAC en Rust | Producción |
| WhiteboxTools | 550 herramientas geomorfometría | Maduro |
| SurtGis | Análisis geoespacial alto rendimiento | En desarrollo |
| Geo Engine | Plataforma cloud geoespacial | Producción |
| zarrs | Zarr V3 en Rust | Activo |

**Ref:** [Awesome GeoRust](https://github.com/pka/awesome-georust)
