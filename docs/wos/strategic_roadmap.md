# SurtGIS Strategic Roadmap (Data-Driven)

Basado en analisis de **19,062 papers** de Web of Science (15 queries, 2020-2026).

---

## Hallazgos Clave

### 1. Deep Learning domina la literatura (952 menciones)
- Random Forest (367), CNN (361), SVM (135) son los metodos mas citados
- **SurtGIS no tiene ML** — esto es el gap mas grande vs la demanda
- Pero: ML requiere training data, GPU, frameworks pesados. No es nuestro juego.
- **Oportunidad:** Features de ingenieria (input para ML) — SurtGIS genera las variables que alimentan modelos ML

### 2. Change detection es masivo (878 menciones)
- Time series analysis (340) + change detection = workflow dominante
- SurtGIS ya tiene `raster_difference` pero falta multi-temporal
- **Oportunidad:** Pipeline de change detection (diferencia de indices multi-fecha)

### 3. SAR/Radar crece rapido (549 menciones)
- Sentinel-1 (158), radar (332), SAR (549)
- SurtGIS soporta SAR via STAC pero no tiene procesamiento SAR especifico
- **Oportunidad a largo plazo:** Filtrado speckle, coherencia, polarimetria basica

### 4. Time series es trend fuerte (340 menciones)
- Analisis multitemporal de NDVI, deforestacion, degradacion
- SurtGIS descarga una fecha a la vez
- **Oportunidad:** Multi-temporal composite + trend analysis

### 5. Landslide susceptibility es nicho fuerte (214+148 menciones)
- Es el caso de uso #1 en riesgos naturales
- SurtGIS ya tiene pipeline de susceptibilidad
- **Oportunidad:** Fortalecer con mas factores condicionantes

### 6. Kriging sigue vigente (490 menciones) pero ML lo complementa
- IDW, kriging, geostatistics = core de interpolacion
- SurtGIS tiene IDW + kriging + natural neighbor
- **Validado:** Nuestro modulo de interpolacion es relevante

### 7. Conservation/biodiversity es audiencia grande (200+180 menciones)
- Landscape metrics (101), fragmentation (116), connectivity (164)
- SurtGIS tiene Shannon, Simpson, patch density, landscape metrics
- **Validado:** Modulo landscape es estrategico

### 8. Solar radiation es nicho establecido (291 menciones)
- DEM + solar = workflow comun en energia renovable y agricultura
- SurtGIS ya tiene solar_radiation
- **Validado:** Feature unica vs WhiteboxTools/GRASS

### 9. Reproducibilidad es demanda creciente (72 menciones directas)
- FAIR data, workflows reproducibles
- SurtGIS workspace save/load aborda esto directamente
- **Oportunidad:** Export workspace como script Python reproducible

### 10. WebGIS + WASM es emergente pero pequeño (25 menciones)
- Pocos papers sobre analisis raster en browser
- **SurtGIS es pionero** en este espacio
- **Oportunidad:** Publicar paper especifico sobre web-based geospatial WASM

---

## Competidores Mencionados

| Software | Menciones | Tendencia |
|----------|-----------|-----------|
| ENVI | 826 | Comercial, declinando |
| GRASS GIS | 109 | Estable, nicho academico |
| Google Earth Engine | 46 | Creciendo rapido (cloud) |
| STAC ecosystem | 44 | Emergente |
| Rust/WASM | 25 | Muy nuevo |
| QGIS | 13 | Subreportado (se usa mas de lo que se cita) |
| ArcGIS | 6 | Comercial, poco citado en open-source lit |
| WhiteboxTools | 1 | Subcitado para su uso real |

**Insight:** GEE es el competidor a observar. ENVI declina. STAC crece.

---

## Roadmap Priorizado por Evidencia

### TIER 1: Alto impacto, alta demanda (implementar pronto)

#### 1.1 Feature Engineering para ML (NEW)
**Evidencia:** ML es #1 (952 menciones), pero necesita features geoespaciales como input
**Que hacer:**
- `surtgis features generate` — genera stack de variables desde DEM:
  slope, aspect, curvature, TPI, TRI, TWI, HAND, solar radiation, northness, eastness, roughness, DEV, convergence index, valley depth
- Output: multi-band GeoTIFF listo para scikit-learn / random forest / XGBoost
- **Esfuerzo:** ~4h (combinar funciones existentes en pipeline)
- **Impacto:** Conecta SurtGIS con el workflow #1 de la literatura

#### 1.2 Multi-temporal Analysis (NEW)
**Evidencia:** Time series (340), change detection (878)
**Que hacer:**
- Descargar N fechas de un indice via STAC
- Calcular: trend (pendiente lineal), anomalias, diferencia entre fechas
- Visualizar timeline en la demo web
- **Esfuerzo:** ~6h
- **Impacto:** Abre deforestacion, degradacion, monitoreo agricola

#### 1.3 Landslide Susceptibility Mejorado
**Evidencia:** 362 menciones combinadas de landslide + susceptibility
**Que hacer:**
- Agregar factores: distancia a rios, distancia a fallas, litologia, land cover
- Modelo estadistico (frequency ratio, logistic regression simple)
- Output: mapa de susceptibilidad con leyenda de clases
- **Esfuerzo:** ~4h
- **Impacto:** Caso de uso estrella para gobierno/ONGs

### TIER 2: Medio impacto, buena diferenciacion

#### 2.1 Speckle Filter para SAR (NEW)
**Evidencia:** SAR 549 menciones, filtrado es paso basico
**Que hacer:**
- Lee, Frost, Gamma MAP filters
- **Esfuerzo:** ~4h
- **Impacto:** Habilita analisis SAR serio

#### 2.2 Export Workspace como Script Python
**Evidencia:** Reproducibilidad 72 menciones
**Que hacer:**
- Boton "Export as Python" en demo web
- Genera script .py que reproduce el analisis: lee datos, computa, escribe
- **Esfuerzo:** ~3h
- **Impacto:** Puente entre demo web → investigacion reproducible

#### 2.3 Object-Based Classification Basica (OBIA)
**Evidencia:** object-based 128 menciones
**Que hacer:**
- Segmentacion simple (mean-shift o SLIC)
- Clasificacion por segmento (no por pixel)
- **Esfuerzo:** ~8h
- **Impacto:** Diferenciador vs herramientas que solo hacen pixel-based

### TIER 3: Nicho pero estrategico

#### 3.1 Connectivity Analysis
**Evidencia:** connectivity 269 menciones, sobre todo en conservacion
**Que hacer:**
- Least-cost path entre patches
- Circuit theory basica (resistencia/conductancia)
- **Esfuerzo:** ~6h

#### 3.2 LiDAR Point Cloud → DEM
**Evidencia:** LiDAR 353 menciones
**Que hacer:**
- Leer .las/.laz → generar DEM via triangulacion
- **Esfuerzo:** ~8h (requiere nueva dependencia)

#### 3.3 Hyperspectral Support
**Evidencia:** Creciente en clasificacion
**Que hacer:**
- PCA sobre multi-band stack
- Ya tenemos PCA en algoritmos, falta UI
- **Esfuerzo:** ~2h para exponer en web

---

## Lo que NO hacer (bajo ROI segun la evidencia)

| Feature | Razon para no priorizar |
|---------|------------------------|
| Deep learning integrado | Requiere GPU, frameworks pesados. Mejor generar features |
| ArcGIS compatibility | Mercado cerrado, no cita open-source |
| 3D visualization | Nicho, requiere WebGL pesado |
| Real-time streaming | Overengineering para el estado actual |
| Full GEE replacement | GEE tiene infraestructura cloud masiva |

---

## Journals Target (para publicar SurtGIS)

Basado en donde se publican papers sobre software GIS:

| Journal | Relevancia | IF |
|---------|-----------|-----|
| Environmental Modelling & Software | **Ya en revision** | 4.9 |
| Computers & Geosciences | Core GIS software | 4.4 |
| ISPRS Int. J. Geo-Information | Open access, WebGIS | 2.8 |
| Transactions in GIS | Software GIS | 2.4 |
| Earth Science Informatics | Herramientas geo | 2.7 |
| Remote Sensing (MDPI) | Open access, alto volumen | 4.2 |
| SoftwareX | Software papers | 2.4 |

**Segundo paper sugerido:** "SurtGIS Web: Browser-based geospatial analysis via WebAssembly" → ISPRS o SoftwareX

---

## Metricas de Exito (basadas en la literatura)

Lo que la comunidad valora (segun keywords):
1. **Accuracy** (223 menciones) → benchmarks rigurosos
2. **Performance** (175 menciones) → tiempos de ejecucion comparativos
3. **Reproducibility** (72 menciones) → workflows exportables
4. **Open source** (118 menciones) → acceso libre
5. **Classification accuracy** → resultados cuantitativos

SurtGIS ya tiene 1, 2, 3, 4. Falta 5 (validacion de clasificaciones).

---

## Resumen Ejecutivo

**SurtGIS esta bien posicionado en:**
- Terrain analysis (core validado por Q4)
- Spectral indices (17 indices cubren los mas citados)
- Hydrology (watershed, HAND, TWI validados)
- Solar radiation (nicho establecido)
- Landscape metrics (conservacion en alza)
- Web-based analysis (pionero en WASM GIS)
- Reproducibility (workspace save/load)

**SurtGIS necesita mejorar en:**
- **Feature engineering pipeline** (input para ML — gap #1)
- **Multi-temporal analysis** (time series + change detection)
- **SAR processing basico** (speckle filter)
- **Export reproducible** (script Python desde workspace)

**SurtGIS NO debe competir en:**
- Deep learning models (dejar a TensorFlow/PyTorch)
- Cloud infrastructure (dejar a GEE/Planetary Computer)
- Full GIS desktop (dejar a QGIS)

**Posicionamiento optimo:**
> "SurtGIS: High-performance geospatial feature engine for environmental research, deployable as native library, Python package, CLI, or browser application."
