# SurtGIS — Revisión del Estado del Arte

Documento vivo. Se actualiza con cada lote de queries de WOS.

---

## 1. Query: WebAssembly + Rust + Geospatial + Scientific Computing

**Fuente**: Web of Science
**Archivo BibTeX**: `biblio/webassembly.bib`
**Total de referencias**: 22
**Relevantes para SurtGIS**: 5
**Ruido (homonimia de "rust")**: 17

### 1.1 Problema de la query

La palabra "rust" en WOS devuelve tres dominios no deseados:

| Tipo de ruido | Refs | Ejemplos |
|---|---|---|
| Enfermedades de plantas (wheat rust, myrtle rust, blister rust, coffee rust) | 10 | Pryzant 2017, Nazneen 2024, Sandino 2018, Evans 2010/2016, Meyer 2023a/b, da Silva 2020, Van Drunen 2018, Hodson 2007 |
| "Rust Belt" (región industrial de EEUU / China) | 5 | Thompson 2018, Ji 2024, Al Hothaufi 2025, Krzyzanowski 2025, Yang 2025 |
| Corrosión metálica / apellido Rust | 2 | Seechurn 2022, Chen 2005 |

**Recomendación para futuras queries**: excluir explícitamente con `NOT TS=("blister rust" OR "wheat rust" OR "myrtle rust" OR "coffee rust" OR "rust belt" OR "corrosion")` o usar `TS=("Rust programming" OR "Rust language" OR "rustc" OR "WebAssembly")`.

---

### 1.2 Referencias relevantes

#### [R1] Bitar 2024 — Rust and Julia for Scientific Computing

- **Cita**: Bitar, M. (2024). *Computing in Science & Engineering*, 26(1), 72–76.
- **DOI**: 10.1109/MCSE.2024.3369988
- **WOS ID**: WOS:001273791700004
- **Citaciones**: 0

**Resumen**: Comparación de Rust y Julia como lenguajes para computación científica. Julia apunta a cerrar la brecha entre lenguajes dinámicos (Python) y estáticos (C/C++). Rust garantiza seguridad de memoria y máximo rendimiento con un compilador que guía hacia código correcto y concurrente. Ambos apelan al dominio científico por su enfoque en performance.

**Relevancia para SurtGIS**: **Alta**. Valida directamente la elección de Rust como lenguaje para procesamiento geoespacial. Argumenta que Rust es viable para computación científica, su principal desafío es la curva de aprendizaje vs. la ergonomía de Julia/Python, pero la garantía de seguridad de memoria y rendimiento justifican la inversión.

**Hallazgos clave**:
- Rust logra rendimiento comparable a C/C++ sin sacrificar seguridad
- Julia ofrece mejor ergonomía para prototipado, Rust para producción
- La elección depende del tipo de proyecto: Rust es superior cuando se necesita fiabilidad y despliegue (como WASM)

---

#### [R2] Lunnikivi et al. 2020 — Transpiling Python to Rust for Optimized Performance

- **Cita**: Lunnikivi, H., Jylkkä, K., & Hämäläinen, T. (2020). *SAMOS 2020, LNCS 12471*, 127–138.
- **DOI**: 10.1007/978-3-030-60939-9_9
- **WOS ID**: WOS:000884756200009
- **Citaciones**: 5

**Resumen**: Propone transpilar código Python/NumPy a Rust semi-automáticamente. Resultados: hasta **12x speedup** y **1.5x menos memoria** en PC; 4x menos memoria en ARM. Casos de prueba: Black-Scholes y optimización de trayectorias robóticas.

**Relevancia para SurtGIS**: **Media-alta**. Cuantifica la ventaja de Rust sobre Python para cómputo numérico — exactamente el argumento de por qué SurtGIS en Rust supera a rasterio/GDAL-Python. Los speedups reportados (12x) son consistentes con lo que se observa en los benchmarks de SurtGIS.

**Hallazgos clave**:
- Python con NumPy transpilado a Rust: 12x más rápido, 1.5x menos memoria
- Rendimiento similar en ARM pero con 4x menos memoria (relevante para edge/WASM)
- La transpilación semi-automática es viable, sugiere que el ecosistema Rust puede absorber workloads científicos de Python

---

#### [R3] Ionita et al. 2020 — Efficient Parallel Simulations of Wireless Signal Wave Propagation

- **Cita**: Ionita, D.-M., Manole, F.-G., & Slusanschi, E.-I. (2020). *SYNASC 2020*, 55–62.
- **DOI**: 10.1109/SYNASC51798.2020.00020
- **WOS ID**: WOS:000674702000009
- **Citaciones**: 0

**Resumen**: Compara implementaciones paralelas de un patrón stencil 2D (propagación de señal) en CUDA, TBB, Rust, OpenMP y HLS/FPGA. CUDA es la más rápida; Rust compite con TBB y OpenMP en CPU multi-core.

**Relevancia para SurtGIS**: **Media**. Los algoritmos de terreno de SurtGIS (slope, aspect, hillshade, focal statistics) son esencialmente operaciones stencil sobre grids 2D. Este paper valida que Rust con rayon es competitivo frente a TBB/OpenMP para este patrón. La limitación WASM (single-thread) es consistente con la arquitectura `maybe_rayon` de SurtGIS.

**Hallazgos clave**:
- Rust es competitivo con TBB/OpenMP en operaciones stencil sobre grids
- CUDA domina en GPU pero requiere hardware específico
- El patrón stencil es exactamente lo que usan los algoritmos de terreno

---

#### [R4] Thaker et al. 2021 — Clamor: Extending Functional Cluster Computing with Fine-Grained Remote Memory Access

- **Cita**: Thaker, P., Ayers, H., Raghavan, D., et al. (2021). *SoCC '21*, 654–669.
- **DOI**: 10.1145/3472883.3486996
- **WOS ID**: WOS:000768717900046
- **Citaciones**: 0

**Resumen**: Framework de cluster computing en Rust que integra con librerías Rust/C++ existentes. Mejora de 5x a 100x vs Spark en workloads con acceso disperso a variables globales, incluyendo **queries geoespaciales distribuidas**.

**Relevancia para SurtGIS**: **Media**. Demuestra que Rust es viable para procesamiento geoespacial distribuido a gran escala. Las queries geoespaciales distribuidas con >100x mejora sobre Spark sugieren un camino de escalabilidad futura para SurtGIS más allá de single-node.

**Hallazgos clave**:
- Rust para procesamiento geoespacial distribuido: >100x sobre Spark
- El acceso fine-grained a memoria remota beneficia workloads con patrones de acceso dispersos (típico de análisis espacial)
- Demuestra interoperabilidad de Rust con librerías existentes en clusters

---

#### [R5] Moses et al. 2021 — Reverse-Mode Automatic Differentiation and Optimization of GPU/CPU Kernels via Enzyme

- **Cita**: Moses, W. S., Churavy, V., Paehler, L., et al. (2021). *SC21*.
- **DOI**: 10.1145/3458817.3476165
- **WOS ID**: WOS:000946520100017
- **Citaciones**: 21

**Resumen**: Enzyme, plugin de LLVM que genera gradientes automáticos de alto rendimiento para C/C++, Fortran, Julia y Rust. Primer AD reverso completamente automático para kernels GPU. Overhead dentro de un orden de magnitud del programa original.

**Relevancia para SurtGIS**: **Baja-media**. Rust es uno de los lenguajes soportados. Enzyme podría ser útil en el futuro para algoritmos que requieran optimización basada en gradientes (inversión de parámetros geofísicos, calibración de modelos). No es inmediatamente aplicable, pero muestra que el ecosistema LLVM/Rust está madurando para HPC serio.

**Hallazgos clave**:
- El ecosistema LLVM beneficia a Rust con herramientas avanzadas de optimización
- Diferenciación automática ahora funciona para Rust
- Los gradientes generados corren dentro de ~10x del programa original

---

### 1.3 Referencias descartadas (ruido)

| # | Clave WOS | Título abreviado | Razón de descarte |
|---|---|---|---|
| 1 | WOS:000861689200001 | Corrosión atmosférica de acero al carbono | Corrosión metálica ("rust") |
| 2 | WOS:000426448300189 | Monitoring Ethiopian Wheat Fungus | Roya del trigo (wheat rust) |
| 3 | WOS:001176592100001 | Disease spectrum in lentil production | Roya de lenteja |
| 4 | WOS:000446291100041 | Tracking building removal in Rust Belt cities | Rust Belt (geografía) |
| 5 | WOS:000435574800016 | Aerial Mapping of Forests with UAVs | Roya del mirto (myrtle rust) |
| 6 | WOS:000283023900005 | Spread Rates of Exotic Diseases in Forests | Roya del pino (blister rust) |
| 7 | WOS:000937585900001 | Whitebark pine ecosystems in California | Roya del pino |
| 8 | WOS:000590009000007 | Coffee field nutritional balance | Roya del café |
| 9 | WOS:001317743600001 | Public service supply in China's rust belt | Rust Belt (China) |
| 10 | WOS:001014039800001 | 3D Visualization of Crop Pathogen Transport | Roya del trigo |
| 11 | WOS:000233134300003 | S-system computation of gamma distribution | Apellido "Rust" en referencias |
| 12 | WOS:000440775600008 | Fungal infection of American chestnut | Tizón del castaño |
| 13 | WOS:001583930200002 | Urban spaces into green infrastructure | Rust Belt en keywords |
| 14 | WOS:000377793800008 | Speed of Invasion: Exotic Forest Insects | Roya del pino |
| 15 | WOS:001659490900001 | Trichloroethylene and Parkinson Disease | Rust Belt en abstract |
| 16 | WOS:001612782900001 | Population Shrinkage in "Rust Belt City" | Rust Belt (China) |
| 17 | WOS:000245455200002 | Spatial analyses for wheat production | Roya del trigo (Ug99) |

---

### 1.4 Síntesis y hallazgos para SurtGIS

#### Vacío en la literatura

De 22 resultados en WOS, **ninguno trata directamente sobre WebAssembly para procesamiento geoespacial**. Esto confirma que el nicho de SurtGIS — procesamiento raster en el navegador vía WASM compilado desde Rust — está **esencialmente inexplorado** en la literatura científica indexada.

#### Evidencia que respalda la arquitectura de SurtGIS

| Aspecto | Evidencia | Fuente |
|---|---|---|
| Rust viable para computación científica | Comparable a C/C++ en rendimiento | Bitar 2024, Lunnikivi 2020 |
| Speedup Rust vs Python | Hasta 12x en cómputo numérico | Lunnikivi 2020 |
| Rust competitivo en stencil/grid (como terreno) | A la par de TBB/OpenMP | Ionita 2020 |
| Rust para queries geoespaciales distribuidas | >100x vs Spark | Thaker 2021 |
| Ecosistema LLVM madura para Rust HPC | AD automático, optimizaciones GPU | Moses 2021 |

#### Oportunidades de publicación

1. **Paper principal**: "Client-side geospatial raster processing via WebAssembly: the SurtGIS approach" — no existe nada comparable en WOS
2. **Benchmark paper**: Comparar SurtGIS-WASM vs GDAL/Python vs GRASS para algoritmos de terreno en browser
3. **Application note**: Demo de procesamiento offline (PWA + WASM) para trabajo de campo sin conectividad

---

## 2. Query: Terrain Analysis

**Fuente**: Web of Science
**Archivos BibTeX**: `biblio/terrain_v1.bib` (16), `terrain_v2.bib` (3), `terrain_v3.bib` (545), `terrain_v4.bib` (22)
**Total de referencias**: 586
**Relevantes para SurtGIS (Alta)**: 23
**Aplicaciones (Media/Baja)**: ~320
**Ruido**: ~243

### 2.1 Distribución por archivo

| Archivo | Entries | Alta | Media | Baja | Ruido |
|---|---|---|---|---|---|
| terrain_v1.bib | 16 | 2 | 5 | 7 | 2 |
| terrain_v2.bib | 3 | 1 | 2 | 0 | 0 |
| terrain_v3.bib | 545 | ~15 | ~60 | ~250 | ~220 |
| terrain_v4.bib | 22 | 5 | 5 | 8 | 4 |

### 2.2 Observación general

A diferencia de la query WebAssembly+Rust (donde el vacío era absoluto), la query Terrain devuelve abundante literatura. Sin embargo, la mayoría son **aplicaciones** de derivados del terreno (slope, aspect, hillshade como inputs para ML, mapeo de suelos, clasificación de deslizamientos), no papers sobre los **algoritmos** en sí. Los papers de alta relevancia que describen o benchmarkean los algoritmos que SurtGIS implementa son la minoría.

---

### 2.3 Referencias de alta relevancia

#### [T1] Zhang et al. 2021 — Parallel Computing Approach to Slope via Spark

- **Cita**: Zhang, J. et al. (2021). *Sensors*.
- **WOS ID**: WOS:000611723100001
- **Citaciones**: 8
- **Archivo**: terrain_v1.bib

**Resumen**: Presenta un enfoque de cómputo paralelo con Spark para algoritmos focales de análisis de vecindad en datos de terreno. Implementa slope con una estrategia de cálculo de ventana dinámica (DCW) sobre tiles de DEM particionados. Resultados con DEM australiano muestran rendimiento paralelo efectivo con precisión equivalente a ArcGIS.

**Relevancia para SurtGIS**: **Alta**. Aborda directamente la paralelización de focal statistics y cómputo de slope sobre rasters de terreno — ambos algoritmos que SurtGIS implementa. Las estrategias de tiling, computación por ventana y ejecución paralela son relevantes para la arquitectura Rust/WASM de SurtGIS, especialmente el crate `parallel` y el módulo `focal`.

---

#### [T2] Bakhsoliani et al. 2025 — Landslide Mapping Using 7 Terrain Indices

- **Cita**: Bakhsoliani, D. et al. (2025). *Geohazards*.
- **WOS ID**: WOS:001646609400001
- **Citaciones**: 0
- **Archivo**: terrain_v1.bib

**Resumen**: Identifica límites de deslizamientos usando DEM de alta resolución (UAV LiDAR). Analiza slope, aspect, curvatura, hillshade, TRI, TPI y TWI. La combinación de estos parámetros + PCA permite identificar límites de deslizamientos.

**Relevancia para SurtGIS**: **Alta**. Usa directamente 7 de los algoritmos que SurtGIS implementa (slope, aspect, curvature, hillshade, TRI, TPI, TWI) en un workflow práctico. Valida la utilidad del conjunto completo de algoritmos de terreno de la librería.

---

#### [T3] Florinsky 2023 — 17 Variables Morfométricas en Antártida

- **Cita**: Florinsky, I. V. (2023). *Polar Science*.
- **DOI**: 10.1016/j.polar.2023.100969
- **WOS ID**: WOS:001142742600001
- **Citaciones**: 1
- **Archivo**: terrain_v2.bib

**Resumen**: Primer modelado geomorfométrico de las Larsemann Hills (Antártida) usando REMA. Deriva **17 variables morfométricas**: slope, aspect, curvatura horizontal, vertical, media, Gaussiana, mínima, máxima, unesfericidad, diferencia, exceso vertical, exceso horizontal, anillo, acumulación, área de captación, índice topográfico y SPI. Validado en campo con 54 rutas (422 km).

**Relevancia para SurtGIS**: **Alta**. Caso de estudio directo para las capacidades de SurtGIS. De las 17 variables, SurtGIS ya implementa: slope, aspect, curvatura (horizontal, vertical, media), TWI, SPI. Las curvaturas avanzadas (Gaussiana, mínima, máxima, unesfericidad, diferencia, exceso, anillo, acumulación) son un roadmap concreto para extender el módulo `multiscale_curvatures`. Referencia primaria para validar outputs de SurtGIS.

---

#### [T4] Newman et al. 2023 — Heterogeneous Multiscale Scaling Strategies

- **Cita**: Newman, D.R. et al. (2023). *Environmental Modelling & Software*.
- **WOS ID**: WOS:000994345600001
- **Citaciones**: 5
- **Archivo**: terrain_v4.bib

**Resumen**: Compara rendimiento de modelos predictivos usando diferentes estrategias de escalado multiescala. El escalado heterogéneo que desacopla escalas de proceso de parametrización analítica supera consistentemente a otros enfoques. Los datos sin escalar rindieron peor en todos los casos.

**Relevancia para SurtGIS**: **Alta**. Directamente relevante para el análisis multiescala de SurtGIS (`multiscale_curvatures`, smoothing). Provee marco teórico para cómo SurtGIS debería exponer parámetros de escala en su API.

---

#### [T5] Newman et al. 2022 — Gaussian Scale-Space as Optimal Scaling

- **Cita**: Newman, D.R. et al. (2022). *Geomatics*.
- **WOS ID**: WOS:001254207000001
- **Citaciones**: 9
- **Archivo**: terrain_v4.bib

**Resumen**: Evalúa enfoques de escalado en geomorfometría: interpolación directa, convolución cúbica, agregación media, regresión cuadrática local, y Gaussian scale-space (fGSS). El método fGSS combina las propiedades deseables e identifica como el método óptimo de escalado.

**Relevancia para SurtGIS**: **Alta**. Paper metodológico central para análisis multiescala. El framework Gaussian scale-space es directamente aplicable a las implementaciones de `multiscale_curvatures` y `smoothing` de SurtGIS. El método fGSS podría ser un objetivo de implementación de referencia.

---

#### [T6] Newman et al. 2018 — Integral Images for Local Topographic Position

- **Cita**: Newman, D.R. et al. (2018). *Geomorphology*.
- **WOS ID**: WOS:000432501100004
- **Citaciones**: 43
- **Archivo**: terrain_v4.bib

**Resumen**: Evalúa tres métricas de posición topográfica local (LTP): DEV (desviación de elevación media), PER (rango porcentual de elevación) y RTP (posición topográfica relativa). DEV, computado usando **imágenes integrales**, ofrece cómputo rápido e invariante a escala con fuerte correlación con percentil de elevación (r² 0.699–0.967).

**Relevancia para SurtGIS**: **Alta**. Directamente relevante para la implementación de TPI de SurtGIS. La técnica de imágenes integrales para calcular DEV es aplicable para cómputo eficiente de TPI/DEV. El framework de comparación de métricas LTP informa decisiones de implementación.

---

#### [T7] Wang & Laffan 2009 — Multi-Scale Valleyness vs MRVBF

- **Cita**: Wang, D. et al. (2009). *MODSIM09*.
- **WOS ID**: WOS:000290045002010
- **Citaciones**: 9
- **Archivo**: terrain_v4.bib

**Resumen**: Desarrolla un método de *valleyness* multiescala (MSV) usando ajuste de función cuadrática para caracterizar rasgos morfométricos difusos. Compara MSV contra D8, D-infinity, flujo acumulado y **MrVBF**. MSV captura áreas no detectadas por otros métodos; fiabilidad negativamente correlacionada con complejidad del terreno.

**Relevancia para SurtGIS**: **Alta**. Compara directamente enfoques alternativos para identificación de fondos de valle, incluyendo MRVBF que SurtGIS implementa. El concepto MSV basado en superficies cuadráticas es relevante para mejoras potenciales al algoritmo MRVBF.

---

#### [T8] Moeller et al. 2008 — Hierarchical Multiscale Terrain Classification

- **Cita**: Moeller, M. et al. (2008). *Journal of Plant Nutrition and Soil Science*.
- **WOS ID**: WOS:000257172700014
- **Citaciones**: 54
- **Archivo**: terrain_v4.bib

**Resumen**: Propone un procedimiento de clasificación jerárquica del terreno usando segmentación basada en regiones a múltiples escalas con el índice de balance de masas. Los objetos de paisaje se definen a múltiples escalas permitiendo contexto jerárquico. Logra 89% de precisión en clasificación de geoformas en Sajonia-Anhalt (Alemania).

**Relevancia para SurtGIS**: **Alta**. Paper fundacional para clasificación multiescala del terreno. La clasificación jerárquica usando segmentación multiescala es relevante para los módulos `landform` y `multiscale_curvatures` de SurtGIS. El concepto de índice de balance de masas se relaciona con procesos de transporte del terreno.

---

### 2.4 Referencias de relevancia media

| # | WOS ID | Autor | Año | Título abreviado | Citaciones | Archivo | Nota |
|---|---|---|---|---|---|---|---|
| 1 | WOS:001341220100001 | Shahabi, H. | 2024 | Deep learning + LiDAR landslide mapping | 6 | v1 | Slope+hillshade superan modelo con 7 features |
| 2 | WOS:000867557000017 | Tu, H. | 2022 | LST downscaling con hillshade, SVF, solar | 8 | v1 | Valida hillshade, SVF, radiación solar |
| 3 | WOS:000747874400001 | Garajeh, M.K. | 2022 | OBIA desert landform con slope/aspect | 18 | v1 | Clasificación de geoformas con derivados |
| 4 | WOS:000835287100006 | Lemenkova, P. | 2022 | GMT/R scripting geomorphometric mapping | 3 | v1 | Workflow reproducible con slope/aspect/hillshade |
| 5 | WOS:000321084800010 | Janke, J.R. | 2013 | LiDAR vs USGS DEM comparison | 24 | v1 | RMSE entre fuentes para slope/aspect/curvature |
| 6 | WOS:001372130900001 | Popov, A.B. | 2025 | Physically-based geomorphological mapping | 4 | v2 | Segmentación con slope/aspect/curvatures |
| 7 | WOS:001432027000002 | Pourfarrashzadeh, F. | 2025 | Ecogeomorphic deforestation modeling | 2 | v2 | TPI, TWI, TRI, curvature como predictores |
| 8 | WOS:000412257100014 | Pipaud, I. | 2017 | Object-based alluvial fan classification | 24 | v4 | Segmentación multiescala con morfometría |
| 9 | WOS:000314989302061 | Summerell, G.K. | 2011 | MRVBF+DEM for groundwater surface | 0 | v4 | Combinación MRVBF normalizado con DEM |
| 10 | WOS:000290030701056 | McKergow, L.A. | 2007 | TWI vs MRVBF vs FLAG for wetlands | 2 | v4 | Comparación práctica TWI/MRVBF |
| 11 | WOS:000290114101068 | Murphy, B. | 2005 | TWI+MRVBF for soil depth | 7 | v4 | Uso combinado TWI+MRVBF |
| 12 | WOS:000290114101074 | Summerell, G.K. | 2005 | MRVBF+slope for channel incision | 0 | v4 | Workflow MRVBF+slope |

---

### 2.5 Hallazgos de terrain_v3.bib (545 entries)

Archivo masivo con 545 entradas, incluyendo papers seminales con cientos de citas. La distribución de citaciones muestra 3 papers con >500 citas, 21 con 100–499, y 22 con 50–99.

#### Papers fundacionales (de terrain_v3.bib)

#### [T9] Jasiewicz & Stepinski 2013 — GEOMORPHONS (Paper Fundacional)

- **Cita**: Jasiewicz, J. & Stepinski, T.F. (2013). *Geomorphology*, 182, 147–156.
- **WOS ID**: WOS:000314328800012
- **Citaciones**: **580**
- **Archivo**: terrain_v3.bib

**Resumen**: Introduce el método geomorphon — clasificación de geoformas basada en reconocimiento de patrones ternarios en lugar de geometría diferencial. Define 498 tipos posibles de morfología del terreno a partir de un solo escaneo de DEM que se auto-adapta a la escala espacial en cada ubicación. Demostrado a escala continental (Polonia, 30m).

**Relevancia para SurtGIS**: **Fundacional**. Este es el paper original del algoritmo `geomorphons` que SurtGIS implementa directamente. La segunda referencia más citada en toda la bibliografía de terreno.

---

#### [T10] Amatulli et al. 2018 — Suite Global de Variables Topográficas

- **Cita**: Amatulli, G. et al. (2018). *Scientific Data*, 5.
- **WOS ID**: WOS:000427902600001
- **Citaciones**: **623**
- **Archivo**: terrain_v3.bib

**Resumen**: Genera una suite global de variables topográficas a partir de GMTED2010 (250m) y SRTM4.1dev (90m): elevation, slope, aspect, eastness, northness, roughness, TRI, TPI, VRM, curvatura de perfil/tangencial, derivadas parciales de primer/segundo orden, y 10 clases geomorfológicas. Agregado a múltiples granos espaciales (1–100 km).

**Relevancia para SurtGIS**: **Fundacional como benchmark**. La referencia más citada de toda la bibliografía (623 citas). Cubre prácticamente todos los algoritmos que SurtGIS implementa. El dataset global resultante es un recurso de validación excelente para verificar la corrección de los outputs de SurtGIS.

---

#### [T11] De Reu et al. 2013 — TPI: Limitaciones en Paisajes Heterogéneos

- **Cita**: De Reu, J. et al. (2013). *Geomorphology*, 186, 39–49.
- **WOS ID**: WOS:000315974400004
- **Citaciones**: **509**
- **Archivo**: terrain_v3.bib

**Resumen**: Evalúa TPI para clasificación automatizada de geoformas en un proyecto geoarqueológico en Bélgica. Encuentra que TPI produce clasificaciones erróneas en paisajes heterogéneos y propone DEV (Deviation from Mean Elevation) como alternativa superior.

**Relevancia para SurtGIS**: **Alta**. Evaluación directa del algoritmo TPI que SurtGIS implementa, identificando sus limitaciones y proponiendo mejoras. Tercera referencia más citada (509 citas). Esencial para documentar cuándo TPI funciona bien vs. mal.

---

#### [T12] Kramm et al. 2017 — Benchmark: Geomorphons vs TPI vs Dikau vs OBIA

- **Cita**: Kramm, T. et al. (2017). *ISPRS Int. J. Geo-Information*, 6(11).
- **WOS ID**: WOS:000416779300047
- **Citaciones**: **227**
- **Archivo**: terrain_v3.bib

**Resumen**: Compara cuatro métodos de clasificación de geoformas — Dikau stepwise, geomorphons, TPI y OBIA — en cuatro DEMs (5m–30m). TPI y geomorphons logran >70% de precisión a alta resolución y superan a otros enfoques en todos los datasets.

**Relevancia para SurtGIS**: **Alta**. Benchmark directo de dos algoritmos que SurtGIS implementa (TPI y geomorphons). Provee métricas de precisión por resolución de DEM, esencial para validación y documentación.

---

#### [T13] Jasiewicz, Netzel & Stepinski 2014 — Similitud de Paisaje con Geomorphons

- **Cita**: Jasiewicz, J. et al. (2014). *Geomorphology*, 221, 104–112.
- **WOS ID**: WOS:000340338000008
- **Citaciones**: 76
- **Archivo**: terrain_v3.bib

**Resumen**: Extiende geomorphons para medir similitud numérica entre paisajes basada en patrones de geoformas. Clasifica topografía de Polonia (30m) en 10 tipos e introduce TerraEx-Pl, un motor de búsqueda de paisajes.

**Relevancia para SurtGIS**: **Alta**. Extensión directa del algoritmo geomorphons. Demuestra aplicaciones avanzadas sobre la base que SurtGIS ya implementa.

---

#### [T14] Gruber et al. 2017 — Algoritmos vs. Topógrafos: Comparación de TPI/TWI

- **Cita**: Gruber, F.E. et al. (2017). *Geoderma*, 308, 9–25.
- **WOS ID**: WOS:000413282100002
- **Citaciones**: 132
- **Archivo**: terrain_v3.bib

**Resumen**: Compara 5 algoritmos open-source de clasificación de geoformas incluyendo TPI y TWI contra evaluación de campo por topógrafos. TPI con radio de búsqueda 70m es el más influyente para posición mesoescala.

**Relevancia para SurtGIS**: **Alta**. Evaluación de campo de los algoritmos TPI y TWI que SurtGIS implementa. Provee ground-truth para calibración de parámetros.

---

#### Papers con relevancia algorítmica adicional (terrain_v3.bib)

| # | WOS ID | Autor | Año | Título abreviado | Cit. | Algoritmos SurtGIS |
|---|---|---|---|---|---|---|
| T15 | WOS:001559348200001 | Lipka, E. | 2025 | Geomorphons vs TPI for glacial features | 0 | geomorphons, TPI |
| T16 | WOS:001489645900013 | Li, J. | 2025 | Geomorphon-based lunar landform classification | 0 | geomorphons |
| T17 | WOS:001635365000001 | Kranjcic, N. | 2025 | Settlement zones: TPI+TRI+TWI+SVF+solar | 0 | TPI, TRI, TWI, SVF, solar radiation |
| T18 | WOS:000569691000001 | Thompson, A. | 2020 | Maya settlements: SVF+TPI evaluation | 28 | SVF, TPI |
| T19 | WOS:000746190200001 | Brunier, G. | 2022 | Reef mapping: openness+TPI+geomorphons | 5 | openness, TPI, geomorphons |

#### Inventario de algoritmos en terrain_v3.bib

| Algoritmo SurtGIS | Papers específicos | Paper más citado |
|---|---|---|
| geomorphons | 17 | Jasiewicz & Stepinski 2013 (580 cit.) |
| TPI | ~545 (eje de toda la query) | De Reu et al. 2013 (509 cit.) |
| TRI | ~50 (como covariate) | Amatulli et al. 2018 (623 cit.) |
| TWI | ~80 (como covariate) | Arulbalaji et al. 2019 (448 cit.) |
| MRVBF | ~10 | Dharumarajan et al. 2017 (116 cit.) |
| SVF | 3 | Thompson 2020 (28 cit.) |
| Openness | 4 | Brunier et al. 2022 (5 cit.) |
| Solar radiation | 2 | Kranjcic et al. 2025 (0 cit.) |
| Viewshed | 3 | — |
| Wind exposure | 7 (como factor condicionante) | — |
| Flow direction | 3 | — |
| Multiscale curvature | 0 | — |

#### Distribución temática de las 545 referencias

| Dominio de aplicación | % estimado |
|---|---|
| Deslizamientos / riesgo geomorfológico | ~35% |
| Potencial de aguas subterráneas | ~20% |
| Ecología / distribución de especies | ~15% |
| Ciencia de suelos | ~10% |
| Arqueología / prospección | ~5% |
| Nieve / criosfera | ~5% |
| Susceptibilidad a inundaciones | ~5% |
| Susceptibilidad a incendios | ~5% |

---

### 2.6 Síntesis y hallazgos para SurtGIS

#### Patrón dominante

La gran mayoría de la bibliografía de terreno usa derivados del DEM (slope, aspect, hillshade, TRI, TPI, TWI, MRVBF) como **inputs para modelos** (ML, regresión logística, OBIA) en aplicaciones de:
- Deslizamientos y riesgo geomorfológico
- Mapeo digital de suelos
- Hidrología y delimitación de humedales
- Clasificación de geoformas

Los papers que abordan los **algoritmos en sí** (cómputo, benchmark, escalado) son una minoría clara, concentrados en los trabajos de Newman (series 2018–2023) sobre escalado multiescala.

#### Vacíos identificados

1. **No hay benchmarks de rendimiento** comparando implementaciones de algoritmos de terreno entre herramientas (GDAL vs GRASS vs SAGA vs SurtGIS)
2. **No hay papers sobre procesamiento de terreno en WASM** — el nicho permanece inexplorado
3. **Las curvaturas avanzadas** (Gaussiana, unesfericidad, anillo, etc.) de Florinsky 2023 son un área de expansión clara para SurtGIS
4. **Papers fundacionales ausentes** en la bibliografía para varios algoritmos SurtGIS:
   - MRVBF: Gallant & Dowling (2003)
   - Sky View Factor: Zakšek et al. (2011)
   - Topographic Openness: Yokoyama et al. (2002)
   - Flow Direction D8: O'Callaghan & Mark (1984)
   - Flow Direction D-infinity: Tarboton (1997)
   - TWI/SPI/STI: Wilson & Gallant (2000)
   - Multiscale Curvature: Minár & Evans (2008)

#### Papers fundacionales del dominio

| Paper | Citas | Algoritmo | Rol |
|---|---|---|---|
| Amatulli et al. 2018 | 623 | Suite completa (slope, aspect, TRI, TPI, curvature, landforms) | Benchmark global |
| Jasiewicz & Stepinski 2013 | 580 | Geomorphons | Paper original |
| De Reu et al. 2013 | 509 | TPI | Evaluación y limitaciones |
| Kramm et al. 2017 | 227 | Geomorphons vs TPI | Benchmark comparativo |
| Gruber et al. 2017 | 132 | TPI vs TWI | Ground-truth vs topógrafos |

#### Líneas de trabajo derivadas

| Línea | Base bibliográfica | Acción SurtGIS |
|---|---|---|
| Gaussian scale-space para multiescala | Newman 2022, 2023 | Implementar fGSS como framework en `smoothing` |
| Imágenes integrales para TPI | Newman 2018, De Reu 2013 | Optimizar TPI con integral images + DEV |
| Curvaturas avanzadas (17 vars) | Florinsky 2023 | Extender `curvature` y `multiscale_curvatures` |
| MSV como complemento a MRVBF | Wang 2009 | Evaluar implementación de valleyness multiescala |
| Validación contra suite global | Amatulli 2018 | Comparar outputs SurtGIS vs dataset global |
| Benchmark multiherramienta | (vacío) | Paper SurtGIS vs GDAL vs GRASS vs SAGA |
| DEV como alternativa a TPI | De Reu 2013, Newman 2018 | Implementar DEV junto a TPI |

## 3. Query: Hydrology / Depression Filling

**Fuente**: Web of Science
**Archivos BibTeX**: `biblio/hydrology_v1.bib` (20), `hydrology_v2.bib` (275)
**Total de referencias**: 295
**Relevantes para SurtGIS (Alta)**: 32 (v1: 18, v2: 14)
**Aplicaciones (Media/Baja)**: 73 (v1: 2, v2: 71)
**Ruido**: 190 (v1: 0, v2: 190)

### 3.1 Observación general

La bibliografía de hidrología es **extraordinariamente enfocada**: el 90% de las entradas en v1 son de alta relevancia directa para los algoritmos que SurtGIS implementa. El dominio está dominado por variantes del algoritmo **Priority-Flood** para depression filling y sus extensiones a flow direction y watershed delineation.

### 3.2 Distribución por archivo

| Archivo | Entries | Alta | Media | Baja | Ruido |
|---|---|---|---|---|---|
| hydrology_v1.bib | 20 | 18 | 2 | 0 | 0 |
| hydrology_v2.bib | 275 | 14 | 29 | 42 | 190 |

---

### 3.3 Referencias de alta relevancia (v1)

#### Paper fundacional

#### [H1] Barnes et al. 2014 — Priority-Flood: El Algoritmo Fundacional

- **Cita**: Barnes, R., Lehman, C. & Mulla, D. (2014). *Computers & Geosciences*, 62, 117–127.
- **WOS ID**: WOS:000328724500013
- **Citaciones**: **181**
- **Archivo**: hydrology_v1.bib

**Resumen**: Presenta el algoritmo Priority-Flood, que unifica y mejora trabajos previos. Inunda DEMs desde los bordes usando una cola de prioridad. Óptimo para enteros O(n) y flotantes O(n log n). Una variación mejorada usando cola simple para depresiones detectadas logra O(m log m), hasta 37% más rápido. Pseudocódigo de 20 líneas e implementación C++ de <100 líneas. Funciona en mallas irregulares y grids n-conectados. Puede etiquetar watersheds y determinar direcciones de flujo.

**Relevancia para SurtGIS**: **Fundacional**. La referencia principal para depression filling moderno. Su simplicidad (20 líneas de pseudocódigo) y garantías de optimalidad lo hacen la referencia primaria para `fill_depressions` de SurtGIS.

---

#### Variantes y optimizaciones del Priority-Flood

#### [H2] Barnes 2016 — Parallel Priority-Flood para DEMs de Trillones de Celdas

- **Cita**: Barnes, R. (2016). *Computers & Geosciences*, 96, 56–68.
- **WOS ID**: WOS:000384855300006
- **Citaciones**: 39

**Resumen**: Priority-Flood paralelo con escalado lineal que subdivide DEMs en tiles. Diseño single-producer/multi-consumer. Garantiza acceso fijo a memoria por subdivisión. Probado en DEMs de hasta 2×10¹² celdas. ~60% de escalado fuerte/débil a 48 cores.

**Relevancia para SurtGIS**: **Alta**. Paralelización por tiles directamente relevante para el crate `parallel` y la arquitectura Rayon/maybe_rayon de SurtGIS.

---

#### [H3] Zhou et al. 2016 — Priority-Flood Eficiente con Region-Growing

- **Cita**: Zhou, G. et al. (2016). *Computers & Geosciences*, 90, 87–96.
- **WOS ID**: WOS:000374807400009
- **Citaciones**: 42

**Resumen**: Reduce operaciones de cola de prioridad usando region-growing para celdas sin depresión/planas. Tres implementaciones (two-pass, one-pass, direct). One-pass promedia 44.6% de speed-up sobre la variante existente más rápida.

---

#### [H4] Zhou et al. 2017 — Priority-Flood Paralelo (OpenMP/MPI)

- **Cita**: Zhou, G. et al. (2017). *International Journal of Geographical Information Science*, 31(6), 1061–1078.
- **WOS ID**: WOS:000399593300001
- **Citaciones**: 14

**Resumen**: Primera implementación paralela del Priority-Flood. Particiona DEMs en franjas, procesa cada una con variante secuencial, identifica celdas mal clasificadas. >4x con 8 threads (OpenMP), >8x sobre TauDEM (MPI).

---

#### [H5] Wei et al. 2019 — Priority-Flood con 70% Menos Celdas en Cola

- **Cita**: Wei, H. et al. (2019). *International Journal of Digital Earth*, 12(4), 390–402.
- **WOS ID**: WOS:000461865100003
- **Citaciones**: 23

**Resumen**: Reduce las celdas procesadas por la cola de prioridad en ~70%. Speed-up de 31–52% (promedio 45%) sobre la variante más rápida existente.

---

#### [H6] Xiong et al. 2019 — Priority-Flood Optimizado para Micro-Relieve

- **Cita**: Xiong, L.-Y. et al. (2019). *Transactions in GIS*, 23(6), 1276–1293.
- **WOS ID**: WOS:000468313900005
- **Citaciones**: 10

**Resumen**: Elimina cálculos redundantes en áreas de micro-relieve local. Usa quick queue para depresiones y flatlands. Hasta 43.2% más rápido que Wang-Liu PF, promedio 31.8%.

---

#### [H7] Condon & Maxwell 2019 — Priority-Flood Modificado con Slope Enforcement

- **Cita**: Condon, L.E. & Maxwell, R.M. (2019). *Computers & Geosciences*, 126, 145–151.
- **WOS ID**: WOS:000464299400008
- **Citaciones**: 26

**Resumen**: PriorityFlow: combina Priority-Flood modificado con información a-priori de red de drenaje, cálculos de slope mejorados incorporando flow directions, y definición flexible de sub-cuencas. Implementación open-source en GitHub.

---

#### Variantes de Planchon-Darboux

#### [H8] Wei et al. 2019 — Variante Mejorada de Planchon-Darboux

- **Cita**: Wei, H. et al. (2019). *ISPRS Int. J. Geo-Information*, 8(7), 290.
- **WOS ID**: WOS:000467499300005
- **Citaciones**: 1

**Resumen**: Reduce cómputo redundante y uso de memoria del algoritmo P&D. Soporta superficies horizontales estrictas o ligeramente inclinadas (esta última facilita el cálculo de flow direction).

---

#### Benchmarks y comparaciones

#### [H9] Sharma & Tiwari 2019 — Benchmark JD vs PD vs Lindsay

- **Cita**: Sharma, A. & Tiwari, K.N. (2019). *Current Science*, 117(9), 1511–1519.
- **WOS ID**: WOS:000494956300030
- **Citaciones**: 5

**Resumen**: Compara tres algoritmos de sink removal: Jenson-Domingue, Planchon-Darboux y Lindsay. P&D tiene el menor impacto en el DEM original, aunque todos alteran significativamente el terreno.

---

#### [H10] Senevirathne & Willgoose 2013 — Benchmark JD vs PD Escalabilidad

- **Cita**: Senevirathne, N. & Willgoose, G. (2013). *MODSIM2013*.
- **WOS ID**: WOS:000357105901095
- **Citaciones**: 1

**Resumen**: Planchon escala linealmente con el tamaño del DEM mientras Jenson escala exponencialmente. Ambos producen artefactos de flujo paralelo en áreas grandes rellenadas.

---

#### Flow direction y alternativas

#### [H11] Wu et al. 2025 — DZFlood: Flow Direction con Doble Cola de Prioridad

- **Cita**: Wu, P. et al. (2025). *Water*, 17(4), 586.
- **WOS ID**: WOS:001486070400001
- **Citaciones**: 0

**Resumen**: Doble cola de prioridad considerando elevación y distancia a outlets. Corrección secundaria para celdas planas usando caminos zigzag. 19.2% más rápido que LCP.

---

#### [H12] Zhao et al. 2023 — Flow Direction Basado en Objetos Hidrológicos

- **Cita**: Zhao, F. et al. (2023). *Water Resources Research*, 59(9).
- **WOS ID**: WOS:001026173300001
- **Citaciones**: 10

**Resumen**: Flow direction que trata lagos y arroyos como objetos hidrológicos. Probado en la meseta tibetana. Compara contra D8, D-infinity y priority-flood.

---

#### [H13] Zhang et al. 2020 — Delimitación Dinámica Sin Filling

- **Cita**: Zhang, H. et al. (2020). *Water*, 12(4), 996.
- **WOS ID**: WOS:000519846500177
- **Citaciones**: 8

**Resumen**: Desafía el paradigma fill-then-route: usa DEM sin rellenar con sinks como "celdas especiales" que desbordan cuando el nivel de agua supera la salida. Mejor correlación con puntos de inundación que métodos D8 estándar.

---

#### [H14] Chen & Zou 2025 — Corrección Adaptativa de DEM por Spill-Lines

- **Cita**: Chen, J. et al. (2025). *Journal of Hydrology*.
- **WOS ID**: WOS:001576204000001
- **Citaciones**: 0

**Resumen**: Algoritmo adaptativo de spill-line que extrae spill points usando D8, análisis de acumulación de agua, y trazado de líneas de flujo. Remueve 94.3–98.8% de depresiones con alteración mínima del terreno (2.1–11‰).

---

#### Depresiones anidadas y watershed

#### [H15] Wu et al. 2019 — Depresiones Anidadas con Level-Set (~150x más rápido)

- **Cita**: Wu, Q. et al. (2019). *Journal of the American Water Resources Association*, 55(4), 911–927.
- **WOS ID**: WOS:000464285900007
- **Citaciones**: 55

**Resumen**: Método level-set basado en teoría de grafos para delinear jerarquías de depresiones anidadas. ~150x más rápido que métodos vectoriales de contour tree. Relevante para simulación fill-merge-spill.

---

#### [H16] Zhou et al. 2026 — Watershed Delineation O(N) Paralelo

- **Cita**: Zhou, G. et al. (2026). *Environmental Modelling & Software*.
- **WOS ID**: WOS:001604555800001
- **Citaciones**: 0

**Resumen**: Algoritmo O(N) de flow path traversal para watershed delineation desde grids de flow direction. Versión paralela (OpenMP) es la más rápida entre todos los evaluados.

---

#### [H17] Wang et al. 2011 — Flow Accumulation O(n)

- **Cita**: Wang, Y. et al. (2011). *ICDIP 2011*.
- **WOS ID**: WOS:000292648900080
- **Citaciones**: 0

**Resumen**: Algoritmo O(n) para matrices de flow accumulation (vs. O(n²) o O(n log n) estándar). Completa en segundos vs. minutos para DEM 1000×1000. Usa D8 con Planchon como preprocesamiento.

---

### 3.4 Referencias de relevancia media (v1)

| # | WOS ID | Autor | Año | Título abreviado | Cit. | Nota |
|---|---|---|---|---|---|---|
| 1 | WOS:001111992400001 | Yan, G. | 2024 | Priority-flood para slope units (landslides) | 17 | Aplicación de PFD para unidades de pendiente |
| 2 | WOS:000527307400002 | Liu, K. | 2020 | Lake-oriented watershed delineation (Tibet) | 35 | Workflow D8 + watershed para cuencas endorreicas |

---

### 3.5 Hallazgos de hydrology_v2.bib (275 entries)

El 69% de las 275 entradas (190 papers) son estudios de susceptibilidad (inundaciones, deslizamientos, erosión) que usan índices topográficos como features para ML — ruido para fines algorítmicos. Los 14 papers de alta relevancia se concentran en **comparación de algoritmos de flow direction** y **metodología de cálculo de TWI**.

#### Papers críticos sobre TWI

#### [H18] Kopecky, Macek & Wild 2021 — Guías de Cálculo de TWI

- **Cita**: Kopecky, M. et al. (2021). *Science of the Total Environment*, 757.
- **WOS ID**: WOS:000604432900063
- **Citaciones**: **234**

**Resumen**: Evalúa 26 variantes de TWI contra humedad del suelo medida. FD8 con dispersión de flujo ~1.0 supera a D8 por >2x en explicar variación de humedad. **Recomendación: FD8 con gradiente de slope local como default para TWI.**

**Relevancia para SurtGIS**: **Crítica**. Directamente provee lineamientos empíricos para la implementación de TWI. Valida que algoritmos MFD son superiores a D8 para TWI.

---

#### [H19] Kopecky & Cížková 2010 — ¿Importa el Algoritmo para TWI?

- **Cita**: Kopecky, M. & Cížková, Š. (2010). *Applied Vegetation Science*, 13(4), 450–459.
- **WOS ID**: WOS:000281556100006
- **Citaciones**: 139

**Resumen**: Compara 11 variantes de TWI con 11 algoritmos de flow routing en 521 plots. Los algoritmos de flujo múltiple (Quinn et al., Freeman) duplican el rendimiento vs. algoritmos de flujo único. **Conclusión: la elección del algoritmo tiene un efecto considerable en TWI.**

---

#### [H20] Qin et al. 2011 — TWI con Máximo Gradiente Descendente

- **Cita**: Qin, C.-Z. et al. (2011). *Precision Agriculture*, 12(1), 32–43.
- **WOS ID**: WOS:000286663800004
- **Citaciones**: 128

**Resumen**: Propone MFD adaptativo que usa el máximo gradiente descendente como beta (en lugar del slope local). Produce menores errores de TWI en terrenos artificiales a resoluciones 1–30m.

**Relevancia para SurtGIS**: **Crítica**. Innovación algorítmica directamente implementable. Mejora la calidad del TWI sin cambiar la estructura fundamental.

---

#### [H21] Radula et al. 2018 — MFD-md es Mejor para Humedad del Suelo

- **Cita**: Radula, M.W. et al. (2018). *Ecological Indicators*, 85, 172–179.
- **WOS ID**: WOS:000430634500018
- **Citaciones**: 137

**Resumen**: Compara 10 algoritmos de flow routing para TWI contra humedad medida. **MFD-md produce la relación TWI-humedad más fuerte.** TWI con cualquier algoritmo supera a valores de indicador de Ellenberg.

---

#### [H22] Pei et al. 2010 — MFD-TWI Superior para Mapeo de SOM

- **Cita**: Pei, T. et al. (2010). *Ecological Indicators*, 10(3), 610–619.
- **WOS ID**: WOS:000275220900006
- **Citaciones**: 135

**Resumen**: TWI basado en MFD correlaciona mejor con materia orgánica del suelo que TWI basado en SFD. Cokriging con MFD-TWI produce los mejores resultados de mapeo.

---

#### Papers sobre comparación de flow direction

#### [H23] Rampi et al. 2014 — 7 Algoritmos de Flow Direction para Humedales

- **Cita**: Rampi, L.P. et al. (2014). *Wetlands*, 34(3), 513–525.
- **WOS ID**: WOS:000336288800010
- **Citaciones**: 25

**Resumen**: Evalúa D8, Rho8, DEMON, D-infinity, MD-infinity, Mass Flux y FD8 para mapeo de humedales con LiDAR-CTI. **MFD (81–92% precisión) supera a SFD.** Recomienda MFD para la mayoría de casos.

---

#### [H24] Xu et al. 2022 — Comparación de 6 Algoritmos de Flow Routing

- **Cita**: Xu, Z. et al. (2022). *Journal of Hydrometeorology*, 23(12), 1913–1928.
- **WOS ID**: WOS:000905795600004
- **Citaciones**: 2

**Resumen**: Compara D8, D-infinity, MD-infinity, MD8, MFD-md y RMD-infinity. Diferencias a nivel de watershed son bajas post-calibración, pero diferencias significativas a nivel de celda en humedad del suelo. MD produce resultados más autocorrelacionados espacialmente.

---

#### [H25] Li et al. 2024 — TFGA: MFD Faceta-a-Faceta (Estado del Arte)

- **Cita**: Li, Z. et al. (2024). *Geomorphology*, 466.
- **WOS ID**: WOS:001332008800001
- **Citaciones**: 0 (reciente)

**Resumen**: Divide el pixel central en 8 sub-facetas, deriva relaciones matemáticas estrictas entre dirección de slope y proporciones de flujo. Mejora la precisión en ~1 orden de magnitud sobre algoritmos existentes.

---

#### Tabla de relevancia media (v2, selección)

| # | WOS ID | Autor | Año | Título abreviado | Cit. | Nota |
|---|---|---|---|---|---|---|
| 1 | WOS:000450551600042 | Choubin | 2019 | Ensemble flood susceptibility | 586 | Usa TWI, slope como factores |
| 2 | WOS:000414922600048 | Khosravi | 2018 | ML flash flood susceptibility | 374 | Usa slope, TWI |
| 3 | WOS:000353516400006 | Tehrany | 2015 | SVM flood susceptibility | 346 | Usa TWI, slope, altitude |
| 4 | WOS:000470317700015 | Moayedi | 2019 | PSO-ANN landslide susceptibility | 287 | Usa slope, aspect, curvature |
| 5 | WOS:000406732200028 | Chen | 2017 | Hybrid ML landslide susceptibility | 281 | Usa TWI, SPI, slope |
| 6 | WOS:000376309900023 | Rahmati | 2017 | ML gully erosion | 207 | Usa SPI, TWI, MRVBF |
| 7 | WOS:000314728600011 | Lang | 2013 | TWI metrics for wetland mapping | 110 | FD8/D-inf > D8 para TWI |
| 8 | WOS:000168253300006 | Chen | 2001 | Topographic water/energy balance | 101 | TWI distribución estadística |

#### Distribución temática de las 275 entradas

| Dominio | % |
|---|---|
| Susceptibilidad a inundaciones / deslizamientos / erosión (ML) | ~60% |
| TWI y flow direction (algorítmico) | ~8% |
| Hidrología aplicada (cuencas, humedales) | ~12% |
| Ecología / suelos con TWI como variable | ~10% |
| Otros (batimetría, nieve, incendios) | ~10% |

---

### 3.6 Síntesis y hallazgos para SurtGIS

#### Estado del arte en depression filling

La familia Priority-Flood (Barnes 2014) es el estándar actual. Las optimizaciones posteriores (Zhou 2016: region-growing; Wei 2019: reducción 70% cola; Xiong 2019: micro-relieve) ofrecen speed-ups acumulativos de 30–50% sobre la versión base.

| Algoritmo | Complejidad | Estado |
|---|---|---|
| Jenson-Domingue (1988) | Exponencial | Obsoleto para DEMs grandes |
| Planchon-Darboux (2002) | O(n) | Bueno, pero superado por PF |
| Wang-Liu (2006) | O(n log n) | Base del Priority-Flood |
| **Priority-Flood (Barnes 2014)** | **O(n log n) / O(n)** | **Estado del arte** |
| PF + region-growing (Zhou 2016) | O(n log n) mejorado | +44.6% vs PF base |
| PF paralelo (Barnes 2016) | O(n log n) paralelo | Escala a 10¹² celdas |

#### Papers fundacionales identificados

| Paper | Citaciones | Rol | En bibliografía |
|---|---|---|---|
| Barnes et al. 2014 | 181 | Priority-Flood original | **Presente** (v1) |
| Barnes 2016 | 39 | PF paralelo | **Presente** (v1) |
| Zhou et al. 2016 | 42 | PF eficiente | **Presente** (v1) |
| Wu et al. 2019 | 55 | Depresiones anidadas | **Presente** (v1) |
| Planchon-Darboux 2002 | — | PD original | Citado pero no como entry |
| Wang-Liu 2006 | — | PF precursor | Citado pero no como entry |
| O'Callaghan-Mark 1984 | — | D8 original | Citado pero no como entry |
| Tarboton 1997 | — | D-infinity | Citado pero no como entry |
| Beven-Kirkby 1979 | — | TWI original | **Ausente** |
| Moore et al. 1991 | — | SPI/STI | **Ausente** |

#### Hallazgo crítico de v2: MFD supera a D8 para TWI

El consenso de la literatura es unánime (Kopecky 2021, Kopecky 2010, Pei 2010, Radula 2018, Lang 2013, Rampi 2014):

> **Los algoritmos de flujo múltiple (MFD) superan consistentemente a D8 para el cálculo de TWI, duplicando la capacidad explicativa de humedad del suelo.**

**Recomendación para SurtGIS**: Implementar FD8/Quinn MFD como default para TWI. D8 debería mantenerse como opción pero no ser el default.

| Algoritmo | Prioridad | Justificación |
|---|---|---|
| D8 (ya implementado) | Baseline | Simple, rápido, suficiente para flow direction |
| **FD8 / Quinn MFD** | **1ª prioridad** | Empíricamente el mejor para TWI (Kopecky 2021, 234 cit.) |
| D-infinity (Tarboton) | 2ª prioridad | Ampliamente citado, ángulos continuos |
| MFD-md | 3ª prioridad | Mejor para humedad del suelo (Radula 2018) |
| MFD adaptativo (Qin) | Futuro | Mejora TWI con máximo gradiente descendente |

#### Vacíos identificados

1. **SPI/STI**: Ampliamente usados como covariables pero sin papers específicos sobre algoritmos de cálculo
2. **Convergence index**: Solo 4 papers lo mencionan, sin referencia algorítmica
3. **El paradigma no-fill** (Zhang 2020) desafía el enfoque actual de SurtGIS y merece evaluación
4. **Papers fundacionales ausentes como entries principales**:
   - Beven & Kirkby 1979 (TWI original) — citado por ~30 entries pero ausente como entry
   - Quinn et al. 1991 (MFD) — citado por ~12 entries
   - Moore et al. 1991 (SPI/STI) — citado por ~15 entries

## 4. Query: Remote Sensing / Spectral Indices

**Fecha**: 2026-01-31
**Archivo**: `biblio/remote_sensing_v1.bib` (66 entries)
**Análisis detallado**: `docs/remote_sensing_analysis.md` + `docs/remote_sensing_biblio.json`

### 4.1 Resumen

| Métrica | Valor |
|---------|-------|
| **Entries totales** | 66 (59 parseadas) |
| **Rango temporal** | 2006–2026 |
| **Alta relevancia (≥3)** | 31 papers (53%) |
| **Fundacionales (≥20 citas + relevancia ≥3)** | 13 papers |
| **Tema dominante** | Índices espectrales de vegetación (93%) |

### 4.2 Distribución por Tema

| Tema | Cantidad | Notas |
|------|----------|-------|
| **Índices/NDVI** | 55 | Tema dominante — índices de vegetación |
| **Análisis espectral** | 2 | Hiperespectral/reflectancia |
| **Detección de cambios** | 2 | Análisis temporal |
| **SAR** | 0 | No representado |
| **Corrección atmosférica** | 0 | No representado |

### 4.3 Papers Fundacionales (Top 10 por Impacto)

| # | Autor (Año) | Citas | Rel. | Contribución clave |
|---|-------------|-------|------|-------------------|
| 1 | **Bian et al. (2019)** | 120 | 5 | UAV térmico, CWSI, detección de bordes Canny |
| 2 | **Li et al. (2014)** | 184 | 3 | Optimización de índices hiperespectrales, PLSR |
| 3 | **Narmilan et al. (2022)** | 85 | 4 | ML + índices espectrales para clorofila |
| 4 | **Jiang et al. (2023)** | 37 | 5 | Fusión UAV + Sentinel-2, agregación multi-resolución |
| 5 | **Zhu et al. (2007)** | 41 | 4 | Relaciones espectrales nitrógeno-arroz |
| 6 | **Rozenstein et al. (2019)** | 28 | 5 | Validación Sentinel-2, framework de precisión |
| 7 | **Edwin Pizarro et al. (2022)** | 23 | 5 | Clasificación ML cobertura de suelo |
| 8 | **Wang et al. (2019)** | 19 | 5 | **Índice espectral de 3 bandas — guía para n-band builder** |
| 9 | **Srivastava et al. (2021)** | 26 | 3 | ANN para clorofila hiperespectral |
| 10 | **Das et al. (2023)** | 24 | 3 | Deep learning detección enfermedades |

### 4.4 Índices Espectrales Identificados — Faltantes en SurtGIS

| Índice | Fórmula | Prioridad | Papers |
|--------|---------|-----------|--------|
| **NDRE** | `(NIR - RedEdge) / (NIR + RedEdge)` | **Crítica** | Múltiples |
| **RECI** | `(NIR / RedEdge) - 1` | **Alta** | 5+ |
| **GNDVI** | `(NIR - Green) / (NIR + Green)` | **Alta** | Múltiples |
| **NGRDI** | `(Green - Red) / (Green + Red)` | **Alta** | Múltiples |
| **MTVI** | Fórmula triangular compleja | Media | Pocos |
| **CWSI** | Basado en banda térmica | Media | 1 (Bian 2019) |
| **MSI** | `SWIR / NIR` | Media | Pocos |

**Nota**: SurtGIS ya tiene NDVI, NDWI, MNDWI, NBR, SAVI, EVI, BSI + `normalized_difference` genérico.

### 4.5 Hallazgo Clave: Framework Genérico de n Bandas

Wang et al. (2019) demuestra que índices de 3 bandas superan a los de 2 bandas. **Recomendación**: implementar un builder genérico de índices espectrales:

```rust
pub fn calculate_custom_index(
    bands: &[(&str, &Raster<f32>)],  // [("NIR", nir), ("Red", red), ("Blue", blue)]
    formula: &str                      // "(NIR - Red) / (NIR + Red + Blue)"
) -> Result<Raster<f32>>
```

Esto desbloquearía cientos de índices personalizados sin necesidad de implementar cada uno individualmente. **Ningún competidor** (WBT/SAGA/GRASS) ofrece esto como función genérica.

### 4.6 Gaps No Cubiertos por Esta Bibliografía

- ❌ SAR/Radar — no representado
- ❌ Corrección atmosférica — no representado
- ❌ Segmentación de imagen — cobertura mínima
- ❌ Teledetección urbana — no representado
- ⚠️ Detección de cambios temporal — solo 2 papers
- ⚠️ Fusión de datos — cobertura limitada

## 5. Query: Interpolation Methods

**Fecha**: 2026-01-31
**Archivo**: `biblio/interpolation.bib` (200 entries, 188 parseadas)
**Análisis detallado**: `docs/interpolation_analysis.md` + `docs/interpolation_biblio.json` + `docs/INTERPOLATION_QUICK_REFERENCE.md`

### 5.1 Resumen

| Métrica | Valor |
|---------|-------|
| **Entries totales** | 200 (188 parseadas) |
| **Alta relevancia (≥3)** | 188 papers |
| **Fundacionales (≥20 citas + rel. ≥4)** | 27 papers |
| **Tema dominante** | Kriging/Geostatísticas (79%) |

### 5.2 Distribución por Tema

| Tema | Papers | % | Observación |
|------|--------|---|-------------|
| **Kriging/Geostatísticas** | 149 | 79% | Dominante — gold standard de la literatura |
| **Machine Learning** | 26 | 14% | Creciente — RF y GB superan kriging en terreno complejo |
| **IDW** | 7 | 4% | Pocas publicaciones propias, pero IDW aparece como baseline en comparaciones |
| **TIN/Delaunay** | 5 | 3% | Pocas publicaciones recientes |

### 5.3 Papers Fundacionales (Top 10 por Citas)

| # | Autor (Año) | Citas | Tema | Contribución |
|---|-------------|-------|------|--------------|
| 1 | **Workneh et al. (2024)** | 71 | Kriging vs IDW | Benchmark en 6 regímenes de lluvia, Etiopía |
| 2 | **Langella et al. (2010)** | 57 | ML | ANN de alta resolución espacio-temporal |
| 3 | **Xu et al. (2023)** | 56 | Kriging | Comparación de 3 métodos en montañas Hengduan |
| 4 | **Stroessenreuther et al. (2020)** | 52 | Kriging | Altimetría de superficie glaciar (interpolación + precisión) |
| 5 | **Chen et al. (2011)** | 43 | Spline | **Coons patch** para generación de DEM |
| 6 | **Bhunia et al. (2018)** | 32 | Kriging | Mapeo de metales pesados en suelo |
| 7 | **Njeban (2018)** | 31 | IDW vs Kriging | Comparación de métodos, datos topográficos |
| 8 | **Hofstra et al. (2008)** | 30 | Kriging | Interpolación de estaciones meteorológicas |
| 9 | **Li et al. (2022)** | 28 | ML | Neural Network para DEM de alta resolución |
| 10 | **Kravchenko & Bullock (1999)** | 27 | Kriging | Comparación IDW/OK/UK para datos de suelo |

### 5.4 Estado Actual de SurtGIS vs. Necesidades

| Método | SurtGIS | Literatura | Gap |
|--------|---------|-----------|-----|
| IDW básico | ✅ | Baseline | — |
| Nearest Neighbor | ✅ | Baseline | — |
| TIN (Bowyer-Watson) | ✅ | Baseline | — |
| **Ordinary Kriging (OK)** | ❌ | 79% de papers | **Crítico** |
| **Regression Kriging (RK)** | ❌ | 15-30% mejor que IDW en montaña | **Crítico** |
| **Universal Kriging (UK)** | ❌ | 5-15% mejor que OK con tendencias | **Alto** |
| **Thin Plate Spline (TPS)** | ❌ | Florinsky Ch.3 también lo recomienda | **Alto** |
| **Natural Neighbor** | ❌ | Complementa TIN existente | **Medio** |
| **Random Forest interpolation** | ❌ | 10-25% mejor que kriging (datos densos) | **Medio** |
| **RBF / Regularized Spline** | ❌ | Estándar en ArcGIS/QGIS/GRASS | Medio |
| IDW adaptativo/anisotrópico | ❌ | Mejora significativa sobre IDW fijo | Medio |
| Cokriging | ❌ | Requiere variables correlacionadas | Bajo |

### 5.5 Mejoras de Precisión Documentadas (Benchmarks)

| Comparación | Mejora RMSE | Contexto |
|------------|-------------|----------|
| **Regression Kriging vs IDW** | **15-30%** | Terreno montañoso |
| **Ordinary Kriging vs Nearest Neighbor** | 10-20% | Datos dispersos |
| **Universal Kriging vs Ordinary Kriging** | 5-15% | Tendencias fuertes |
| **Random Forest vs Kriging** | 10-25% | Terreno complejo + datos densos |

### 5.6 Hallazgo: Variograma como Prerequisito

Todas las variantes de kriging requieren un **modelo de variograma** (semivarianza vs. distancia). Componentes:
- **Nugget**: Varianza a distancia 0 (ruido de medición)
- **Sill**: Varianza máxima (meseta)
- **Range**: Distancia a la cual se alcanza el sill
- **Modelos**: Esférico, exponencial, gaussiano, Matérn

Implementar un módulo de variograma es prerequisito para cualquier kriging. Es un componente que SAGA, GRASS y ArcGIS ofrecen.

### 5.7 Mejoras Recomendadas para IDW Actual

| Mejora | Descripción | Impacto |
|--------|-------------|---------|
| **Potencia adaptativa** | Variar exponente según densidad local de puntos | Alto |
| **Anisotropía** | Diferentes radios de búsqueda en diferentes direcciones | Alto — terreno |
| **Kernel weighting** | Reemplazar $d^{-p}$ con Gaussiano u otros kernels | Medio |
| **Barreras** | Respetar barreras topográficas (crestas, valles) | Medio |
| **Parámetros típicos** | p=2.0 (default), rango 1.0-3.0; 10-20 vecinos cercanos | Documentar |

### 5.8 Guía de Selección de Método (de la Literatura)

| Escenario | Método Recomendado |
|-----------|-------------------|
| **Generación de DEM** | Regression Kriging > Universal Kriging > Random Forest > TPS |
| **Datos dispersos (<100 pts)** | Ordinary Kriging > Natural Neighbor > Universal Kriging |
| **Datos densos (>1000 pts)** | Random Forest/GB > IDW optimizado > Regression Kriging |
| **Superficies suaves** | Thin Plate Spline > Universal Kriging > Regression |
| **Múltiples variables auxiliares** | Regression Kriging > Random Forest > Cokriging |

### 5.9 Convergencia con Florinsky (2025)

El Cap. 3 de Florinsky identifica **Thin Plate Spline** como faltante crítico para SurtGIS. Esta bibliografía confirma:
- TPS es estándar en software GIS (ArcGIS, QGIS, GRASS)
- Ideal para superficies suaves (batimetría, clima)
- Florinsky prefiere Chebyshev (Cap. 7) pero reconoce TPS como alternativa práctica

## 5b. Query: Solar Radiation / Viewshed

**Fecha**: 2026-01-31
**Archivos**: `biblio/solar_v1.bib` (80 entries — **mislabeled**: 100% viewshed), `biblio/solar_v2.bib` (55 entries — solar radiation real)
**Análisis detallados**: `docs/viewshed_analysis.md` + `docs/viewshed_biblio.json`, `docs/solar_analysis.md` + `docs/solar_biblio.json`

> **Hallazgo importante**: `solar_v1.bib` NO contiene papers de radiación solar. Es 100% análisis de viewshed/visibilidad. Se renombra conceptualmente como bibliografía de viewshed.

### 5b.1 Resumen General

| Archivo | Entries | Tema real | Alta relevancia | Fundacionales (≥30 citas) |
|---------|---------|-----------|-----------------|---------------------------|
| `solar_v1.bib` | 80 | **Viewshed / Visibilidad** (100%) | 80 | 9 |
| `solar_v2.bib` | 55 | Radiación solar DEM-based | 55 | 6 |
| **Total** | **135** | — | **135** | **15** |

### 5b.2 Parte A: Viewshed y Visibilidad (solar_v1.bib)

#### Distribución por Tema

| Categoría | Papers | % | Descripción |
|-----------|--------|---|-------------|
| **GPU / Paralelo / Distribuido** | 30 | 37.5% | CUDA, Spark, computación distribuida |
| **Aplicaciones de dominio** | 28 | 35.0% | Paisaje, ecología, arqueología, incendios |
| **3D / Urbano / LiDAR** | 26 | 32.5% | Viewshed con edificios y nubes de puntos |
| **Algoritmos de viewshed** | 21 | 26.2% | R2, R3, XDraw, familia PDERL |
| **Ubicación de observadores** | 21 | 26.2% | Optimización MCLP, torres de vigilancia |
| **Metaheurísticas** | 10 | 12.5% | DE, GA, PSO para optimización |
| **Fuzzy / Probabilístico** | 5 | 6.2% | Incertidumbre en DEM |
| **Estructuras de datos** | 4 | 5.0% | Quadtree, K-D trees |
| **Intervisibilidad** | 3 | 3.8% | Comunicaciones, línea de vista |

> Nota: papers pueden pertenecer a múltiples categorías.

#### Top 10 Papers por Citas

| # | Autor (Año) | Citas | Contribución |
|---|-------------|-------|--------------|
| 1 | **Fisher (1993)** | 118 | **Fundacional**: incertidumbre algorítmica y de implementación en viewshed |
| 2 | **Bao et al. (2015)** | 92 | Optimización de torres de vigilancia (MCLP) |
| 3 | **Zhao et al. (2013)** | 78 | Primer viewshed GPU paralelo con CUDA |
| 4 | **Civicioglu et al. (2021)** | 68 | Bezier Search Differential Evolution |
| 5 | **Zhang et al. (2020)** | 50 | Despliegue multi-factor de torres de incendios |
| 6 | **Kloucek et al. (2015)** | 36 | Impacto de precisión del DEM en viewshed |
| 7 | **Yaagoubi et al. (2015)** | 35 | HybVOR — ubicación de cámaras con Voronoi |
| 8 | **Fang Chao et al. (2011)** | 32 | Viewshed paralelo en GPU |
| 9 | **Wang Feng et al. (2015)** | 31 | Viewshed paralelo en Digital Earth 3D |
| 10 | **Cimburova et al. (2022)** | 30 | Exposición visual a vegetación urbana |

#### Taxonomía de Algoritmos de Viewshed

| Algoritmo | Tipo | Precisión | Velocidad | Papers |
|-----------|------|-----------|-----------|--------|
| **R2 (Franklin)** | Aproximado | Baja | Rápido | 3 |
| **R3** | Exacto | Perfecta | Lento | 4 |
| **XDraw** | Aproximado | Buena | Rápido | 12 |
| **Reference Plane** | Aproximado | Moderada | Rápido | 4 |
| **PDERL** | Exacto | Perfecta | Medio | 4 |
| **XPDERL** | Aproximado | Muy buena | Rápido | 2 |
| **HiPDERL** | Exacto | Perfecta | Medio | 1 |
| **Sweep-line** | Exacto | Alta | Medio | 2 |
| **Q-View** | Query-based | Alta | Muy rápido | 1 |

#### Gaps de SurtGIS vs. Literatura de Viewshed

| Gap | Prioridad | Papers | Recomendación |
|-----|-----------|--------|---------------|
| **No tiene XDraw** | ALTA | 12 | Speedup 2-5× sobre ray tracing actual |
| **No tiene PDERL/XPDERL** | ALTA | 6 | Precisión R3 a velocidad de XDraw |
| **No tiene R3 exacto** | MEDIA | 4 | Como referencia/benchmark de precisión |
| **No viewshed probabilístico** | MEDIA | 6 | Propagación de incertidumbre del DEM (Fisher 1993) |
| **No optimización de observadores** | MEDIA | 21 | MCLP/greedy para maximizar cobertura |
| **No viewshed cumulativo ponderado** | MEDIA | 10+ | Pesos por distancia, ángulo o importancia |
| **No viewshed 3D/urbano** | BAJA | 28 | Con modelos de edificios/LiDAR |
| **No Fresnel zone** | BAJA | 4 | Para aplicaciones de telecomunicaciones |

**Implementación actual** (`viewshed.rs`): Ray tracing Bresenham desde observador a perímetro, con max_angle para LOS. Paralelismo rayon. Soporta single/multiple observer.

### 5b.3 Parte B: Radiación Solar (solar_v2.bib)

#### Distribución por Tema

| Subtema | Papers | % |
|---------|--------|---|
| **DEM-based solar mapping** | 55 | 100% |
| **Métodos computacionales** | 43 | 78.2% |
| **Benchmarks y validación** | 39 | 70.9% |
| **Integración temporal** | 34 | 61.8% |
| **Remote sensing** | 32 | 58.2% |
| **Clima y ecología** | 29 | 52.7% |
| **Nieve y glaciología** | 17 | 30.9% |
| **Downscaling** | 13 | 23.6% |
| **Radiación reflejada** | 11 | 20.0% |
| **Cómputo de horizonte** | 9 | 16.4% |
| **Sky View Factor** | 7 | 12.7% |
| **Herramientas GIS** | 7 | 12.7% |
| **Fotovoltaica** | 6 | 10.9% |

#### Top 10 Papers por Citas

| # | Autor (Año) | Citas | Contribución |
|---|-------------|-------|--------------|
| 1 | **Corripio (2003)** | 161 | **Fundacional**: álgebra vectorial para geometría terreno-solar |
| 2 | **Gorsevski et al. (2016)** | 132 | Radiación solar como feature ML para deslizamientos |
| 3 | **Ruiz-Arias et al. (2009)** | 93 | **Benchmark**: r.sun vs Solar Analyst vs SRAD vs Solei-32 |
| 4 | **Lopez-Serrano et al. (2016)** | 76 | Corrección atmosférica con geometría solar |
| 5 | **Long et al. (2010)** | 71 | Balance completo: directa + difusa + reflejada |
| 6 | **Wang et al. (2018)** | 60 | Shortwave operacional sobre terreno rugoso |
| 7 | **Zhang, Li & Bai (2015)** | 54 | MODIS + DEM para productos atmosféricos |
| 8 | **Marsh et al. (2012)** | 49 | Sombras de montaña para fusión de nieve en TINs |
| 9 | **Ahmadi et al. (2020)** | 45 | ML forestal con input solar |
| 10 | **Bourque & Pomeroy (2001)** | 45 | Impacto de radiación solar en ecología fluvial |

#### Estado de Algoritmos: SurtGIS vs. Literatura

**Posición solar y geometría**:

| Algoritmo | SurtGIS | Estado |
|-----------|---------|--------|
| Declinación Spencer (1971) | ✅ | Correcto |
| Ángulo horario / amanecer-ocaso | ✅ | Correcto |
| Altitud y azimut solar | ✅ | Correcto |
| Ángulo de incidencia en pendiente | ✅ | Correcto |
| Masa de aire (Kasten-Young) | ✅ | Correcto |
| Matrices de rotación vectorial (Corripio) | ❌ | Más elegante, sin edge-cases trigonométricos |

**Radiación directa (beam)**:

| Componente | SurtGIS | Gap |
|------------|---------|-----|
| Transmitancia atmosférica | Coef. fijo (0.7) | **Simplificado** — falta Linke turbidity |
| Factor de turbidez Linke | ❌ | **Crítico** — 10-30% error en condiciones hazy |
| Modelo ESRA clear-sky | ❌ | Opcional |

**Radiación difusa**:

| Modelo | SurtGIS | Gap |
|--------|---------|-----|
| Isotrópico (Liu-Jordan) | ✅ (simplificado) | Básico — solo `(1+cos(slope))/2` |
| **Klucher anisotrópico** | ❌ | **Crítico** — corrige circumsolar y horizonte |
| **Hay anisotrópico** | ❌ | Medio — ponderación por normal directa |
| **Perez anisotrópico** | ❌ | **Alto** — más preciso, 8 bins de cielo |
| **SVF-corrected diffuse** | ❌ | **Alto** — SurtGIS ya tiene SVF pero no lo integra |

**Radiación reflejada**:

| Componente | SurtGIS | Gap |
|------------|---------|-----|
| Reflexión isotrópica del terreno | ❌ | **Crítico** — `albedo × GHI × (1-cos(slope))/2` |
| Reflexión anisotrópica (facetas) | ❌ | Bajo — factor de vista entre facetas adyacentes |

**Shadow casting / Horizonte**:

| Algoritmo | SurtGIS | Gap |
|-----------|---------|-----|
| Detección de sombras (Corripio) | ❌ | **Crítico** — 10-40% sobreestimación en montaña |
| **HORAYZON** ray-tracing (Steger 2022) | ❌ | **Crítico** — 10-100× más rápido que fuerza bruta |
| HPFTOA sub-grid (Li 2024) | ❌ | Opcional |
| Sombras en TIN (Marsh 2012) | ❌ | Opcional |

**Integración temporal**:

| Feature | SurtGIS | Gap |
|---------|---------|-----|
| Integración diaria | ✅ | Correcto |
| Acumulación mensual/anual | ❌ | **Alto** — limita a análisis de un solo día |
| Corrección por nubes | ❌ | Bajo — requiere input MODIS |

### 5b.4 Impacto Cuantificado de los Gaps

| Gap | Impacto en Error | Referencia |
|-----|-------------------|------------|
| **No shadow casting** | 10-40% sobreestimación en montaña | Marsh (2012) |
| **Solo difusa isotrópica** | 5-15% error en componente difusa | Long (2010), Zhang Y. (2017) |
| **No radiación reflejada** | 5-20% subestimación en nieve, 2-5% general | Long (2010), Wang (2018) |
| **No Linke turbidity** | 10-30% error en condiciones hazy | Ruiz-Arias (2009) |
| **Latitud única para todo el raster** | Significativo para áreas >1° de extensión | Múltiples |

### 5b.5 Hallazgos Clave

1. **HORAYZON (Steger 2022)** es el estado del arte para cómputo de horizonte: ray-tracing con simplificación del terreno, 10-100× más rápido que fuerza bruta. SurtGIS debería implementar esto en lugar de un approach naïve.

2. **Corripio (2003)** — El paper fundacional (161 citas) propone álgebra vectorial con matrices de rotación para TODA la geometría solar. Elimina edge-cases trigonométricos. Método preferido por la comunidad.

3. **Ruiz-Arias (2009)** — Benchmark definitivo. Compara r.sun, Solar Analyst, SRAD y Solei-32. RMSE objetivo: <2.5 MJ/m²/día para radiación global (cielo claro), MBE <0.5 MJ/m²/día.

4. **SVF ya existe en SurtGIS** (`sky_view_factor.rs`) pero NO está conectado al módulo solar. La conexión es trivial (`diffuse_terrain = diffuse_horizontal × SVF`) pero tiene alto impacto en valles profundos.

5. **Viewshed y solar comparten infraestructura**: horizon angles sirven tanto para shadow casting como para SVF. Un módulo compartido de horizon computation beneficiaría a ambos.

6. **SurtGIS sería primero en su clase** como librería Rust-nativa de radiación solar DEM-based. Ningún paper en la bibliografía usa Rust o WASM para este tipo de cómputo.

### 5b.6 Roadmap de Mejoras por Fases

#### Fase 1 — Impacto inmediato
| Mejora | Módulo | Impacto |
|--------|--------|---------|
| **Horizon angles** (36 direcciones, Bresenham) | `solar_radiation.rs` + `viewshed.rs` | Elimina 10-40% sobreestimación |
| **Radiación reflejada isotrópica** | `solar_radiation.rs` | +20 líneas, albedo configurable |
| **Latitud per-cell** del GeoTransform | `solar_radiation.rs` | ~10 líneas |
| **Integrar SVF existente** | `solar_radiation.rs` | Modular difusa con SVF |

#### Fase 2 — Precisión del modelo
| Mejora | Módulo | Impacto |
|--------|--------|---------|
| **Linke turbidity** (Kasten 1989) | `solar_radiation.rs` | Reemplaza coef. fijo 0.7 |
| **Klucher anisotrópico** | `solar_radiation.rs` | Upgrade de difusa isotrópica |
| **Acumulación mensual/anual** | `solar_radiation.rs` | Desbloquea casos de uso PV y ecología |

#### Fase 3 — Estado del arte
| Mejora | Módulo | Impacto |
|--------|--------|---------|
| **Perez diffuse model** | `solar_radiation.rs` | Más preciso, 8 clases de cielo |
| **HORAYZON ray-tracing** | `terrain::horizon` (nuevo) | 10-100× más rápido para DEMs grandes |
| **XDraw viewshed** | `viewshed.rs` | 2-5× speedup sobre Bresenham actual |
| **PDERL viewshed** | `viewshed.rs` | Precisión R3 a velocidad ~XDraw |

#### Fase 4 — Viewshed avanzado
| Mejora | Módulo | Impacto |
|--------|--------|---------|
| **Viewshed probabilístico** (Fisher 1993) | `viewshed.rs` | Propagación de incertidumbre del DEM |
| **Optimización de observadores** (MCLP) | `viewshed.rs` | Maximización de cobertura |
| **Viewshed cumulativo ponderado** | `viewshed.rs` | Pesos por distancia/ángulo |
| **Álgebra vectorial Corripio** | `solar_radiation.rs` | Rewrite elegante sin trigonometría edge-cases |

### 5b.7 Convergencia con Florinsky (2025) y Otras Secciones

- **Florinsky Cap. 2** (§7.3): Variables solares RL (hillshade), I (insolación), σ (horizonte), Cs (SVF) — todas en el sistema de 57 variables
- **Florinsky Cap. 8** (§7.7): Visualización con hillshade multidireccional usa shadow casting
- **Sección 2 (Terrain)**: Papers de SVF y openness comparten infraestructura con horizon computation
- **Sección 6.2.4**: Matriz competitiva muestra que GRASS (r.sun) es líder en solar; SurtGIS está "Par" con los demás

---

## 6. Análisis Competitivo y Roadmap de Nuevos Algoritmos

**Fecha**: 2026-01-31
**Fuentes**: Inventario SurtGIS (50+ algoritmos), WhiteboxTools (480+ tools, Open Core + WTE), SAGA GIS (~25 módulos morfometría), GRASS GIS (~30 módulos terreno/hidrología), bibliografía revisada (903 refs).

### 6.1 Inventario actual de SurtGIS

| Categoría | Cantidad | Algoritmos |
|---|---|---|
| Terreno | 19 | slope, aspect, hillshade, curvature (general/profile/plan), TPI, TRI, TWI, SPI, STI, landform, geomorphons, viewshed, SVF, openness (+/-), convergence, multiscale curvatures, smoothing, wind exposure, solar radiation, MRVBF/MRRTF |
| Hidrología | 4 | fill_sinks (Planchon-Darboux), flow_direction (D8), flow_accumulation (D8), watershed |
| Imágenes | 10 | NDVI, NDWI, MNDWI, SAVI, EVI, NBR, BSI, normalized_difference, band_math, reclassify |
| Interpolación | 3 | IDW, nearest neighbor, TIN (Bowyer-Watson) |
| Morfología | 7 | erode, dilate, opening, closing, gradient, top-hat, black-hat |
| Estadística | 3 | focal (9 stats), zonal, autocorrelation (Moran's I, Getis-Ord Gi*) |
| Vector | 6 | buffer, simplify (DP/VW), spatial ops, clip, measurements |
| **Total** | **~52** | |

### 6.2 Matriz comparativa: SurtGIS vs Competidores

#### 6.2.1 Terreno — Derivados básicos

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Notas |
|---|---|---|---|---|---|
| Slope | ✅ | ✅ | ✅ | ✅ | Ubicuo |
| Aspect | ✅ | ✅ | ✅ | ✅ | Ubicuo |
| Hillshade | ✅ | ✅ | ✅ | ✅ (r.relief) | Ubicuo |
| **Multidirectional hillshade** | ❌ | ✅ | ❌ | ❌ | **WBT único** |
| **Hypsometrically tinted hillshade** | ❌ | ✅ | ❌ | ❌ | **WBT único** |
| **Shadow animation** | ❌ | ✅ (WTE) | ❌ | ❌ | WTE de pago |
| **Time in daylight** | ❌ | ✅ | ❌ | ❌ | Diferenciador |

#### 6.2.2 Terreno — Curvaturas

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| Curvatura general/media | ✅ | ✅ (MeanCurvature) | ✅ | ✅ | Zevenbergen & Thorne 1987 |
| Curvatura profile | ✅ | ✅ | ✅ | ✅ | |
| Curvatura plan/tangential | ✅ | ✅ (TangentialCurvature) | ✅ | ✅ | |
| Multiscale curvatures (5×5) | ✅ | ✅ | ❌ | ❌ | Florinsky 2016 |
| **Gaussian curvature** | ❌ | ✅ | ❌ | ❌ | **Florinsky 2023** |
| **Maximal curvature** | ❌ | ✅ | ✅ | ❌ | **Florinsky 2023** |
| **Minimal curvature** | ❌ | ✅ | ✅ | ❌ | **Florinsky 2023** |
| **Accumulation curvature** | ❌ | ✅ (WTE) | ❌ | ❌ | **Florinsky 2023** |
| **Difference curvature** | ❌ | ✅ (WTE) | ❌ | ❌ | **Florinsky 2023** |
| **Ring curvature** | ❌ | ✅ | ❌ | ❌ | **Florinsky 2023** |
| **Horizontal excess curvature** | ❌ | ✅ (WTE) | ❌ | ❌ | **Florinsky 2023** |
| **Vertical excess curvature** | ❌ | ✅ | ❌ | ❌ | **Florinsky 2023** |
| **Unsphericity** | ❌ | ✅ | ❌ | ❌ | **Florinsky 2023** |
| **Curvedness** | ❌ | ✅ (WTE) | ❌ | ❌ | **Florinsky 2023** |
| **Shape index** | ❌ | ✅ | ❌ | ❌ | **Florinsky 2023** |
| **Rotor** | ❌ | ✅ | ❌ | ❌ | **Florinsky 2023** |
| **Generating function** | ❌ | ✅ (WTE) | ❌ | ❌ | **Florinsky 2023** |
| Upslope/Downslope curvature | ❌ | ❌ | ✅ | ❌ | SAGA único |

#### 6.2.3 Terreno — Posición topográfica y escalas

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| TPI | ✅ | ❌ (usa DEV) | ✅ | ✅ (add-on) | Weiss 2001 |
| TRI | ✅ | ✅ (RuggednessIndex) | ✅ | ✅ (add-on) | Riley 1999 |
| **DEV (Deviation from Mean Elev)** | ❌ | ✅ (DevFromMeanElev) | ❌ | ❌ | **De Reu 2013 (509 cit.)** — superior a TPI |
| **DiffFromMeanElev** | ❌ | ✅ | ❌ | ❌ | Newman 2018 |
| **ElevPercentile** | ❌ | ✅ | ❌ | ❌ | |
| **ElevRelativeToMinMax** | ❌ | ✅ | ❌ | ❌ | |
| **RelativeTopographicPosition** | ❌ | ✅ | ✅ | ❌ | |
| **MultiscaleTopographicPositionImage** | ❌ | ✅ | ❌ | ❌ | **Newman 2018 (43 cit.)** — integral images |
| **MultiscaleElevationPercentile** | ❌ | ✅ | ❌ | ❌ | |
| **MaxElevationDeviation** | ❌ | ✅ | ❌ | ❌ | |
| **MaxAnisotropyDev** | ❌ | ✅ | ❌ | ❌ | |
| MRVBF/MRRTF | ✅ | ❌ | ✅ | ✅ (add-on) | Gallant & Dowling 2003 |
| **MSV (Multiescale Valleyness)** | ❌ | ❌ | ❌ | ❌ | **Wang 2009** — ninguno lo tiene |
| Landform classification | ✅ | ✅ (PennockLandformClass) | ✅ (TPI-based) | ❌ | Weiss/Jenness |
| **Fuzzy landform elements** | ❌ | ❌ | ✅ | ❌ | SAGA único |
| **Iwahashi-Pike classification** | ❌ | ❌ | ✅ | ❌ | SAGA único |

#### 6.2.4 Terreno — Visibilidad e iluminación

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| Viewshed | ✅ | ✅ | ❌ | ❌ | |
| **VisibilityIndex** (cumulative viewshed) | ❌ | ✅ | ❌ | ❌ | |
| SVF | ✅ | ✅ | ❌ | ❌ | Zakšek 2011 |
| Openness (+/-) | ✅ | ✅ | ❌ | ❌ | Yokoyama 2002 |
| **HorizonAngle** | ❌ | ✅ | ❌ | ✅ (r.horizon) | |
| **HorizonArea** | ❌ | ✅ (WTE) | ❌ | ❌ | WTE de pago |
| **AverageHorizonDistance** | ❌ | ✅ (WTE) | ❌ | ❌ | WTE de pago |
| **SkylineAnalysis** | ❌ | ✅ | ❌ | ❌ | |
| Solar radiation | ✅ | ❌ | ✅ | ✅ (r.sun) | Hofierka & Šúri 2002 |
| Wind exposure | ✅ | ✅ (ExposureTowardsWindFlux) | ✅ | ❌ | |
| **Diurnal anisotropic heating** | ❌ | ❌ | ✅ | ❌ | SAGA único |
| **Morphometric protection index** | ❌ | ❌ | ✅ | ❌ | SAGA único |

#### 6.2.5 Terreno — Rugosidad y textura

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Notas |
|---|---|---|---|---|---|
| **MultiscaleRoughness** | ❌ | ✅ | ❌ | ❌ | |
| **MultiscaleStdDevNormals** | ❌ | ✅ | ❌ | ❌ | |
| **SphericalStdDevOfNormals** | ❌ | ✅ | ❌ | ❌ | |
| **CircularVarianceOfAspect** | ❌ | ✅ | ❌ | ❌ | |
| **AverageNormalVectorAngularDev** | ❌ | ✅ | ❌ | ❌ | |
| **Vector Ruggedness Measure (VRM)** | ❌ | ❌ | ✅ | ✅ (add-on) | Sappington 2007 |
| **StandardDeviationOfSlope** | ❌ | ✅ | ❌ | ❌ | |
| **Surface area ratio** | ❌ | ✅ | ✅ (Real Area) | ❌ | |
| **EdgeDensity** | ❌ | ✅ | ❌ | ❌ | |
| **Terrain Surface Texture** | ❌ | ❌ | ✅ | ✅ (r.texture) | |
| **Terrain Surface Convexity** | ❌ | ❌ | ✅ | ❌ | SAGA único |

#### 6.2.6 Terreno — Escalado multiescala

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| Feature-preserving smoothing | ✅ | ✅ | ❌ | ❌ | Sun 2007 |
| **Gaussian scale-space (fGSS)** | ❌ | ✅ (GaussianScaleSpace) | ❌ | ❌ | **Newman 2022** |
| **DirectionalRelief** | ❌ | ✅ | ❌ | ❌ | |
| **FetchAnalysis** | ❌ | ✅ | ❌ | ❌ | |
| **DownslopeIndex** | ❌ | ✅ | ✅ (Downslope Dist Gradient) | ❌ | |
| **Mass Balance Index** | ❌ | ❌ | ✅ | ❌ | SAGA único |
| **BreaklineMapping** | ❌ | ✅ (WTE) | ❌ | ❌ | WTE de pago |
| **EmbankmentMapping** | ❌ | ✅ | ❌ | ❌ | |

#### 6.2.7 Hidrología — Flow direction y accumulation

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| D8 flow direction | ✅ | ✅ | ✅ | ✅ (r.watershed) | O'Callaghan & Mark 1984 |
| D8 flow accumulation | ✅ | ✅ | ✅ | ✅ | |
| **D-infinity direction** | ❌ | ✅ (DInfPointer) | ❌ | ❌ | **Tarboton 1997** |
| **D-infinity accumulation** | ❌ | ✅ (DInfFlowAccumulation) | ❌ | ❌ | **Tarboton 1997** |
| **FD8 direction** | ❌ | ✅ (Fd8Pointer) | ❌ | ❌ | **Quinn et al. 1991** |
| **FD8 accumulation** | ❌ | ✅ (Fd8FlowAccumulation) | ❌ | ❌ | **Kopecky 2021: el mejor para TWI** |
| **Quinn accumulation** | ❌ | ✅ (QuinnFlowAccumulation) | ❌ | ❌ | Quinn et al. 1991 |
| **Qin accumulation** | ❌ | ✅ (QinFlowAccumulation) | ❌ | ❌ | **Qin 2011 (128 cit.)** |
| **MDInf accumulation** | ❌ | ✅ (MdInfFlowAccumulation) | ❌ | ❌ | |
| **Rho8 direction** | ❌ | ✅ | ❌ | ❌ | |
| MFD (SAGA variante) | ❌ | ❌ | ✅ | ❌ | |
| **r.terraflow (massive grids)** | ❌ | ❌ | ❌ | ✅ | I/O-efficient |

#### 6.2.8 Hidrología — Depression handling

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| Fill depressions (Planchon-Darboux) | ✅ | ✅ | ✅ | ✅ (r.fill.dir) | Planchon & Darboux 2002 |
| Fill (Wang-Liu) | ❌ | ✅ | ❌ | ❌ | Wang & Liu 2006 |
| **Breach depressions (Lindsay)** | ❌ | ✅ | ❌ | ❌ | **Lindsay 2016 — preferido sobre fill** |
| **Breach least-cost** | ❌ | ✅ | ❌ | ❌ | |
| **Priority-Flood** | ❌ | ❌ | ❌ | ❌ | **Barnes 2014 (181 cit.) — SOTA** |
| **Stochastic depression analysis** | ❌ | ✅ | ❌ | ❌ | |
| **DepthInSink** | ❌ | ✅ | ❌ | ❌ | |
| **Nested depression delineation** | ❌ | ❌ | ❌ | ❌ | **Wu 2019 (55 cit.) — ninguno** |

#### 6.2.9 Hidrología — Índices y análisis de cuenca

| Algoritmo | SurtGIS | WhiteboxTools | SAGA | GRASS | Paper base |
|---|---|---|---|---|---|
| TWI | ✅ | ✅ (WetnessIndex) | ✅ | ✅ (r.topidx) | Beven & Kirkby 1979 |
| SPI | ✅ | ✅ (StreamPowerIndex) | ✅ | ❌ | |
| STI | ✅ | ✅ (SedimentTransportIndex) | ❌ | ❌ | |
| Watershed delineation | ✅ | ✅ | ✅ | ✅ | |
| **HAND (Height Above Nearest Drainage)** | ❌ | ✅ (ElevAboveStream) | ❌ | ❌ | **Nobre 2011 — clave para inundaciones** |
| **DepthToWater** | ❌ | ✅ | ❌ | ❌ | |
| **Strahler order basins** | ❌ | ✅ | ❌ | ❌ | |
| **Isobasins** | ❌ | ✅ | ❌ | ❌ | |
| **Hillslopes (left/right bank)** | ❌ | ✅ | ❌ | ❌ | |
| **HydrologicConnectivity** | ❌ | ✅ | ❌ | ❌ | |
| **ImpoundmentSizeIndex** | ❌ | ✅ | ❌ | ❌ | |
| **FloodOrder** | ❌ | ✅ | ❌ | ❌ | |
| **Stream network extraction** | ❌ | ✅ | ✅ | ✅ | |
| **Flowpath analysis (avg/max length)** | ❌ | ✅ (6+ tools) | ❌ | ✅ (r.flow) | |
| **Hypsometric analysis** | ❌ | ✅ | ✅ | ❌ | |

---

### 6.3 Algoritmos WTE (pago) de WhiteboxTools — Candidatos a implementar como open-source

Los siguientes algoritmos son **de pago** en WhiteboxTools y representan oportunidades de diferenciación si SurtGIS los implementa como open-source:

| Algoritmo WTE | Categoría | Descripción | Paper base | Prioridad |
|---|---|---|---|---|
| **AccumulationCurvature** | Curvatura | Producto de curvatura profile × tangential | Florinsky 2023 | Alta |
| **Curvedness** | Curvatura | RMS de curvaturas máxima y mínima | Florinsky 2023 | Alta |
| **DifferenceCurvature** | Curvatura | (profile − tangential) / 2 | Florinsky 2023 | Alta |
| **GeneratingFunction** | Curvatura | Deflexión de curvatura tangential desde loci de curvatura extrema | Florinsky 2023 | Media |
| **HorizontalExcessCurvature** | Curvatura | Exceso horizontal sobre curvatura plan | Florinsky 2023 | Media |
| **BreaklineMapping** | Terreno | Detección automática de quiebres de pendiente | — | Alta |
| **HorizonArea** | Visibilidad | Área del polígono de horizonte por celda | — | Media |
| **AverageHorizonDistance** | Visibilidad | Distancia media al horizonte | — | Media |
| **LocalHypsometricAnalysis** | Morfometría | Hipsometría en ventanas locales | — | Media |
| **DemVoidFilling** | Preprocesamiento | Fusión de dos DEMs para rellenar vacíos | — | Baja |
| **AssessRoute** | Análisis | Evaluación de ruta (pendiente, elevación, visibilidad) | — | Baja |
| **ShadowAnimation** | Visualización | Animación de sombras para un día | — | Baja |

**Valor estratégico**: Las curvaturas de Florinsky (6 algoritmos WTE) son fórmulas matemáticas publicadas en Florinsky (2023, Digital Terrain Analysis, 3rd ed.). Implementarlas como open-source en SurtGIS ofrecería la **única alternativa libre** a WhiteboxTools WTE para estas métricas.

---

### 6.4 Algoritmos que nadie tiene — Ventaja competitiva potencial

Estos algoritmos están publicados en la literatura pero **ningún** competidor open-source los implementa:

| Algoritmo | Paper | Cit. | Impacto | Notas |
|---|---|---|---|---|
| **MSV (Multiescale Valleyness)** | Wang 2009 | 9 | Detecta features que MRVBF, D8, D-inf y flow acum fallan en capturar | Complemento natural a MRVBF de SurtGIS |
| **Priority-Flood (Barnes 2014)** | Barnes 2014 | 181 | Estado del arte absoluto para depression filling | WBT usa breach/fill, no PF puro |
| **Nested depression delineation** | Wu 2019 | 55 | ~150× más rápido que contour tree | Único enfoque level-set publicado |
| **TFGA (facet-to-facet MFD)** | Li 2024 | 0 | ~1 orden de magnitud más preciso | Publicado 2024, nadie lo implementa |
| **O(N) parallel watershed** | Zhou 2026 | 0 | El más rápido publicado | Publicado 2026, nadie lo implementa |
| **MFD adaptativo (Qin)** | Qin 2011 | 128 | TWI más preciso que Quinn MFD | WBT tiene QinFlowAccumulation pero no la variante adaptativa |
| **Hierarchical terrain classification** | Moeller 2008 | 54 | 89% precisión en landforms | Mass balance + segmentación |
| **Método espectral analítico (Chebyshev)** | Florinsky & Pankratov 2016 | — | Unifica interpolación + filtrado + derivadas en framework global analítico. Parámetro de escala continuo | Solo existe implementación MATLAB no publicada. **SurtGIS sería la primera open-source** |
| **2D-SSA para DEMs** | Golyandina, Usevich & Florinsky 2007 | — | Denoising model-free + descomposición multiescala. Método preferido de Florinsky | Ninguna implementación GIS. Solo MATLAB/R genérico |
| **12 curvaturas completas de Florinsky** | Florinsky 2025 (Cap. 2) | — | Sistema formal: 3 independientes + 6 simples + 3 totales. WBT cobra por 6 de ellas (WTE) | **Única alternativa open-source** a WBT Extension |
| **Fórmulas 5×5 cerradas (derivadas 3er orden)** | Florinsky 2009 / 2025 Cap. 4 | — | Mejores que Evans-Young 3×3. Permiten horizontal deflection (Dk_h) para líneas estructurales | Ningún competidor usa estas fórmulas específicas |

---

### 6.5 Roadmap priorizado de nuevos algoritmos

#### Prioridad 1 — Impacto inmediato (cierra gaps críticos vs competidores)

| # | Algoritmo | Módulo SurtGIS | Justificación | Ref. bibliográfica |
|---|---|---|---|---|
| 1 | **FD8 / Quinn MFD** (flow direction + accumulation) | `hydrology::flow_direction_mfd` | Duplica la calidad de TWI vs D8 actual. Consenso unánime en bibliografía (6+ papers). WBT lo tiene, SurtGIS no. | Kopecky 2021 (234 cit.), Quinn 1991 |
| 2 | **D-infinity** (flow direction + accumulation) | `hydrology::flow_direction_dinf` | Ángulos continuos, ampliamente citado. WBT lo tiene, SurtGIS no. | Tarboton 1997 |
| 3 | **Breach depressions** | `hydrology::breach` | Lindsay (2016): breaching es preferido sobre filling en la mayoría de casos. Es el default de WBT. | Lindsay 2016 |
| 4 | **DEV (Deviation from Mean Elevation)** | `terrain::dev` | Superior a TPI en paisajes heterogéneos (509 cit.). WBT lo tiene, SAGA/GRASS no. | De Reu 2013, Newman 2018 |
| 5 | **HAND (Height Above Nearest Drainage)** | `hydrology::hand` | Clave para mapeo de inundaciones. WBT lo tiene, nadie más. | Nobre 2011 |
| 6 | **Priority-Flood** (depression filling) | `hydrology::fill_priority_flood` | SOTA para depression filling (181 cit.). Reemplazar Planchon-Darboux como default. **Nadie lo tiene como open-source puro.** | Barnes 2014 |

#### Prioridad 2 — Diferenciación (superar a competidores)

| # | Algoritmo | Módulo SurtGIS | Justificación | Ref. bibliográfica |
|---|---|---|---|---|
| 7 | **Curvaturas avanzadas (Florinsky)** — 12 métricas adicionales | `terrain::curvature_advanced` | Gaussian, maximal, minimal, accumulation, difference, ring, excess H/V, unsphericity, curvedness, shape index, rotor, generating function. Son WTE de pago en WBT. Implementar open-source da ventaja. | Florinsky 2023, Digital Terrain Analysis 3rd ed. |
| 8 | **Gaussian scale-space (fGSS)** | `terrain::gaussian_scale_space` | Framework óptimo para análisis multiescala. WBT lo tiene, nadie más. | Newman 2022 |
| 9 | **MultiscaleTopographicPosition** | `terrain::multiscale_topo_position` | Integral images para TPI/DEV multiescala. Único en WBT. | Newman 2018 |
| 10 | **Multidirectional hillshade** | `terrain::hillshade` (extensión) | Mejor visualización que hillshade estándar. Solo WBT. | |
| 11 | **Vector Ruggedness Measure (VRM)** | `terrain::vrm` | Complemento a TRI basado en vectores normales. SAGA y GRASS add-on. | Sappington 2007 |
| 12 | **Stream network extraction** | `hydrology::stream_network` | WBT, SAGA y GRASS lo tienen; SurtGIS no. Fundamental para workflows hidrológicos completos. | |
| 13 | **BreaklineMapping** | `terrain::breaklines` | Detección automática de quiebres de pendiente. WTE de pago. | |

#### Prioridad 3 — Estado del arte de investigación (nadie lo tiene)

| # | Algoritmo | Módulo SurtGIS | Justificación | Ref. bibliográfica |
|---|---|---|---|---|
| 14 | **MSV (Multiescale Valleyness)** | `terrain::msv` | Ningún competidor lo implementa. Complementa MRVBF. | Wang 2009 |
| 15 | **TFGA (facet-to-facet MFD)** | `hydrology::flow_direction_tfga` | ~1 orden de magnitud más preciso que MFD existentes. Paper 2024, nadie lo tiene. | Li 2024 |
| 16 | **Nested depression delineation** | `hydrology::nested_depressions` | 150× más rápido. Ningún competidor. | Wu 2019 (55 cit.) |
| 17 | **MFD adaptativo** | `hydrology::flow_direction_mfd_adaptive` | Mejora TWI usando max downslope gradient. | Qin 2011 (128 cit.) |
| 18 | **O(N) parallel watershed** | `hydrology::watershed` (optimización) | El más rápido publicado (2026). | Zhou 2026 |
| 19 | **Hypsometric analysis** (local + global) | `terrain::hypsometry` | WBT y SAGA lo tienen; SurtGIS no. | |
| 20 | **Flowpath analysis** (longitud, slope promedio) | `hydrology::flowpath` | WBT tiene 6+ herramientas de flowpath; SurtGIS ninguna. | |

---

### 6.6 Síntesis estratégica

#### Posición competitiva actual

| Categoría | SurtGIS vs WBT | SurtGIS vs SAGA | SurtGIS vs GRASS |
|---|---|---|---|
| Curvaturas básicas | Par | Par | Par |
| Curvaturas avanzadas | **Muy atrás** (0 vs 14) | Ligeramente atrás | Adelante |
| Posición topográfica | Atrás (TPI vs DEV+5) | Par | Adelante |
| Geomorphons | Par | No tiene | Par |
| Flow direction | **Muy atrás** (1 vs 7) | Atrás | Atrás |
| Depression handling | Atrás (1 vs 5) | Par | Par |
| Índices hidrológicos | Par (TWI/SPI/STI) | Par | Par |
| MRVBF | ✅ único con WBT | Par | Par (add-on) |
| Viewshed/SVF/Openness | **Adelante** | **Adelante** | Par |
| Solar radiation | Par | Par | Par |
| Smoothing | Par con WBT | **Adelante** | **Adelante** |
| Multiescala | Atrás | Atrás | Atrás |
| Visualización (hillshade++) | Atrás | Atrás | Atrás |

#### Estrategia recomendada

1. **Cerrar el gap hidrológico primero** (P1: items 1-6): SurtGIS tiene solo D8 mientras WBT tiene 7 algoritmos de flow. Implementar FD8, D-inf y breach cierra la brecha más visible.

2. **Implementar curvaturas de Florinsky como open-source** (P2: item 7): Las 12 métricas adicionales son WTE de pago en WBT. Ser la alternativa libre es diferenciación directa.

3. **Capturar "papers sin implementación"** (P3: items 14-18): Algoritmos publicados que ningún competidor implementa. Posiciona a SurtGIS como herramienta de investigación de vanguardia.

4. **Priorizar WASM**: Cada algoritmo nuevo funciona automáticamente en browser vía `maybe_rayon`. Ningún competidor ofrece terreno/hidrología en WASM.

#### Impacto por esfuerzo

| Algoritmo | Esfuerzo estimado | Impacto | Ratio |
|---|---|---|---|
| DEV | Bajo (variante de TPI) | Alto (509 cit.) | ★★★★★ |
| FD8/Quinn MFD | Medio | Crítico (duplica TWI) | ★★★★★ |
| Curvaturas Florinsky (12) | Medio (fórmulas publicadas) | Alto (WTE open-source) | ★★★★ |
| HAND | Bajo (D8 + traverse) | Alto (inundaciones) | ★★★★★ |
| Breach depressions | Medio | Alto (default WBT) | ★★★★ |
| D-infinity | Medio | Alto (citadísimo) | ★★★★ |
| Priority-Flood | Medio-alto | Muy alto (SOTA) | ★★★★ |
| Multidirectional hillshade | Bajo | Medio (visualización) | ★★★★ |
| VRM | Bajo | Medio | ★★★★ |
| Gaussian scale-space | Alto | Alto (framework) | ★★★ |
| TFGA | Alto | Muy alto (SOTA 2024) | ★★★ |
| MSV | Medio | Alto (exclusivo) | ★★★ |

---

## 7. Revisión del Libro de Referencia: Florinsky (2025)

**Fuente**: Florinsky, I.V. (2025). *Digital Terrain Analysis*. 3rd Edition. Elsevier. 479 pp.
**Archivo**: `biblio/9780443247996.pdf` (151 MB)
**Estado**: **Libro completo — 21 de 21 capítulos revisados** (Partes I-IV).
**Documentos generados**: `docs/florinsky_ch{2,4,5,6,7,8}_equations.tex` (ecuaciones LaTeX), `docs/florinsky_ch4_catalog.md`

### 7.1 Estrategia de revisión

Revisión capítulo por capítulo, priorizados por relevancia para SurtGIS:

| Cap | Título | Relevancia | Estado |
|-----|--------|------------|--------|
| 2 | Morphometric Variables | ★★★★★ | **Completado** |
| 4 | Calculation Methods | ★★★★★ | **Completado** |
| 6 | Filtering and Generalizing | ★★★★ | **Completado** |
| 7 | Spectral Analytical Modeling | ★★★★ | **Completado** |
| 1 | State of the Art 2016-2024 | ★★★ | **Completado** |
| 3 | Digital Elevation Models | ★★★ | **Completado** |
| 5 | Errors and Accuracy | ★★★ | **Completado** |
| 8 | Mapping and Visualization | ★★★ | **Completado** |
| 21 | Concluding Remarks & Pending Problems | ★★★★ | **Completado** |
| 11 | Predictive Soil Mapping | ★★★ | **Completado** |
| 13 | Folds and Folding | ★★★ | **Completado** |
| 14 | Lineaments and Faults | ★★★★ | **Completado** |
| 16 | Global DTA & Tectonic Structures | ★★★★ | **Completado** |
| 9 | Influence of Topography on Soil Properties | ★★★ | **Completado** |
| 10 | Adequate Resolution of Models | ★★★★ | **Completado** |
| 12 | Analyzing Relationships in Topography-Soil System | ★★★★ | **Completado** |
| 15 | Accumulation Zones and Fault Intersections | ★★★ | **Completado** |
| 17 | Glacier Motion and Evolution | ★★ | **Completado** |
| 18 | Crevasses | ★★★ | **Completado** |
| 19 | Catastrophic Glacier Events | ★★ | **Completado** |
| 20 | Antarctic Oases | ★★★ | **Completado** |

**Nota técnica**: Las páginas impresas del PDF tienen un offset de +22 páginas (p. impresa 17 = p. PDF 39).

### 7.2 Capítulo 2 — Variables Morfométricas

Florinsky define un catálogo formal de **57 variables y clasificaciones morfométricas**, organizadas en:

#### 7.2.1 Sistema Completo de 12 Curvaturas

Florinsky define formalmente un sistema de 12 curvaturas en 3 grupos:

| Grupo | Curvaturas | Unidad |
|-------|-----------|--------|
| **Independientes** (3) | H (media), E (diferencia), M (no-esfericidad) | m⁻¹ |
| **Simples** (6) | k_min, k_max, k_h, k_v, k_he, k_ve | m⁻¹ |
| **Totales** (3) | K (Gaussiana), K_a (acumulación), K_r (anillo) | m⁻² |

**Relación fundamental** (Teorema 2.7): K_a = K + K_r

Las 3 independientes (H, E, M) generan todas las demás:
- k_min = H − M, k_max = H + M
- k_h = H − E, k_v = H + E
- k_he = M − E, k_ve = M + E
- K = H² − M², K_a = H² − E², K_r = M² − E²

#### 7.2.2 Catálogo completo de variables

| # | Variable | Categoría | Orden | SurtGIS | Dificultad |
|---|----------|-----------|-------|---------|------------|
| 1 | Pendiente G | Local/forma | 1° | ✅ | — |
| 2 | Aspecto A | Local/forma | 1° | ✅ | — |
| 3 | Northness A_N | Local/forma | 1° | ❌ | Trivial |
| 4 | Eastness A_E | Local/forma | 1° | ❌ | Trivial |
| 5 | Factor gradiente GF | Local/forma | 1° | ❌ | Trivial |
| 6 | Plan curvature k_p | Local/forma | 2° | ✅ | — |
| 7 | Horizontal curvature k_h | Local/flujo | 2° | ✅ | — |
| 8 | Vertical curvature k_v | Local/flujo | 2° | ✅ | — |
| 9 | Mean curvature H | Local/forma | 2° | ✅ | — |
| 10 | Gaussian curvature K | Local/forma | 2° | ✅ | — |
| 11 | Minimal curvature k_min | Local/forma | 2° | ✅ | — |
| 12 | Maximal curvature k_max | Local/forma | 2° | ✅ | — |
| 13 | Unsphericity M | Local/forma | 2° | ❌ | Trivial |
| 14 | Difference E | Local/flujo | 2° | ❌ | Trivial |
| 15 | Horizontal excess k_he | Local/flujo | 2° | ❌ | Trivial |
| 16 | Vertical excess k_ve | Local/flujo | 2° | ❌ | Trivial |
| 17 | Accumulation K_a | Local/flujo | 2° | ❌ | Trivial |
| 18 | Ring K_r | Local/flujo | 2° | ❌ | Trivial |
| 19 | Rotor rot | Local/flujo | 2° | ❌ | Trivial |
| 20 | Laplaciano ∇²z | Local/forma | 2° | ⚠️ | Trivial |
| 21 | Deflexión horizontal Dk_h | Local/deflexión | 3° | ❌ | Media-alta |
| 22 | Deflexión vertical Dk_v | Local/deflexión | 3° | ❌ | Media-alta |
| 23 | Catchment area CA | No-local | — | ✅ | — |
| 24 | Dispersive area DA | No-local | — | ❌ | Media |
| 25 | Specific catchment SCA | No-local | — | ⚠️ | Trivial |
| 26 | Specific dispersive SDA | No-local | — | ❌ | Media |
| 27 | TPI | No-local | — | ✅ | — |
| 28-31 | Líneas estructurales | Estructural | 3° | ❌ | Alta |
| 32 | Hillshade R_L | Solar | — | ✅ | — |
| 33 | Insolación I | Solar | — | ✅ | — |
| 34 | Ángulo horizonte σ | Solar | — | ✅ | — |
| 35 | Sky view factor C_s | Solar | — | ✅ | — |
| 36 | Aspecto relativo A_r | Eólico | — | ❌ | Trivial |
| 37 | WEf, WEx | Eólico | — | ✅ | — |
| 38 | Viewshed V | Visual | — | ✅ | — |
| 39 | Total viewshed VS | Visual | — | ❌ | Alta (O(n²)) |
| 40 | Openness TO± | Visual | — | ✅ | — |
| 41 | TWI | Combinada | — | ✅ | — |
| 42 | SPI | Combinada | — | ✅ | — |
| 43 | Shape index I_S | Combinada | — | ❌ | Trivial |
| 44 | Curvedness C | Combinada | — | ❌ | Trivial |

**Resumen**: 21 implementadas, 2 parciales, 21 no implementadas. De las no implementadas: **14 triviales**, 4 medias, 2 media-alta, 1 alta.

#### 7.2.3 Clasificaciones de formas del terreno

| # | Clasificación | Variables usadas | Clases | SurtGIS | Dificultad |
|---|--------------|------------------|--------|---------|------------|
| 1 | Gaussiana | K, H | 4-9 | ❌ | Trivial |
| 2 | Efremov-Krcho | k_v, k_h | 4-9 | ❌ | Trivial |
| 3 | Shary | K, H, E, k_h, k_v | 12-46 | ❌ | Media |
| 4 | Pennock | k_v, k_h, G | 7 | ❌ | Trivial |
| 5 | MacMillan | G, k_v, k_h, TWI | 15 | ❌ | Media-alta |
| 6 | Schmidt-Hewitt | k_h, k_v, k_max, k_min, BTH | 45 | ❌ | Media-alta |
| 7 | TPI-based (Weiss) | TPI_S, TPI_L, G | 10 | ✅ | — |
| 8 | Iwahashi-Pike | G, ∇²z, textura | 8-16 | ❌ | Media |
| 9 | Geomorfones | Comparación ternaria 8 dirs | 10 | ✅ | — |

#### 7.2.4 Hallazgo crítico: Auditoría de fórmulas de curvatura

Las curvaturas 3×3 de SurtGIS usan **fórmulas simplificadas de Zevenbergen & Thorne (1987)** que omiten el denominador `√(1+p²+q²)`:

| Pendiente | Error estimado |
|-----------|---------------|
| < 10° | < 3% |
| 30° | ~15% |
| 45° | ~30% |

**Recomendación**: Ofrecer ambas opciones (simplified/florinsky) y documentar la diferencia. El módulo `multiscale_curvatures` ya usa fórmulas completas con ventana 5×5.

**Distinción crítica k_p vs k_h**: La curvatura plan (k_p) y la curvatura horizontal (k_h) comparten numerador pero difieren en denominador. k_h = k_p · sin(γ). Esta confusión es frecuente en la literatura.

#### 7.2.5 Variables de implementación rápida (batch)

Las siguientes 14 variables pueden implementarse con solo una fórmula, sin nueva infraestructura:

1. **A_N** = cos(A) — Northness
2. **A_E** = sin(A) — Eastness
3. **GF** = √(p² + q²) — Gradient factor
4. **M** = √(H² − K) — Unsphericity
5. **E** = (k_v − k_h)/2 — Difference curvature
6. **k_he** = M − E — Horizontal excess
7. **k_ve** = M + E — Vertical excess
8. **K_a** = k_h · k_v — Accumulation curvature
9. **K_r** = M² − E² — Ring curvature
10. **rot** = ((p²−q²)s − pq(r−t)) / (p²+q²)^(3/2) — Rotor
11. **I_S** = (2/π)·arctan(H/√(H²−K)) — Shape index
12. **C** = √((k_max² + k_min²)/2) — Curvedness
13. **A_r** = |A − δ| — Relative terrain aspect
14. **∇²z** = r + t — Laplacian (explícito)

Todas las fórmulas completas están en `docs/florinsky_ch2_equations.tex`.

### 7.3 Capítulo 4 — Métodos de Cálculo (completado)

**Documentos generados**: `docs/florinsky_ch4_equations.tex`, `docs/florinsky_ch4_catalog.md`

#### 7.3.1 Tres métodos de estimación de derivadas

| Método | Ventana | Derivadas | Precisión | Referencia |
|--------|---------|-----------|-----------|------------|
| **Evans-Young** | 3×3 | 1°, 2° (r,t,s,p,q) | Alta | Evans 1979 |
| **Florinsky 5×5** | 5×5 | 1°, 2°, **3°** (g,h,k,m,r,t,s,p,q) | **Máxima** | Florinsky 2009 |
| **Esferoidal 3×3** | 3×3 | 1°, 2° (r,t,s,p,q) | Alta | Florinsky 1998c |

**Florinsky recomienda su método 5×5 como estándar** sobre Evans-Young, con menor RMSE y mejor supresión de ruido de alta frecuencia. Validado en DEM de Stavropol Upland: sin diferencia significativa en distribuciones (K-S test P=0.77) pero RMSE esencialmente menor.

#### 7.3.2 Hallazgo CRÍTICO: `multiscale_curvatures.rs` incorrecto

La implementación actual en SurtGIS que dice usar "Florinsky's (2016) method" con ventana 5×5 **NO es correcta**:

1. Usa polinomio **cuadrático** (2° orden), no el **cúbico** (3° orden) de Florinsky
2. No puede calcular derivadas de 3er orden (g, h, k, m) → no puede computar Dkh
3. Usa sumas genéricas de mínimos cuadrados en vez de las fórmulas cerradas Eqs. 4.14-4.22

**SurtGIS usa Z&T y Horn** (no mencionados por Florinsky como recomendados). Él establece que Evans-Young es el mejor método 3×3.

#### 7.3.3 Gap: Sin soporte para grids esferoidales

SurtGIS asume grids planos cuadrados. Análisis de SRTM/ASTER/Copernicus DEM sin reproyección introduce errores sistemáticos. Florinsky (2017b) demostró estos errores.

#### 7.3.4 Otros hallazgos

- **Flow routing**: D8 (implementado en SurtGIS) es aceptable, pero MFD recomendado para hidrología
- **Deflexión horizontal Dkh**: Requiere derivadas de 3er orden. Florinsky nota limitaciones prácticas (loci interrumpidos, sensibilidad al ruido)
- **Clasificaciones**: Geomorfones y TPI-based (ambos en SurtGIS) son válidos
- **Elipsoide triaxial**: Solo para ciencia planetaria, baja prioridad

#### 7.3.5 Prioridades de implementación (del Cap. 4)

| Prioridad | Acción | Impacto |
|-----------|--------|---------|
| **P0-Crítica** | Implementar fórmulas cerradas Florinsky 5×5 (Eqs. 4.14-4.22) como módulo compartido `derivatives` | Fundamento de todo análisis de terreno |
| **P0-Crítica** | Reemplazar Z&T por Evans-Young como fallback 3×3 | Precisión inmediata |
| **P1-Alta** | Soporte grid esferoidal (Eqs. 4.27-4.31 + 4.32-4.42) | DEMs globales correctos |
| **P2-Media** | Agregar opción MFD para flow routing | Mejor hidrología |
| **P2-Media** | Implementar Dkh con derivadas 3er orden | Líneas estructurales |
| **P3-Baja** | Elipsoide triaxial | Solo planetario |

### 7.4 Capítulo 6 — Filtrado y Generalización (completado)

Florinsky identifica tres tareas del filtrado: descomposición (separar frecuencias), denoising (suprimir ruido antes de derivadas) y generalización. **Principio clave**: "Los mapas de curvatura derivados de DEMs sin denoisear son casi ilegibles y no pueden usarse en investigación."

#### 7.4.1 Métodos descritos por Florinsky

| Método | Tipo | Ecuaciones | Preserva rasgos | SurtGIS |
|--------|------|-----------|-----------------|---------|
| Media ponderada | Local | 6.8–6.9 | No (inversión de fase) | Parcial (focal mean, sin pesos ni iteración) |
| Mediana | Local | — | Bordes | ✅ (focal statistics) |
| **Gaussiano** | Local | **6.10** | No | ❌ **Falta** |
| **FPDEMS** (Lindsay 2019) | Local | **6.11–6.12** | Sí (2 etapas) | ⚠️ Bilateral, no FPDEMS real |
| FFT pasa-bajo | Global | 6.2–6.5 | No | ❌ |
| Wavelet 2D | Global | 6.6–6.7 | Parcial | ❌ |
| 2D-SSA (Golyandina) | Global | 6.18–6.22 | Sí | ❌ |
| Método de corte | Post | — | N/A | ❌ (trivial) |

#### 7.4.2 Hallazgo: FPDEMS de SurtGIS no es correcto

El `smoothing.rs` actual usa filtro bilateral de una etapa con decaimiento Gaussiano. El FPDEMS real (Lindsay et al. 2019 / Florinsky Eq. 6.11-6.12) es de **dos etapas**:
1. **Suavizar normales**: peso = cos²(ángulo) / Σcos² con umbral duro Θ_t
2. **Ajustar elevaciones**: desde planos tangentes de vecinos con normales suavizadas

#### 7.4.3 Lo que Florinsky usa en su propia investigación

- Su método de diferencias finitas 5×5 (Cap. 4), que incluye suavizado implícito
- **Media iterativa 3×3** como preprocesamiento: 3 iteraciones para estudios regionales, hasta 10 para DEMs ruidosos
- **2D-SSA** para denoising de calidad investigativa
- Transformación logarítmica de modelos de curvatura (Eq. 8.1) para visualización

#### 7.4.4 Prioridades de implementación (del Cap. 6)

| Prioridad | Método | Complejidad | Valor |
|-----------|--------|-------------|-------|
| **P1** | Suavizado Gaussiano (Eq. 6.10) | Baja | Alto — funcionalidad estándar ausente |
| **P1** | Media iterativa con pesos (Eqs. 6.8-6.9, m=0,1,2) | Baja | Alto — workhorse de Florinsky |
| **P1** | Método de corte (curvatura < k_t → 0) | Trivial | Medio |
| **P2** | FPDEMS correcto de 2 etapas (Eqs. 6.11-6.12) | Media | Alto — fix de implementación actual |
| **P2** | FFT pasa-bajo (Eqs. 6.2-6.3) | Media | Alto — crate `rustfft` disponible |
| **P3** | Wavelet denoising (Eqs. 6.6-6.7) | Alta | Medio |
| **P3** | 2D-SSA (Eqs. 6.18-6.22) | Muy alta | Muy alto — preferido de Florinsky |

**TeX generado**: `docs/florinsky_ch6_equations.tex`

### 7.5 Capítulo 7 — Modelado Espectral Analítico Universal (completado)

#### 7.5.1 Concepto

Método global basado en **polinomios de Chebyshev** que resuelve interpolación, filtrado y cálculo de derivadas en un único framework analítico. A diferencia de los métodos locales (ventana móvil), una sola función analítica define la superficie completa del terreno. Las derivadas parciales son también funciones analíticas (no diferencias finitas).

| Aspecto | Métodos tradicionales | Método espectral |
|---------|----------------------|------------------|
| Alcance | Local (3×3, 5×5) | Global (todo el DEM) |
| Escala | Fija por ventana | Continua (parámetro $l$) |
| Derivadas | Diferencias finitas | Analíticas en dominio espectral |
| Framework | Separado por tarea | Unificado (interpolación + filtrado + derivadas) |

#### 7.5.2 Algoritmo en 4 etapas

1. **Expansión**: Serie bivariada de Chebyshev z(x,y) = Σ c_ij T_i(x) T_j(y) (Eq. 7.1)
2. **Coeficientes**: Cuadratura de Gauss + interpolación lineal del DEM a nodos (Eq. 7.7)
3. **Fejér**: Atenuación lineal c̃_j = (l-j)/l · c_j para suprimir oscilaciones de Gibbs (Eq. 7.9)
4. **Derivadas**: Recurrencia para coeficientes de derivadas sin recalcular (Eqs. 7.10-7.12)

Implementación matricial (Eqs. 7.18-7.22): Solo multiplicación de matrices, sin resolución de sistemas lineales. Ideal para Rust + `ndarray`.

#### 7.5.3 Control de escala

El número de coeficientes $l$ actúa como parámetro de escala continuo (caso 480×481 DEM):

| $l$ | Efecto |
|-----|--------|
| 2880 | Reconstrucción casi idéntica (residuos ~0.1 ± 8.8 m, dentro del error SRTM) |
| 480 | Mapas de curvatura legibles con estructuras de flujo |
| 240 | Rasgos estructurales claros |
| 120 | Estructuras tectónicas a gran escala |

#### 7.5.4 Evaluación para SurtGIS

**Prioridad: ALTA**

- **Capacidad única**: Ninguna biblioteca open-source implementa este método (GDAL, GRASS, SAGA, WhiteboxTools). SurtGIS sería la primera y única implementación disponible.
- **Publicación**: IJGIS (Florinsky & Pankratov, 2016) + libro de referencia definitivo.
- **Valor práctico**: Una función que simultáneamente aproxima, denoisa, generaliza y calcula variables morfométricas.
- **Complejidad**: Media-alta. Solo multiplicación de matrices (`ndarray`). Sin dependencias externas pesadas.
- **Limitación**: DEM diagonal < 0.1 × radio del planeta (aplica a la mayoría de estudios regionales).

#### 7.5.5 vs `multiscale_curvatures.rs` actual

| Aspecto | `multiscale_curvatures.rs` | Método espectral |
|---------|---------------------------|------------------|
| Enfoque | 5×5 local LS | Chebyshev global |
| Escala | Fija (una ventana) | Continua (parámetro l) |
| Derivadas | Diferencias finitas locales | Analíticas globales |
| Denoising | Limitado | Inherente |
| Output | Un tipo de curvatura por call | Todas las derivadas simultáneamente |

No reemplaza `multiscale_curvatures` sino que lo complementa como herramienta de nivel superior para análisis multiescala.

**TeX generado**: `docs/florinsky_ch7_equations.tex`

### 7.6 Capítulo 1 — Estado del Arte 2016-2024 (completado)

Capítulo estratégico que valida la dirección de SurtGIS y enmarca las tendencias actuales.

#### 7.6.1 Tres Fuerzas Motrices 2016-2024

| Factor | Detalle |
|--------|---------|
| **UAS + SfM** | DEMs de resolución centimétrica son rutinarios. Revolución en glaciología, erosión, arqueología |
| **DEMs globales gratuitos** | Copernicus GLO-30 y ALOS AW3D30 son los mejores, pero fallan en pendientes pronunciadas |
| **Machine Learning** | Dominancia total: Random Forest, Deep Learning, SVM reemplazan métodos geoestadísticos |

#### 7.6.2 Hallazgo Clave: Variables Morfométricas como Covariables ML

El uso dominante de variables morfométricas en 2024 es como **covariables/predictores para modelos ML**, no interpretadas directamente. Las variables fundamentales (slope, aspect, curvaturas, TWI, catchment area) siguen siendo las más usadas en los 15+ dominios de aplicación. La selección correcta de predictores importa más que el algoritmo ML elegido.

#### 7.6.3 Gaps de SurtGIS según Ch.1

| Capacidad Faltante | Contexto Florinsky | Prioridad |
|--------------------|--------------------|-----------|
| **Flow accumulation / catchment area** | Variable no-local más citada. Usada en hidrología, suelos, geología, planetología | **Crítica** |
| **Watershed delineation** | Operación hidrológica fundamental | **Crítica** |
| **DEM preprocessing (fill/breach)** | Prerequisito para todo análisis hidrológico. Lindsay (2016) es SOTA | **Crítica** |
| **Stream network extraction** | Geomorfología, hidrología, geología | **Alta** |
| **Multi-temporal DEM differencing** | Estándar en glaciología, erosión, vulcanismo | **Alta** |

#### 7.6.4 Ventaja Competitiva de SurtGIS

Florinsky no menciona ninguna herramienta browser-based para análisis de terreno. SurtGIS con WASM sería genuinamente única. La ventaja de rendimiento de Rust aborda directamente el desafío de procesar los DEMs ultra-alta-resolución que UAS/SfM producen rutinariamente.

### 7.7 Capítulo 3 — Modelos Digitales de Elevación (completado)

#### 7.7.1 Tipos de Grilla

| Tipo | Descripción | Estado SurtGIS |
|------|-------------|----------------|
| **Cuadrada plana** | Grilla regular GDAL-compatible. `GeoTransform` con affine transform | Soportada |
| **Angular esferoidal** | Lat/lon con espaciado en arco-segundos. Todos los DEMs globales (SRTM, Copernicus, ETOPO, etc.) | **NO soportada** |
| **DGGS** | Teselación icosaédrica jerárquica. "Aún marginal" según Florinsky | No necesaria |

**Hallazgo crítico**: La Tabla 3.1 de Florinsky cataloga ~30 DEMs globales, **todos** en grilla angular esferoidal. Sin soporte esferoidal, SurtGIS no puede procesar correctamente ningún DEM global. Confirma P1-High del Cap. 4.

#### 7.7.2 Métodos de Interpolación — Auditoría SurtGIS

| Método | Florinsky | SurtGIS | Estado |
|--------|-----------|---------|--------|
| **IDW** (Eq. 3.1) | $F(x,y) = \sum w_i f_i / \sum w_i$, $w_i = 1/d_i^p$ | `idw.rs` con power, radio, max puntos | **Correcto** |
| **TIN Delaunay + lineal** | Triangulación + interpolación baricéntrica | `tin.rs` Bowyer-Watson + baricéntrico | **Correcto** (solo lineal) |
| **TIN + blending no lineal** | Funciones polinómicas para suavizado C1 | — | **Falta** |
| **Thin-plate spline** (Eq. 3.2) | Minimiza seminorma de rugosidad | — | **Falta** |
| **Spline regularizado con tensión** | Mitasova & Mitas (1993). Usado en GRASS `v.surf.rst` | — | **Falta** |

#### 7.7.3 Gaps Identificados

| Gap | Prioridad | Justificación |
|-----|-----------|---------------|
| **Indexación espacial** (k-d tree) | P1-Alta | Complejidad O(n·m) actual impráctica para nubes de millones de puntos |
| **Grilla angular esferoidal** | P1-Alta | Desbloquea todos los DEMs globales |
| **Thin-plate spline con tensión** | P1-Alta | Produce superficies suaves para derivadas (curvaturas). Método principal en GRASS |
| **TIN no lineal** (Clough-Tocher) | P2-Media | Superficie C1 vs C0 actual |
| **Validación de resolución** | P2-Media | $w \leq \lambda_\text{min} / (2n)$, avisar al usuario |

### 7.8 Capítulo 5 — Errores y Precisión (completado)

#### 7.8.1 Cadena de Propagación de Errores

Ecuaciones RMSE para **todas** las variables morfométricas, desde $m_z$ (RMSE de elevación) hasta $m_{K_r}$ (RMSE de curvatura de anillo). Cadena de dependencias:

```
mz → [Tabla 5.1] → mp, mq, mr, ms, mt
  → [Eq. 5.2] → mG
  → [Eqs. 5.3-5.4] → mkh, mkv
  → [Eq. 5.5] → mK
  → [Eq. 5.7] → mH = mE
  → [Eq. 5.9] → mM
  → [Eq. 5.6] → mkmin = mkmax
  → [Eqs. 5.8, 5.10-5.12] → mKa, mKr, mkhe, mkve
```

#### 7.8.2 Tabla 5.1 — Comparación Evans-Young vs Florinsky

| RMSE | Evans-Young (3×3) | Florinsky (5×5) | Ratio |
|------|-------------------|-----------------|-------|
| $m_r, m_t$ | $\sqrt{2} \cdot m_z / w^2$ | $\sqrt{2/35} \cdot m_z / w^2$ | Florinsky ~6× mejor |
| $m_s$ | $m_z / (2w^2)$ | $m_z / (10w^2)$ | Florinsky 5× mejor |
| $m_p, m_q$ | $m_z / (\sqrt{6} \cdot w)$ | $\sqrt{527/70} \cdot m_z / (\sqrt{6} \cdot w)$ | Evans-Young ~10% mejor |

**Conclusión**: Para curvaturas (derivadas de 2do orden), el método Florinsky 5×5 es **muy superior**. Para pendiente (derivadas de 1er orden), Evans-Young es marginalmente mejor.

#### 7.8.3 Resultados Teóricos Clave

| Resultado | Descripción |
|-----------|-------------|
| **Isotropía** | Todos los operadores morfométricos locales (excepto aspecto $A$) son isotrópicos (Sección 5.7.2). Lineamientos cardinales provienen de los datos, no de los operadores |
| **Teorema de muestreo** | $w \geq$ distancia media entre puntos. Interpolación NO aumenta resolución. Sobreresolucion crea artefactos amplificados por diferenciación |
| **Gibbs** | Sobrepaso 8.95% cerca de discontinuidades con interpolación suave. Remedio: sumación de Fejér |
| **Desplazamiento de grilla** | Diferencias estadísticamente insignificantes (Kolmogorov-Smirnov 95%) |

#### 7.8.4 Oportunidad: Mapas de Incertidumbre como Diferenciador

**Ninguna biblioteca open-source** (GDAL, GRASS, SAGA, WhiteboxTools) proporciona mapas de incertidumbre per-pixel junto con las salidas morfométricas. SurtGIS podría ser la primera en ofrecer:

```rust
fn slope_with_uncertainty(dem, mz) -> (slope_grid, mG_grid)
fn curvature_with_uncertainty(dem, mz) -> (kh_grid, mkh_grid)
```

Costo computacional: esencialmente "gratis" — usa las mismas derivadas parciales ya calculadas.

**TeX generado**: `docs/florinsky_ch5_equations.tex`

### 7.9 Capítulo 8 — Visualización y Mapeo (completado)

#### 7.9.1 Transformada Logarítmica (Eq. 8.1) — Diferenciador Clave

Fórmula para visualizar variables con rango dinámico muy amplio (especialmente curvaturas):

$$Y' = \text{sign}(Y) \cdot \ln(1 + 10^{n \cdot m} \cdot |Y|)$$

Parámetros:
- $n$: depende de resolución del DEM (Tabla 8.1)
- $m = 2$ para $K, K_a, K_r$ (curvaturas combinadas); $m = 1$ para el resto

| Espaciado (m) | $n$ | Ejemplo |
|----------------|-----|---------|
| < 1 | 2 | Lidar UAS |
| 1–10 | 3 | Lidar ALS |
| 10–100 | 4 | SRTM 30m |
| 100–1000 | 5 | SRTM 90m |
| 1000–5000 | 6 | Regional |
| 5000–75000 | 7–8 | Global |
| > 75000 | 9 | Planetario |

**Ningún competidor** implementa esta transformada como función built-in. SurtGIS sería el primero.

#### 7.9.2 Otras Técnicas de Visualización

| Técnica | Descripción | Prioridad SurtGIS |
|---------|-------------|-------------------|
| Color relief shading | Elevación coloreada × hillshade | Baja (trivial en web demo) |
| Normalización centrada en cero | Zero ↦ 0.5 en escala de color divergente | Baja (fix en `colormap.js`) |
| Extracción de perfiles | Muestreo bilineal a lo largo de polilínea | Media |
| Hachuras de Samsonov | Grosor = pendiente, color = aspecto | Baja (nicho) |
| Globe web 3D | Three.js/Cesium con texturas morfométricas | Alta (pero esfuerzo alto) |

#### 7.9.3 Evaluación para SurtGIS

La **transformada logarítmica (Eq. 8.1) + auto-detección de $n$** es la implementación más clara del capítulo: ~20 líneas de Rust + binding WASM. Alto impacto, bajo esfuerzo, genuinamente único.

**TeX generado**: `docs/florinsky_ch8_equations.tex`

### 7.10 Capítulo 21 — Conclusiones y Problemas Pendientes (completado)

Capítulo estratégico que identifica 10 problemas abiertos en geomorfometría. Relevancia directa para la hoja de ruta de SurtGIS.

#### 7.10.1 Los 10 Problemas Pendientes de Florinsky

| # | Problema | Impacto SurtGIS | Prioridad |
|---|----------|-----------------|-----------|
| P1 | Clasificación de índices de rugosidad — sin base teórica clara | Implementar con caveats | Media |
| P2 | Características texturales de Haralick — sin significado físico para DEMs | No prioritario | Baja |
| P3 | **Método espectral analítico universal** (Chebyshev + Fejér) — unifica interpolación, denoising, generalización y cálculo morfométrico | Diferenciador único; ningún competidor lo implementa | **Crítica** |
| P4 | **Cálculo vectorial de variables locales** — más suave, menos sensible a errores | Alternativa a métodos de diferencias finitas actuales | **Alta** |
| P5 | **Algoritmos de enrutamiento de flujo son "obsoletos"** — reemplazar con Gallant-Hutchinson analítico | Riesgo estratégico para módulo de hidrología | **Crítica** |
| P6 | Extracción de red de thalwegs sin base teórica | No invertir fuertemente en pipelines basados en flow routing | Alta |
| P7 | Geodesia en elipsoides multiaxiales | Nicho (planetología) | Baja |
| P8 | Selección de espaciado de grilla — sin solución universal | Funciones de diagnóstico/advisory | Media-Baja |
| P9 | Suavizado óptimo imposible sin dañar señal | Limitación inherente — documentar | Baja |
| P10 | Frontera entre suavizado y generalización | Diseño de API claro | Media |

#### 7.10.2 Direcciones Futuras Clave

1. **Método espectral analítico (Chebyshev)**: Pipeline unificado DEM. Extensión a Legendre, Fourier, Jacobi, Hermite.
2. **Atributos no-locales analíticos**: Ecuación diferencial de Gallant-Hutchinson como reemplazo de D8/MFD. Refs: Qin et al. (2017), Bonetti et al. (2018), Li Z. et al. (2021).
3. **Cálculo vectorial**: Modelos más suaves y robustos. Refs: Hu G. et al. (2020, 2021).
4. **Investigación antártica**: Boom de DEMs polares — ventaja de rendimiento de Rust.

#### 7.10.3 Crítica Más Fuerte: Flow Routing "Obsoleto"

Florinsky llama explícitamente "obsoletos" a los algoritmos de enrutamiento de flujo (D8, MFD, D-infinity). Afirma que producen "resultados poco reproducibles e implausibles" y que diferentes algoritmos aplicados a la misma área producen "modelos de flujo muy diferentes". Propone reemplazarlos por métodos analíticos basados en la ecuación diferencial de Gallant-Hutchinson.

**Implicación para SurtGIS**: El módulo `hydrology::flow_direction` está construido sobre algoritmos que Florinsky considera fundamentalmente defectuosos. Riesgo estratégico de construir más funcionalidades sobre esta base.

### 7.11 Capítulo 13 — Pliegues y Theorema Egregium (completado)

#### 7.11.1 Theorema Egregium (Gauss 1828)

**Resultado fundamental**: La curvatura Gaussiana $K$ es invariante bajo deformaciones isométricas (doblado sin estiramiento/compresión/ruptura). Esto significa que $K$ preserva la geometría intrínseca de una superficie original incluso después de inclinación y doblado posteriores.

**Aplicación demostrada** (Shary et al., 1991):
- Superficie con estructura de plegamiento compleja → inclinada y doblada
- Mapas de $k_h$ y $k_v$: patrones cambian completamente pre/post inclinación
- Mapa de $K$: **patrones casi idénticos** — la estructura oculta es claramente visible

**Limitación práctica**: Terrenos reales violan la suposición isométrica por erosión, estiramiento tectónico, compresión y fallamiento.

#### 7.11.2 Clasificación de Pliegues por K-H

| K | H | Clase (Roberts 2001) | Clase (Lisle-Toimil 2007) |
|---|---|----------------------|--------------------------|
| K > 0 | H > 0 | Domo | Antiforme sinclástico |
| K > 0 | H < 0 | Cuenco | Sinforme sinclástico |
| K < 0 | cualquier | Silla | Antiforme/Sinforme anticlástico |
| K = 0 | H > 0 | Cresta | — |
| K = 0 | H = 0 | Plano | — |
| K = 0 | H < 0 | Sinforma | — |

#### 7.11.3 Variables Faltantes en SurtGIS

| Variable | Fórmula | Estado |
|----------|---------|--------|
| **Shape index** | $S = -\frac{2}{\pi} \arctan\frac{k_{max}+k_{min}}{k_{max}-k_{min}}$ | **Falta** — clasificación continua de forma superficial |
| **Curvedness** | $C = \sqrt{(k_{max}^2 + k_{min}^2)/2}$ | **Falta** — complementa shape index |
| Fold classification (6-8 clases) | sign(K) × sign(H) | **Falta** — trivial de implementar |

**Nota arquitectónica**: Todas las adiciones de alta prioridad se construyen sobre `multiscale_curvatures` existente (K, H, kmax, kmin ya calculados). Solo requieren post-procesamiento.

### 7.12 Capítulo 14 — Lineamientos y Fallas (completado)

#### 7.12.1 Algoritmo de Detección de Lineamientos

Método basado en mapas binarios de curvatura horizontal ($k_h$) y vertical ($k_v$):

```
Pipeline:
1. DEM → computar kh y kv (Evans-Young o Florinsky 5×5)
2. Umbralizar en cero → mapas binarios (positivo=1, negativo=0)
3. Identificar patrones lineales en mapas binarios
4. Clasificar lineamientos por tipo de falla
```

#### 7.12.2 Regla de Clasificación de Fallas

| Visible en | Solo mapa kh | Solo mapa kv | Ambos mapas |
|------------|-------------|-------------|-------------|
| **Tipo de falla** | **Strike-slip** (movimiento horizontal) | **Dip-slip / inversa / cabalgamiento** (movimiento vertical) | **Oblicua o abierta** (movimientos combinados) |

#### 7.12.3 Validación

- **Sintética**: 7 tipos de falla modelados en superficie 60m×60m. 6 de 7 coinciden con teoría.
- **Crimea**: 210km×132km, resultados correlacionan con zonas de falla conocidas.
- **Kurchatov**: Aplicado a 4 superficies (terreno + 3 horizontes geológicos). Lineamientos persisten a través de períodos geológicos.

#### 7.12.4 Estado SurtGIS — Componentes Disponibles

| Componente | Módulo SurtGIS | Estado |
|------------|---------------|--------|
| Curvatura kh, kv | `terrain/curvature.rs`, `multiscale_curvatures.rs` | Disponible |
| Reclasificación binaria | `imagery/reclassify.rs` | Disponible |
| Gradiente morfológico | `morphology/gradient.rs` | Disponible |
| Hillshade (visualización) | `terrain/hillshade.rs` | Disponible |
| **Esqueletización** | — | **Falta** (Zhang-Suen) |
| **Transformada de Hough** | — | **Falta** (detección de líneas) |
| **Strike/dip desde 3 puntos** | — | **Falta** (ajuste de plano) |

### 7.13 Capítulo 11 — Mapeo Predictivo de Suelos (completado)

#### 7.13.1 Hallazgo Principal: Selección de Predictores > Algoritmo

Florinsky enfatiza que "la selección correcta de predictores, más que el aparato matemático, juega un rol decisivo" en modelos predictivos de suelos. Esto posiciona a SurtGIS como proveedor de predictores (variables morfométricas) más que como implementador de algoritmos ML.

#### 7.13.2 Conjunto de 18 Variables de Florinsky (Gold Standard)

Variables recomendadas como predictores para mapeo de suelos:

| Categoría | Variables | Estado SurtGIS |
|-----------|----------|----------------|
| **Locales (14)** | G, A, $k_h$, $k_v$, H, K, $k_{min}$, $k_{max}$, $K_a$, E, $K_r$, $k_{he}$, $k_{ve}$, M | 8 de 14 implementadas |
| **No-locales (2)** | CA (catchment area), **DA** (dispersive area) | CA sí, **DA falta** |
| **Combinadas (2)** | TWI, SPI | Ambas implementadas |

#### 7.13.3 Variables Faltantes para Completar el Conjunto

| Variable | Símbolo | Descripción | Prioridad |
|----------|---------|-------------|-----------|
| **Dispersive area** | DA/SDA | Área que recibe flujo desde un punto (complemento de CA) | **Crítica** |
| **Unsphericity curvature** | $K_a$ | Parte del sistema Florinsky 12-curvaturas | **Alta** |
| **Laplacian** | E | Parte del sistema Florinsky 12-curvaturas | **Alta** |
| **Ring curvature** | $K_r$ | Parte del sistema Florinsky 12-curvaturas | Media |
| **Horizontal excess curv.** | $k_{he}$ | Parte del sistema Florinsky 12-curvaturas | Media |
| **Vertical excess curv.** | $k_{ve}$ | Parte del sistema Florinsky 12-curvaturas | Media |
| **Rotor** | M | Parte del sistema Florinsky 12-curvaturas | Media |
| **Eastness/Northness** | $A_E$, $A_N$ | cos(A), sin(A) — necesarias para regresión (aspecto circular) | **Alta** |

#### 7.13.4 Advertencia de Colinealidad

Nunca usar juntas en regresión:
- $k_h + k_v + H$ (H es combinación lineal de $k_h$ y $k_v$)
- CA + G + TWI (TWI se deriva de CA y G)

### 7.14 Capítulo 16 — DTA Global Esferoidal (completado)

#### 7.14.1 Pipeline Completo de Procesamiento Global

Florinsky demuestra un pipeline end-to-end para análisis morfométrico global:

```
1. Ensamblar DEM global (grilla angular esferoidal 5' = 721×361 puntos)
2. Definir parámetros del elipsoide (Krasovsky, Marte, Venus, Luna)
3. Suavizar N veces con filtro media 3×3 esferoidal
4. Computar variables locales (método esferoidal, Eqs. 4.27-4.31)
5. Computar variables no-locales (D8 adaptado, áreas de celda variables Eq. 4.43)
6. Aplicar transformada logarítmica (Eq. 8.1, n=9 para ~55 km)
7. Generar mapas binarios de CA/DA
8. Análisis visual de lineamientos globales
```

#### 7.14.2 Topología de Grilla Cerrada

La grilla global es "virtualmente cerrada":
- **Longitud**: Bordes este/oeste son adyacentes (periódico)
- **Polos**: Tratamiento especial requerido
- **Resultado**: Todos los puntos obtienen valores computados sin efectos de borde

#### 7.14.3 Resultados: 5 Estructuras Helicoidales Dobles en la Tierra

Detectadas desde mapas binarios de CA, coincidiendo con el modelo teórico de deformación torsional de Rance:

| Estructura | Brazo Izq. (km) | Brazo Der. (km) | Intersección |
|-----------|-----------------|-----------------|--------------|
| Caucasus-Clipperton | 55,800 | 31,500 | 46.4N 44.8E |
| Biscay-Santa Cruz | 39,600 | 29,800 | 44.4N 7.3W |
| Marcus | 26,500 | 24,900 | 21.4N 157.5E |
| Dakar | 17,700 | 17,200 | 14.9N 16.0W |
| Palawan | 15,400 | 15,300 | 9.9N 119.1E |

#### 7.14.4 Requisitos para Soporte Esferoidal en SurtGIS

| Requisito | Descripción | Prioridad |
|-----------|-------------|-----------|
| **Tipo de grilla esferoidal** | Primera clase en core. Definida por espaciado angular + elipsoide | **Crítica** |
| **Problema geodésico inverso** | Eqs. 4.32-4.42 para tamaños de ventana a,b,c,d,e | **Crítica** |
| **Área de celda esferoidal** | Eq. 4.43 — varía con latitud | **Crítica** |
| **Derivadas parciales esferoidales** | Eqs. 4.27-4.31 con ventana variable | **Crítica** |
| **Topología cerrada** | Wrapping en longitud, manejo de polos | **Alta** |
| **Multi-planeta** | Parámetros de elipsoide para Tierra, Marte, Venus, Luna | Media |

**Implicación arquitectónica**: Cada algoritmo que asume espaciado de grilla constante $w$ debe generalizarse para aceptar dimensiones de ventana variables $(a,b,c,d,e)$. Esto requiere fórmulas matemáticas diferentes (Eqs. 4.27-4.31 vs Evans-Young) y diferente flujo de datos.

### 7.15 Capítulo 9 — Influencia de la Topografía en Propiedades del Suelo (completado)

Capítulo teórico que establece los mecanismos físicos por los cuales la topografía controla las propiedades del suelo. Sintetiza más de un siglo de investigación desde Dokuchaev (1883) hasta la actualidad.

#### 7.15.1 Dos Mecanismos Principales

Florinsky identifica dos "herramientas" clave de la topografía:

1. **Migración lateral de agua por gravedad**: Transporte y acumulación de agua superficial e intrasuelo
2. **Diferenciación espacial del régimen térmico**: Control de insolación y evapotranspiración por pendiente y aspecto

#### 7.15.2 Variables Locales y Humedad del Suelo

| Variable | Efecto sobre humedad | Correlación típica | Mecanismo |
|----------|---------------------|-------------------|-----------|
| **G** (gradiente) | ↑G → ↓humedad | Negativa | Mayor velocidad de flujo, menor infiltración |
| **A** (aspecto) | Norte > Oeste/Este > Sur (hemisferio N) | Varía | Control de insolación y evapotranspiración |
| **$k_h$** (curv. horizontal) | $k_h<0$ → ↑humedad | Negativa | Convergencia de flujo → acumulación |
| **$k_v$** (curv. vertical) | $k_v<0$ → ↑humedad | Negativa | Desaceleración de flujo → acumulación |
| **H** (curv. media) | r ≈ 0.9 con humedad (Sinai et al., 1981) | Negativa | Control de transporte lateral intrasuelo |

**Hallazgo clave**: Sinai et al. (1981) reportaron correlación r≈0.9 entre humedad del suelo en zona de raíces y curvatura media aproximada por el Laplaciano, en condiciones áridas con topografía plana en Israel. Este resultado extremo se atribuye al control microtopográfico del transporte lateral intrasuelo (no redistribución superficial). Kuryakova et al. (1992) confirmaron dependencia fuerte de H con humedad en Moscú meridional — clima y topografía contrastantes.

#### 7.15.3 Variables No-Locales y Humedad

| Variable | Efecto | Correlación |
|----------|--------|-------------|
| **CA** (catchment area) | ↑CA → ↑humedad (Zakharov 1940: "agua por unidad de área aumenta de arriba a abajo") | Positiva |
| **TWI** | ↑TWI → ↑humedad (combina CA y G; Thompson & Moore 1996: \|r\| con TWI > \|r\| con CA o G solos) | Positiva |
| **$k_h \times CA$** | Producto empírico (Burt & Butcher 1985): correlaciones superiores a TWI y $k_h$ por separado | Positiva |

**Observación**: Ni TWI ni $k_h$ individualmente predicen bien la dinámica de humedad. Burt & Butcher (1985) propusieron $k_h \times CA$ como variable compuesta con mayor poder predictivo.

#### 7.15.4 Análisis de Superficie Dual

Florinsky señala que suelos sobre formaciones densas de arcilla o roca requieren analizar **dos superficies**:
- **Superficie del terreno**: DEM convencional
- **Superficie superior del horizonte de arcilla/roca subyacente**: Controla flujos intrasuelo a lo largo de la roca

Refs: Florinsky & Arlashina (1998), Freer et al. (2002), Chaplot & Walter (2003).

**SurtGIS**: Todas las variables locales están implementadas. Faltaría soporte para análisis dual-DEM (pipeline que compute morfometría en ambas superficies y combine resultados).

#### 7.15.5 Implicaciones para SurtGIS

| Aspecto | Estado | Acción |
|---------|--------|--------|
| Variables locales (G, A, kh, kv, H) | ✅ Implementadas | — |
| Variables no-locales (CA, TWI) | ✅ Implementadas | — |
| Variable compuesta kh×CA | ❌ Falta | Agregar como operación raster simple |
| Análisis dual-superficie | ❌ Falta | Pipeline para dos DEMs simultáneos |
| Zonas de acumulación relativa (kh<0 AND kv<0) | ❌ Falta como función dedicada | Implementar clasificación de zonas |

### 7.16 Capítulo 10 — Resolución Adecuada de Modelos (completado)

Capítulo metodológico fundamental que presenta el método REA (Representative Elementary Area) para determinar el espaciado de grilla óptimo para estudios basados en DTM.

#### 7.16.1 El Problema

La elección incorrecta del espaciado de grilla $w$ produce resultados inválidos:
- **$w$ demasiado pequeño**: Speight (1980) no encontró relación suelo-$k_h$ porque $w$ era inadecuado
- **$w$ demasiado grande**: Sinai et al. (1981) no observó correlación salinización-H por $w$ excesivo
- **Artefacto demostrado**: Con $w$ inadecuado, correlación Moist-G cambia de negativa (correcta) a positiva (artefacto)

#### 7.16.2 Método REA — Algoritmo Paso a Paso

1. Derivar un conjunto de DTMs usando una serie de valores de $w$ ($w, 2w, 3w, ..., nw$)
2. Derivar modelos de variables morfométricas desde cada DTM
3. Calcular correlaciones entre la propiedad del paisaje y cada variable morfométrica para cada $w$
4. Graficar coeficientes de correlación vs. $w$
5. Identificar **porciones suaves** del gráfico → intervalos de $w$ adecuado
6. Distinguir intervalo adecuado **principal** (común a todas las variables)
7. Realizar regresión solo con $w$ dentro del intervalo adecuado

**Recomendaciones**: $\Delta w < w_{min}/2$; tamaño de muestra ≥ 40; ignorar correlaciones no significativas.

#### 7.16.3 Caso de Estudio: Severny Gully (Rusia)

| Parámetro | Valor |
|-----------|-------|
| **Sitio** | Barranco Severny, sur de Moscú, bosque subboreal |
| **DEM irregular** | 374 puntos, levantamiento taquimétrico |
| **DEMs regulares** | 13 grillas: $w$ = 1, 1.5, 2, ..., 7 m |
| **Propiedad** | Humedad superficial (62 muestras en contorno de 4.25 m) |
| **Variables** | G, $k_h$, $k_v$, H (no CA/TWI por falta de cabecera de cuenca) |

**Resultados clave**:

| Resultado | Valor |
|-----------|-------|
| Intervalo adecuado W1 | $w \approx 2.25 - 4$ m (para H, $k_v$, $k_h$) |
| Intervalo adecuado W2 (principal) | $w \approx 2.5 - 3$ m (para todas las variables) |
| Correlaciones en W2 ($w$=3 m) | G: -0.28, $k_h$: -0.52, $k_v$: -0.50, H: **-0.60** |
| Regresión ($w$=2.5 m) | Moist = f(G, H), **R² = 0.45** |
| **Área adecuada** | **20–40 m²** (equivalente: $4w^2$) |

**Implicación**: Para regiones forestales subborales, $w$ = 3–4 m es adecuado para estudios de suelo/hidrología. Esto implica que el control topográfico opera sobre geoformas con área típica de 36–64 m².

#### 7.16.4 Oportunidad para SurtGIS: Función de Diagnóstico REA

Ningún competidor ofrece una función automatizada de análisis REA. SurtGIS podría implementar:

```
fn rea_analysis(dem_irregular, property_samples, w_range, delta_w)
  -> REAResult { adequate_intervals, correlations, recommended_w }
```

| Requisito | Descripción | Complejidad |
|-----------|-------------|-------------|
| Multi-resolución | Generar DTMs con serie de $w$ desde DEM original | Media |
| Correlación iterativa | Computar correlaciones morfometría-propiedad por cada $w$ | Baja |
| Detección de intervalos suaves | Identificar porciones suaves del gráfico r vs. $w$ | Media |
| Reporte/gráfico | Salida estructurada con recomendación | Baja |

### 7.17 Capítulo 12 — Análisis de Relaciones Topografía-Suelo (completado)

Capítulo empírico central — el más detallado del libro (1069 líneas). Presenta resultados de campo con >5000 muestras de suelo en dos sitios de Manitoba, Canadá.

#### 7.17.1 Diseño Experimental

| Parámetro | Miniota | Minnedosa |
|-----------|---------|-----------|
| **Tamaño** | 809×820 m (∆z ≈ 6 m) | 1680×820 m (∆z ≈ 13 m) |
| **Plot** | 450×150 m | 500×200 m |
| **Puntos de muestreo** | 210 (10 transectos × 21 pts) | 40 (4 transectos × 10 pts) |
| **Profundidades** | 0–0.3, 0.3–0.6, 0.6–0.9, 0.9–1.2 m | ~10 cm |
| **Temporadas** | 6 (May, Jul, Aug × 1997, 1998) | 2 (Jul 2000, Jul 2001) |
| **Propiedades** | Humedad gravimétrica (%) | Humedad, densidad, 6 índices microbianos |
| **DEM** | GNSS (4211 pts), w=15 m | GNSS (7193 pts), w=20 m |
| **Método** | Evans-Young + transformada log (n=4) | Evans-Young + transformada log (n=4) |
| **Variables morfométricas** | 18 (recomendación del Cap. 11) | 18 |

#### 7.17.2 Hallazgo 1: Variables que NUNCA Son Significativas

De las 18 variables morfométricas recomendadas, **7 nunca mostraron correlación significativa** con ninguna propiedad del suelo en ninguna temporada ni profundidad:

| Variable | Símbolo | Clasificación Florinsky |
|----------|---------|------------------------|
| Horizontal excess curvature | $k_{he}$ | Exceso |
| Vertical excess curvature | $k_{ve}$ | Exceso |
| Laplacian | E | Total |
| Rotor | M | Total |
| Ring curvature | $K_r$ | Combinada |
| Unsphericity curvature | $K_a$ | Combinada |
| Gaussian curvature | K | Combinada |

**Implicación para SurtGIS**: Priorizar implementación de las ~10 variables que sí son predictivas. Las 7 "nunca significativas" son de interés teórico (sistema completo de 12 curvaturas) pero no empíricamente útiles para mapeo predictivo de suelos.

#### 7.17.3 Hallazgo 2: Variables Más Predictivas

Ranking empírico de poder predictivo (basado en correlaciones de Tabla 12.6 y regresiones de Tablas 12.8-12.10):

| Ranking | Variable | Correlación típica (0–0.3 m) | Frecuencia en regresiones |
|---------|----------|------------------------------|--------------------------|
| 1 | **$k_v$** (curv. vertical) | r = -0.37 a **-0.65** | Muy alta |
| 2 | **z** (elevación) | r = -0.29 a -0.51 | Alta |
| 3 | **H** (curv. media) | r = -0.23 a -0.48 | Alta |
| 4 | **TWI** | r = 0.30 a 0.53 | Alta |
| 5 | **SCA** (catchment area esp.) | r = 0.25 a 0.46 | Media-Alta |
| 6 | **G** (gradiente) | r = -0.25 a -0.41 | Media |
| 7 | **$k_{max}$** (curv. máxima) | r = -0.19 a -0.46 | Media |
| 8 | **$k_{min}$** (curv. mínima) | r = -0.21 a -0.35 | Baja-Media |

#### 7.17.4 Hallazgo 3: Variabilidad Temporal

Las relaciones topografía-suelo son **temporalmente inestables**:
- Diferentes ecuaciones de regresión para diferentes estaciones
- Distintos conjuntos de predictores seleccionados por stepwise regression
- R² varía significativamente entre temporadas (0.40–0.49)
- Solo 2–7 variables seleccionadas de 18 candidatas

**Causa**: Las propiedades del suelo son resultado integral de procesos con diferentes escalas temporales. La topografía es temporalmente estable, pero los factores co-actuantes (vegetación, precipitación) varían.

#### 7.17.5 Hallazgo 4: Variabilidad con la Profundidad

- **Capa efectiva**: 0–0.3 m para propiedades dinámicas (humedad)
- Correlaciones **disminuyen monotónicamente** con la profundidad
- A 0.9–1.2 m, solo z mantiene correlación significativa consistente
- Causa: variabilidad espacial de la disminución de conductividad hidráulica con la profundidad

#### 7.17.6 Hallazgo 5: Umbral de Humedad

En Minnedosa (Tabla 12.7):
- **2000** (año húmedo, Moist media = 33.4%): Correlaciones significativas con múltiples variables ($k_v$: -0.60, z: -0.51, TWI: 0.53)
- **2001** (año seco, Moist media = 23.2%): Solo z y $k_v$ retienen significancia; todas las demás correlaciones desaparecen

**Interpretación**: Existe un **umbral de humedad** (~25–30%) por debajo del cual el control topográfico de las propiedades del suelo se colapsa. En condiciones secas, la variabilidad aleatoria domina sobre los procesos gravitatorios.

#### 7.17.7 Valores de R² Típicos

| Modelo | R² | Variables | Contexto |
|--------|-----|-----------|----------|
| Moist 0–0.3 m, May 1998, Miniota | **0.49** | z, G, $A_E$, $k_v$, $k_{min}$, lnSCA | Mejor modelo |
| Moist 0–0.3 m, Jul 1997, Miniota | 0.45 | z, G, $A_E$, $A_N$, $k_v$, $k_{min}$, lnSCA | 7 predictores |
| Moist 0–0.3 m, May 1997, Miniota | 0.40 | z, G, $A_E$, H | 4 predictores |
| Bulk density 2000, Minnedosa | 0.48 | z, $k_v$ | 2 predictores |

**Nota**: R² > 0.70 es inusual; R² ≈ 0.50 es típico para modelos espaciales cuantitativos de suelo (Beckett & Webster 1971; Malone et al. 2009).

#### 7.17.8 Implicaciones para SurtGIS

1. **Priorizar las 8-10 variables empíricamente validadas** sobre completar el sistema de 12 curvaturas
2. **R² = 0.45-0.49 es el techo realista** para predicción basada en morfometría sola
3. **Validación temporal**: Los modelos deben recalibrarse estacionalmente
4. **Profundidad de muestreo**: 0–0.3 m para propiedades dinámicas
5. **Condiciones húmedas**: La topografía es más predictiva cuando hay suficiente agua para que la gravedad actúe

### 7.18 Capítulo 15 — Zonas de Acumulación e Intersecciones de Fallas (completado)

#### 7.18.1 Concepto Central

La clasificación de zonas de acumulación/disipación/tránsito se basa en signos de $k_h$ y $k_v$:

| Zona | Condición | Significado físico | Equivalente geológico |
|------|----------|-------------------|----------------------|
| **Acumulación** | $k_h < 0$ AND $k_v < 0$ | Convergencia + desaceleración | Intersecciones de fallas |
| **Disipación** | $k_h > 0$ AND $k_v > 0$ | Divergencia + aceleración | Bloques entre fallas |
| **Tránsito** | Otros | Sin acción concurrente unidireccional | Segmentos de falla fuera de intersecciones |

Este es el algoritmo de clasificación más simple de Florinsky: solo requiere signos de dos curvaturas ya computadas.

#### 7.18.2 Validación: Caso de Estudio en Crimea

Parámetros del estudio:

| Parámetro | Valor |
|-----------|-------|
| **Área** | Península de Crimea, 26,703 km² |
| **DEM** | Mismo que Cap. 14 (análisis de lineamientos) |
| **Fenómenos validados** | Manantiales/pozos con descarga anormalmente alta; fracturamiento intenso; fracturamiento muy intenso |
| **Método de validación** | Coeficiente de asociación $R^{as}_i = W_i \cdot P_i$ |

#### 7.18.3 Coeficiente de Asociación $R^{as}$

Fórmula:

$$R^{as}_i = W_i \cdot P_i$$

donde:
- $W_i = \frac{4}{3} - E_i$ (peso de la zona, con $E_i = S_i/S_S$ = proporción de área)
- $P_i = U_i/U_S$ (proporción del fenómeno en la zona)
- Normalizado: $R^{as}_a + R^{as}_t + R^{as}_d = 1$

#### 7.18.4 Resultados

| Fenómeno | $R^{as}_a$ (acumulación) | $R^{as}_t$ (tránsito) | $R^{as}_d$ (disipación) |
|----------|--------------------------|------------------------|--------------------------|
| Manantiales/pozos alta descarga | **0.74** | 0.23 | 0.03 |
| Fracturamiento muy intenso | **0.74** | 0.15 | 0.11 |
| Fracturamiento intenso | **0.54** | 0.31 | 0.15 |

Las zonas de acumulación topográfica están fuertemente asociadas con intersecciones de fallas ($R^{as}_a$ = 0.74 para descarga anómala y fracturamiento extremo).

#### 7.18.5 Implementación para SurtGIS

| Función | Descripción | Complejidad |
|---------|-------------|-------------|
| `classify_accumulation_zones(kh, kv)` | Clasificación trinaria basada en signos | **Trivial** |
| `association_coefficient(zones, phenomena)` | Cálculo de $R^{as}$ para validación | Baja |

Ambas son funciones simples que operan sobre las curvaturas ya computadas. Valor: análisis geológico estructural sin costo computacional adicional.

### 7.19 Capítulo 17 — Movimiento y Evolución Glaciar (completado)

Capítulo introductorio breve (159 líneas) que revisa tres aspectos de la geomorfometría glaciar.

#### 7.19.1 Tres Temas Cubiertos

| Tema | Método | Datos requeridos |
|------|--------|-----------------|
| **Balance de masa** | Resta DEM₁ - DEM₂ (series temporales) | DEMs multitemporales |
| **Velocidad de flujo** | COSI-Corr o pseudo-paralaje sobre ortoimágenes/hillshades multitemporales | Ortomosaicos + DEMs |
| **Balance de fuerzas** | Cambios en G y curvaturas de superficie glaciar | DEMs multitemporales |

#### 7.19.2 Hallazgos Relevantes para SurtGIS

- **DEM differencing**: Método establecido desde Etzelmüller & Sollid (1997). SurtGIS ya soporta operaciones raster básicas.
- **COSI-Corr**: Correlación cruzada sub-pixel en dominio de Fourier. Implementación compleja — mejor dejar a software especializado.
- **Curvaturas en glaciología**: Florinsky nota que las curvaturas están **infrautilizadas** en estudios glaciares. $k_h$ puede estimar velocidad de emergencia; cambios de signo de curvaturas indican cambios en dinámica de flujo.
- **Oportunidad**: SurtGIS ya computa las variables necesarias (G, $k_h$, $k_v$, H). El valor está en pipelines de análisis temporal (comparar morfometría entre épocas).

### 7.20 Capítulo 18 — Detección de Grietas (Crevasses) (completado)

Caso de estudio aplicado sobre detección de grietas ocultas en glaciares mediante geomorfometría de DEMs derivados de UAS.

#### 7.20.1 Contexto

- **Grietas ocultas**: Cubiertas por puentes de nieve, extremadamente peligrosas (1–5 m de ancho)
- **DEMs convencionales**: Resolución insuficiente para detectar grietas de 1–5 m
- **Solución**: DEMs de UAS con resolución centimétrica

#### 7.20.2 Diseño Experimental

| Parámetro | Valor |
|-----------|-------|
| **Ubicación** | Ruta de trineos Progress→Vostok, Antártida Oriental |
| **UAS** | Geoscan 201 Geodesy (ala volante, 2.3 m envergadura) |
| **GSD** | 6 cm |
| **Área de prueba** | 1300 × 770 m |
| **DEMs generados** | 3 resoluciones: 25 cm, 50 cm, 1 m |
| **Variables computadas** | 16 variables morfométricas |
| **Método** | Evans-Young (Sección 4.1) + transformada log (Eq. 8.1) |

#### 7.20.3 Resultados Clave

| Resultado | Detalle |
|-----------|---------|
| **Mejor resolución** | **1 m** (no 25-50 cm — las resoluciones más finas son ruidosas) |
| **Mejores variables** | **$k_h$** y **$k_{min}$** para detección de lineamientos |
| **Relación señal/ruido** | Factor crítico: DEMs de 25 cm demasiado ruidosos para derivadas |
| **Ratio GSD:DEM** | 1:4 recomendado (GSD 6 cm → DEM óptimo ~25 cm, pero 1 m mejor para morfometría) |

**Convergencia con Cap. 14**: $k_h$ es la variable clave tanto para lineamientos geológicos (Cap. 14) como para grietas glaciares (Cap. 18). Esto valida $k_h$ como detector universal de lineamientos.

#### 7.20.4 Implicaciones para SurtGIS

- Las variables necesarias ($k_h$, $k_{min}$) ya están implementadas
- La transformada logarítmica (Eq. 8.1, Cap. 8) es esencial para visualización de resultados
- **Recomendación**: Implementar pipeline de detección de lineamientos que combine $k_h$, mapas binarios y esqueletización
- El control de resolución/ruido es más importante que la cantidad de variables

### 7.21 Capítulo 19 — Eventos Catastróficos Glaciares (completado)

Caso de estudio sobre la subsidencia catastrófica de 2017 en el Glaciar Dålk, Antártida Oriental.

#### 7.21.1 El Evento

- **Fecha**: 30 de enero de 2017
- **Ubicación**: Margen noroeste del Glaciar Dålk (69°23'58"S, 76°24'49"E)
- **Efecto**: Formación abrupta de depresión amplia y profunda que destruyó sección de ruta de trineos

#### 7.21.2 Metodología

Puramente basada en DEM differencing y morfometría estándar:

1. **DEMs pre/post evento**: Generados 10 días antes y 10 días después (Geoscan 201, w=22 cm)
2. **Resta de DEMs**: Pre-evento restado de post-evento → volumen de caverna pre-colapso
3. **Red de canales**: DEM pre-evento → corrección hidrológica (fill sinks) → CA → extracción de canales
4. **Resultado**: Red de canales de fusión superficial convergían hacia zona de subsidencia

#### 7.21.3 Implicaciones para SurtGIS

**SurtGIS ya puede soportar completamente este tipo de análisis** con algoritmos existentes:
- Operaciones raster (resta de DEMs)
- Fill sinks (depresión filling)
- Flow routing (CA)
- Extracción de redes de drenaje

No se requieren algoritmos nuevos. El valor está en la integración del pipeline.

### 7.22 Capítulo 20 — Atlas Geomorfométrico Antártico (completado)

Caso de estudio a gran escala: primer atlas geomorfométrico de las Larsemann Hills, un oasis antártico.

#### 7.22.1 Parámetros del Atlas

| Parámetro | Valor |
|-----------|-------|
| **DEM fuente** | REMA v1.1 (Reference Elevation Model of Antarctica) |
| **Área** | 18,168 × 14,968 m (Larsemann Hills + glaciares adyacentes) |
| **Grid spacing** | 8 m |
| **Puntos** | 4,253,184 (2272 × 1872) |
| **Variables computadas** | **17**: 14 locales (G, A, $k_h$, $k_v$, E, $k_{he}$, $k_{ve}$, $K_a$, $K_r$, $k_{min}$, $k_{max}$, H, K, M) + CA + TWI + SPI |
| **Método** | Evans-Young (SAGA 7.8.2) |
| **Proyección** | Polar stereográfica → UTM zona 43S |
| **Suavizado** | Ninguno (REMA suficientemente suave) |
| **Validación** | 54 rutas de campo, 6–10 km cada una |

#### 7.22.2 Pipeline de Procesamiento

```
1. Extraer fragmento REMA (polar stereográfica, WGS84)
2. Reproyectar a UTM zona 43S (8 m)
3. Editar DEM (remover icebergs-artefacto, enmascarar lagos)
4. Computar 6 variables directas con Evans-Young: G, A, kh, kv, kmin, kmax
5. Computar 8 variables derivadas con álgebra raster: E, khe, kve, Ka, Kr, H, K, M
6. Computar CA (total catchment area)
7. Derivar TWI y SPI desde G y CA
8. Producir mapas (17 capas)
```

#### 7.22.3 Observaciones Relevantes

1. **Sin suavizado**: El REMA a 8 m no requiere filtrado previo — nivel de ruido bajo y suficiente suavidad
2. **Evans-Young suficiente**: Para 8 m de resolución, Evans-Young (3×3) es adecuado
3. **Variables derivadas por álgebra raster**: 8 de 14 variables locales se calculan como combinaciones algebraicas de las 6 primarias — SurtGIS debería exponer esto como operaciones raster simples
4. **Reproyección necesaria**: El procesamiento se hizo en UTM, no en polar stereográfica. El soporte de reproyección es importante.
5. **Enmascarado de lagos**: Los lagos en REMA tienen valores de elevación interpolados (artefactos). Necesidad de máscaras externas.

#### 7.22.4 Validación de Campo

- 54 rutas a pie (6–10 km, 5–8 h cada una)
- Distancia entre rutas: 300–500 m (10× la distancia mínima teórica de 30–50 m)
- Distancia mínima teórica = 4–6 × w = 32–48 m (consecuencia del teorema de muestreo)
- Interpretación geomorfológica in situ de los mapas morfométricos

#### 7.22.5 Implicaciones para SurtGIS

| Aspecto | Estado | Prioridad |
|---------|--------|-----------|
| Evans-Young 3×3 | ✅ Implementado | — |
| Variables derivadas (H, K, E, etc.) | ⚠️ Parcial (falta E, Ka, Kr, khe, kve, M) | Alta |
| CA, TWI, SPI | ✅ Implementados | — |
| Álgebra raster | ❌ Falta como API pública | Media |
| Enmascarado de capas | ❌ Falta | Media |
| Pipeline batch (múltiples variables) | ❌ Falta | Alta |

---

## Registro de actualizaciones

| Fecha | Sección | Acción |
|---|---|---|
| 2026-01-31 | 1. WebAssembly + Rust | Revisión inicial — 22 refs, 5 relevantes |
| 2026-01-31 | 2. Terrain Analysis | Revisión completa — 586 refs (v1–v4), 23 alta relevancia, 5 papers fundacionales (>100 citas) |
| 2026-01-31 | 3. Hydrology | Revisión completa — 295 refs (v1–v2), 32 alta relevancia. Hallazgo: MFD >> D8 para TWI |
| 2026-01-31 | 6. Análisis Competitivo | Matriz comparativa SurtGIS vs WBT/SAGA/GRASS. 20 algoritmos priorizados. Identificados 7 que nadie implementa y 12 WTE de pago para open-source |
| 2026-01-31 | 7. Florinsky (2025) | **Parte I completa (8 caps)**. Cap. 2: 57 variables, 12 curvaturas, auditoría Z&T. Cap. 4: hallazgo crítico — multiscale_curvatures.rs incorrecto (cuadrático vs cúbico), fórmulas 5×5 cerradas. Cap. 6: 11 métodos de filtrado, FPDEMS incorrecto, falta Gaussiano. Cap. 7: Método espectral analítico Chebyshev — ningún competidor lo implementa. Cap. 1: ML dominante, UAS/SfM revolución, hidrología incompleta como gap crítico. Cap. 3: grilla esferoidal obligatoria para DEMs globales, thin-plate spline faltante. Cap. 5: fórmulas RMSE completas (Eqs. 5.2-5.12), Florinsky 5×5 ~6× mejor que Evans-Young para curvaturas, isotropía probada. Cap. 8: transformada logarítmica (Eq. 8.1) — diferenciador único. 6 TeX generados |
| 2026-01-31 | 7. Florinsky (2025) | **Caps. 21, 11, 13, 14, 16 completados** (Partes II-IV). Cap. 21: 10 problemas pendientes — flow routing "obsoleto", método espectral analítico, cálculo vectorial. Cap. 13: Theorema Egregium (K invariante bajo doblado), shape index y curvedness faltantes. Cap. 14: detección de lineamientos con mapas binarios kh/kv, clasificación de fallas, falta esqueletización. Cap. 11: conjunto de 18 variables Florinsky como gold standard, DA/SDA y 6 curvaturas faltantes. Cap. 16: pipeline esferoidal completo, topología cerrada, 5 estructuras helicoidales en la Tierra. Total: 13 de 21 capítulos revisados |
| 2026-01-31 | 4. Remote Sensing | Revisión completa — 66 refs (v1), 31 alta relevancia, 13 fundacionales. 93% índices espectrales. Hallazgo: n-band index builder como diferenciador. 7 índices faltantes (NDRE, RECI, GNDVI, NGRDI, MTVI, CWSI, MSI). Wang et al. 2019: 3 bandas > 2 bandas |
| 2026-01-31 | 5. Interpolation | Revisión completa — 200 refs, 188 parseadas, 27 fundacionales. Kriging domina (79%). Gap crítico: SurtGIS no tiene kriging (OK, RK, UK). RK 15-30% mejor que IDW en montaña. RF/GB 10-25% mejor que kriging con datos densos. TPS confirmado como faltante (convergencia con Florinsky Ch.3). Variograma como prerequisito |
| 2026-01-31 | 7. Florinsky (2025) | **Libro completo — 21/21 capítulos revisados**. Caps. 9, 10, 12, 15, 17, 18, 19, 20 completados. Cap. 9: mecanismos físicos topografía→suelo, H r≈0.9, kh×CA como compuesto. Cap. 10: método REA para grid spacing adecuado, w=2.5-3.0 m para subboreal, área=4w². Cap. 12: **capítulo empírico central** — 5000+ muestras, 7/19 variables NUNCA significativas (khe,kve,E,M,Kr,Ka,K), kv predictor más fuerte (r=0.60-0.65), umbral humedad ~25-30%, R²=0.45-0.49 típico. Cap. 15: zonas acumulación kh<0∧kv<0 → intersecciones de fallas, Ras=0.74. Cap. 17-19: glaciología — DEM differencing, COSI-Corr, kh+kmin para grietas, 1m grid óptimo. Cap. 20: atlas antártico 17 variables, 8m grid, REMA, Evans-Young, sin suavizado necesario |
| 2026-01-31 | 5b. Solar/Viewshed | Revisión completa — 135 refs (v1+v2), 15 fundacionales. **solar_v1.bib mislabeled**: 100% viewshed (80 papers). solar_v2.bib: 55 papers solar real. Viewshed: XDraw (12 papers) y PDERL como mejoras prioritarias sobre Bresenham actual, GPU 37.5%. Solar: shadow casting ausente (10-40% error en montaña), HORAYZON 10-100× más rápido, Klucher/Perez difusa anisotrópica faltante, SVF existente no integrado. Corripio 2003 (161 citas) fundacional. Roadmap 4 fases |
