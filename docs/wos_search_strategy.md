# Estrategia de Busqueda Sistematica en Web of Science para SurtGIS

## Objetivo

Identificar gaps, necesidades y tendencias en software geoespacial para posicionar SurtGIS estrategicamente. Cada query esta disenada para responder una pregunta especifica de desarrollo.

**Instrucciones:** Ejecuta cada query en WoS (Advanced Search), exporta como BibTeX (.bib), y nombra el archivo segun el ID de la query (Q1.bib, Q2.bib, etc.).

**Periodo:** 2020-2026 (salvo que se indique otro)
**Base:** Web of Science Core Collection

---

## Q1. Estado del arte en software GIS open-source

**Pregunta:** Que limitaciones reportan los investigadores en las herramientas GIS actuales (QGIS, GRASS, GDAL, WhiteboxTools)?

```
TS=("geospatial software" OR "GIS software" OR "geospatial tool*") AND TS=("open source" OR "open-source" OR "FOSS") AND TS=("limitation*" OR "challenge*" OR "comparison" OR "benchmark" OR "performance" OR "evaluation")
```

**Refinamiento:** Document Types = Article OR Review. Periodo: 2020-2026.
**Resultado esperado:** ~50-150 papers. Nos dice que falta en el ecosistema actual.

---

## Q2. WebGIS y analisis geoespacial en el navegador

**Pregunta:** Quien esta haciendo analisis raster en el browser? Que tecnologias usan? Que limitaciones tienen?

```
TS=("web GIS" OR "WebGIS" OR "web-based GIS" OR "browser-based" OR "client-side") AND TS=("raster analysis" OR "terrain analysis" OR "geospatial processing" OR "spatial analysis" OR "WebAssembly" OR "WASM")
```

**Resultado esperado:** ~30-80 papers. Nos posiciona vs la competencia web.

---

## Q3. Cloud-native geospatial y STAC

**Pregunta:** Como se esta usando STAC y COG en investigacion? Que workflows son mas comunes?

```
TS=("STAC" OR "SpatioTemporal Asset Catalog" OR "Cloud Optimized GeoTIFF" OR "COG") AND TS=("remote sensing" OR "earth observation" OR "satellite" OR "geospatial")
```

**Resultado esperado:** ~20-60 papers. Valida nuestra inversion en STAC browser.

---

## Q4. Analisis de terreno: workflows y necesidades

**Pregunta:** Que algoritmos de terreno usan mas los investigadores? Que pipelines son mas solicitados?

```
TS=("digital elevation model" OR "DEM" OR "terrain analysis") AND TS=("slope" OR "aspect" OR "curvature" OR "flow accumulation" OR "hillshade" OR "geomorphometry" OR "topographic") AND TS=("workflow" OR "pipeline" OR "processing chain" OR "automated" OR "reproducib*")
```

**Resultado esperado:** ~80-200 papers. Define prioridades de algoritmos.

---

## Q5. Hidrologia computacional: herramientas y gaps

**Pregunta:** Que necesitan los hidrologos que las herramientas actuales no proveen?

```
TS=("hydrological model*" OR "watershed analysis" OR "catchment" OR "drainage") AND TS=("DEM" OR "digital elevation") AND TS=("software" OR "tool" OR "algorithm" OR "GIS" OR "open source") AND TS=("limitation*" OR "challenge*" OR "improvement" OR "novel" OR "new approach")
```

**Resultado esperado:** ~60-150 papers. Guia el modulo de hidrologia.

---

## Q6. Indices espectrales y teledeteccion

**Pregunta:** Que indices espectrales son mas utilizados y cuales estan emergiendo?

```
TS=("spectral ind*" OR "vegetation index" OR "NDVI" OR "remote sensing index") AND TS=("Sentinel-2" OR "Landsat" OR "multispectral") AND TS=("new" OR "novel" OR "improved" OR "comparison" OR "evaluation" OR "review")
```

**Periodo:** 2021-2026 (mas reciente)
**Resultado esperado:** ~100-300 papers. Valida los 17 indices que ya tenemos y sugiere nuevos.

---

## Q7. Interpolacion espacial: metodos y aplicaciones

**Pregunta:** Que metodos de interpolacion se usan mas? Kriging vs IDW vs machine learning?

```
TS=("spatial interpolation" OR "geostatistic*" OR "kriging" OR "IDW" OR "inverse distance") AND TS=("comparison" OR "evaluation" OR "accuracy" OR "performance" OR "machine learning" OR "deep learning")
```

**Periodo:** 2020-2026
**Resultado esperado:** ~50-120 papers. Guia el modulo de interpolacion.

---

## Q8. Metricas de paisaje y ecologia

**Pregunta:** Que metricas de paisaje se usan en conservacion y planificacion?

```
TS=("landscape metric*" OR "landscape ecology" OR "patch" OR "fragmentation" OR "connectivity") AND TS=("software" OR "tool" OR "FRAGSTATS" OR "GIS" OR "spatial analysis") AND TS=("conservation" OR "biodiversity" OR "land use" OR "planning")
```

**Resultado esperado:** ~40-100 papers. Valida nuestro modulo landscape.

---

## Q9. Clasificacion de imagenes satelitales

**Pregunta:** Que metodos de clasificacion se usan mas? Supervised vs unsupervised vs deep learning?

```
TS=("image classification" OR "land cover classification" OR "land use classification") AND TS=("satellite" OR "remote sensing" OR "Sentinel" OR "Landsat") AND TS=("k-means" OR "random forest" OR "deep learning" OR "convolutional" OR "unsupervised" OR "supervised" OR "comparison")
```

**Periodo:** 2022-2026 (muy reciente)
**Resultado esperado:** ~200-500 papers. Guia clasificacion: es deep learning? random forest? mixto?

---

## Q10. Analisis de cambio y deteccion de cambios

**Pregunta:** Que metodos de change detection son mas usados? Que datos?

```
TS=("change detection" OR "land cover change" OR "deforestation" OR "urban expansion" OR "change analysis") AND TS=("satellite" OR "remote sensing" OR "multitemporal" OR "time series") AND TS=("method*" OR "algorithm" OR "approach" OR "technique")
```

**Periodo:** 2022-2026
**Resultado esperado:** ~150-400 papers. SurtGIS ya tiene diferencia de bandas, pero hay mas?

---

## Q11. Rust y lenguajes de alto rendimiento en geoespacial

**Pregunta:** Se esta usando Rust u otros lenguajes de alto rendimiento en GIS? Que resultados reportan?

```
TS=("Rust programming" OR "Rust language" OR "WebAssembly" OR "WASM" OR "high performance") AND TS=("geospatial" OR "GIS" OR "remote sensing" OR "raster" OR "terrain" OR "spatial analysis")
```

**Resultado esperado:** ~10-30 papers. Pocos — valida que SurtGIS es pionero.

---

## Q12. Reproducibilidad en ciencias de la tierra

**Pregunta:** Que barreras hay para investigacion geoespacial reproducible?

```
TS=("reproducib*" OR "replicab*" OR "FAIR data") AND TS=("geospatial" OR "GIS" OR "remote sensing" OR "earth science" OR "environmental") AND TS=("software" OR "workflow" OR "pipeline" OR "tool" OR "platform")
```

**Resultado esperado:** ~30-80 papers. SurtGIS + workspace save/load + demo web abordan esto directamente.

---

## Q13. Analisis geoespacial en educacion

**Pregunta:** Como se ensena GIS? Que herramientas usan? Que barreras hay?

```
TS=("GIS education" OR "geospatial education" OR "teaching GIS" OR "GIS curriculum") AND TS=("tool*" OR "software" OR "web-based" OR "online" OR "interactive" OR "hands-on")
```

**Resultado esperado:** ~20-60 papers. La demo web de SurtGIS es un argumento de educacion.

---

## Q14. Radiacion solar y modelamiento energetico

**Pregunta:** Que metodos de radiacion solar sobre terreno se usan? (SurtGIS ya tiene solar_radiation)

```
TS=("solar radiation" OR "solar irradiance" OR "solar energy") AND TS=("DEM" OR "terrain" OR "topograph*" OR "shadow" OR "hillshade") AND TS=("model*" OR "algorithm" OR "GIS" OR "simulation")
```

**Resultado esperado:** ~40-100 papers. Guia mejoras al modulo solar.

---

## Q15. Analisis de riesgos naturales y desastres

**Pregunta:** Que herramientas geoespaciales se usan en gestion de riesgos? Que falta?

```
TS=("natural hazard*" OR "disaster" OR "landslide" OR "flood" OR "wildfire") AND TS=("GIS" OR "geospatial" OR "remote sensing" OR "DEM") AND TS=("susceptibility" OR "risk assessment" OR "vulnerability" OR "mapping" OR "model*")
```

**Resultado esperado:** ~200-500 papers. SurtGIS tiene pipeline de susceptibilidad — esto lo valida.

---

## Resumen de Queries

| ID | Tema | Query corta | Papers est. |
|----|------|-------------|-------------|
| Q1 | Software GIS open-source | limitaciones, comparacion | 50-150 |
| Q2 | WebGIS / browser analysis | web-based raster, WASM | 30-80 |
| Q3 | STAC / COG cloud-native | workflows satelitales | 20-60 |
| Q4 | Terrain analysis workflows | DEM pipelines | 80-200 |
| Q5 | Hidrologia computacional | gaps y herramientas | 60-150 |
| Q6 | Indices espectrales | nuevos indices, comparacion | 100-300 |
| Q7 | Interpolacion espacial | kriging vs IDW vs ML | 50-120 |
| Q8 | Metricas de paisaje | FRAGSTATS, conservacion | 40-100 |
| Q9 | Clasificacion de imagenes | DL vs ML vs classic | 200-500 |
| Q10 | Change detection | metodos multitemporales | 150-400 |
| Q11 | Rust/WASM en geoespacial | alto rendimiento | 10-30 |
| Q12 | Reproducibilidad | FAIR, workflows | 30-80 |
| Q13 | Educacion GIS | herramientas, barreras | 20-60 |
| Q14 | Radiacion solar | DEM + solar models | 40-100 |
| Q15 | Riesgos naturales | susceptibilidad, mapping | 200-500 |

**Total estimado:** ~1000-2800 papers (con overlap entre queries)

---

## Instrucciones para descargar de WoS

1. Ir a **Web of Science** → Advanced Search
2. Copiar la query exacta (la que esta entre comillas de codigo)
3. Seleccionar **Timespan**: 2020-2026 (o lo indicado por query)
4. Seleccionar **Document Types**: Article, Review
5. Ejecutar busqueda
6. Click **Export** → **BibTeX** → **Full Record**
7. Guardar como `Q1.bib`, `Q2.bib`, etc. en `docs/wos/`
8. Repetir para cada query

## Que hare con los .bib

1. **Analisis bibliometrico**: keywords mas frecuentes, clusters tematicos
2. **Gap analysis**: que se menciona como limitacion/challenge en cada area
3. **Trend detection**: que metodos estan creciendo en citaciones
4. **Feature prioritization**: mapear gaps detectados a features de SurtGIS
5. **Competitor landscape**: que herramientas se mencionan mas
6. **Roadmap data-driven**: priorizar v0.5.0+ basado en evidencia cientifica

---

## Prioridad de ejecucion

**Imprescindibles** (ejecutar primero):
- Q1 (software GIS) — posicionamiento general
- Q4 (terrain) — core de SurtGIS
- Q6 (indices) — validar modulo imagery
- Q11 (Rust/WASM) — argumentar novedad

**Importantes**:
- Q2 (WebGIS) — validar demo web
- Q3 (STAC) — validar STAC browser
- Q5 (hidrologia) — guiar modulo hydrology
- Q15 (riesgos) — casos de uso aplicados

**Complementarias**:
- Q7-Q14 — modulos especificos
