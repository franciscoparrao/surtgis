# SurtGIS — Oportunidades de desarrollo (noviembre 2026)

**Fecha**: 2026-11-04
**Base evaluada**: SurtGIS v1.6.0 (main @ b3ca6f7).
**Método**: cuatro revisiones independientes en paralelo, centradas en lo que cambió desde el análisis del 2026-09-28 (`sota_vanguardia_2026-09.md`):

1. Bibliometría en OpenAlex: 18 temas, artículos de revista 2021–2026, crecimiento relativo a una base.
2. Ecosistema de herramientas y estándares (releases, changelogs, issues).
3. GeoAI: embeddings, inferencia y agentes.
4. Métodos de hazards y geomorfometría con demanda en Chile.

Los reportes íntegros van en los anexos A–D. Cuatro afirmaciones decisivas se verificaron además en la fuente primaria al sintetizar: el incidente de Planetary Computer (issue #516), la cita del manual Grfin del USGS, el correo de Even Rouault en gdal-dev y los campos que hoy registra la provenance de SurtGIS.

**Leyenda de evidencia**: *verificado* = comprobado en la fuente primaria o en el código; *inferido* = juicio razonado, con grado de confianza; *supuesto* = no comprobado.

---

## 1. Conclusión

SurtGIS no está estancado por falta de ideas, sino porque desde septiembre todo el tiempo se fue en responder reportes. Desde el 2026-09-28 no entró ningún commit `feat`; entraron 5 `fix`, 5 `docs` y 4 releases (verificado, `git log`). Los arreglos fueron valiosos, pero ninguno movió los frentes del análisis anterior.

La revisión encuentra tres apuestas con demanda verificada, ventaja real de SurtGIS y nadie que las cubra:

1. **Alcance de flujos de detritos de valle confinado a cono aluvial** (tipo Grfin/LAHARZ corregido). Resuelve la falla que Tal_Tal reportó en Taltal (Jaccard ≤ 0,17). El USGS describe exactamente esa falla en su manual de 2025 y su solución corre solo en Windows con TauDEM. No existe ningún paper que modele el aluvión de Taltal 2015.
2. **Embeddings v2**: leer varios productos (AlphaEarth 2025, OlmoEarth v1.3 en Zarr v3, TESSERA v2), emitir la extensión STAC `embeddings` v1.0 y ofrecer las operaciones que la literatura hace a mano: cambio entre años, asignación a prototipos, clustering y sondas few-shot con validación espacial. Es el tema de mayor crecimiento cercano a SurtGIS: los artículos que nombran AlphaEarth o "satellite embeddings" pasan de 1 en 2024 a 10 en 2025 y 124 en 2026.
3. **Backend determinista para agentes**: un servidor MCP nativo en Rust con contratos tipados (CRS, unidades, nodata), errores con causa y remedio, y provenance por llamada, evaluado en un benchmark público que ya se distribuye como servidor MCP (GeoNatureAgent). Los agentes LLM para GIS crecen 13,8 veces más rápido que la base "remote sensing".

Hay además un **arreglo rápido de alto valor demostrativo**: que la provenance registre los ítems STAC y la fecha de descarga de cada input. Planetary Computer sirvió 26.116 escenas S2 L2A corruptas entre el 26 y el 30 de octubre (verificado, issue #516), y hoy SurtGIS no puede decirle a un usuario si su composite las usó.

Frentes que conviene **abandonar**: el toolbox general en el navegador, el tiling COG en el cliente y los tiles sobre datacubes Zarr. opengeos y Earthmover los cubren con releases semanales.

---

## 2. Diagnóstico del estancamiento (verificado)

| Período | `feat` | `fix` | `docs` | `release` |
|---|---|---|---|---|
| Jun–ago 2026 | 62 | 22 | 12 | 9 |
| 1–28 sep 2026 | 11 | 4 | 5 | 2 |
| 28 sep – 4 nov 2026 | 0 | 5 | 5 | 4 |

Del análisis de septiembre se cerraron B (provenance y determinismo) y D1 (embeddings como raster). Quedaron sin tocar A (ejecución incierta), C (híbrido cliente–servidor), D2 (inferencia), E (solver de detritos probabilístico o en GPU), la geomorfometría esferoidal y GeoZarr. La acumulación ponderada de flujo entró dentro de un commit `fix` (Tal_Tal #12) y es la única función nueva del período.

**Inferido (muy probable)**: el patrón seguirá mientras cada reporte externo se atienda de inmediato con un release. Una regla simple ayuda: agrupar los arreglos no urgentes en un patch quincenal y proteger bloques de trabajo para un frente a la vez.

---

## 3. Qué cambió afuera desde el 28 de septiembre

**Datos y catálogos**
- **Planetary Computer, S2 L2A corrupto** (verificado). Valores uint16 dañados al pasar por uint8 desde el 2026-10-27 21:00 UTC. Hay 26.116 ítems afectados según un usuario; el backfill del 26 al 30 de octubre se declaró completo el 2026-11-04 (issue #516). Aparte, los COG escritos desde abril venían sin `GDAL_NODATA` (issue #513); SurtGIS es robusto a eso porque descarta 0 en el composite.
- **Landsat C2 L2 en el hemisferio sur** llega con EPSG 326xx y northings negativos (issue #510). Es relevante para Chile cuando se mezcla Landsat con S2.
- **Copernicus DEM en CDSE**: el View Service de GLO-30 quedó restringido y entrega GLO-90 por defecto; la copia en AWS sigue abierta.

**Embeddings**
- **OlmoEarth v1.3** (Ai2, 2026-10-22): embeddings globales 2017–2025 en **Zarr v3** por zona UTM, int8 con la misma fórmula de dequantización que AlphaEarth, 128 dimensiones Matryoshka (las 64 primeras son autónomas), sin normalizar en L2. Licencia con restricciones de uso militar y extractivo.
- **Extensión STAC `embeddings` v1.0.0** (2026-10-28): `emb:type`, `emb:dimensions`, `emb:quantization_method`, `emb:temporal_period`, enlace al encoder vía STAC MLM.
- **AlphaEarth 2025** registrado como dataset (2026-10-23). **TESSERA v2** se distribuye en Zarr.
- **Ningún modelo EO publica ONNX oficial** (verificado en Hugging Face por el agente). `ort` sigue en RC.

**Estándares**
- **STAC Processing v1.3.0** (2026-10-14): relaciones `source_for` (provenance hacia adelante).
- **OGC API Processes**: avanza Part 5 (provenance de jobs). Sin aprobación de v2 en el período (snippet).
- **GeoZarr**: las convenciones siguen en v0.1 desde junio; no hay spec v1.

**Competencia**
- **GDAL en crisis de gobernanza**: Even Rouault dejó la presidencia y el PSC el 2026-11-04 (verificado, gdal-dev). Hubo un PR para retirar los drivers Zarr e Icechunk, cerrado sin merge, y una política nueva contra contribuciones generadas por IA. GDAL 3.14 no tiene beta ni RC; muy probable que no salga en noviembre.
- **opengeos acelera**: whitebox-wasm 0.7 (licencia MIT/Apache), geolibre-rust 1.8, GeoLibre 3.6 con herramientas MCP que ejecutan Whitebox desde agentes, cog-tiler-wasm 0.6. El upstream `whitebox_next_gen` no tiene commits desde agosto; el ritmo lo lleva opengeos.
- **xpublish-tiles 0.13**: tiles MVT, WMS de series temporales y HEALPix sobre Zarr.
- **Agentes**: GeoNatureAgent (2026-10-06) publica su benchmark como servidor MCP de 16 herramientas y 103 tareas (Claude Sonnet 4: 61,7 %). Google GRA atribuye la mayor ganancia a herramientas de dominio y planificación. Esri y CARTO lanzan MCP comerciales. No hay ningún MCP GIS con contratos de CRS o provenance verificables.

Una corrección al análisis anterior: el abstract de GISAgentBench **no** dice que la desalineación de CRS sea la falla más costosa; según el cuerpo, dominan las operaciones omitidas o mal ordenadas (verificado en el abstract; el cuerpo, solo por snippets).

---

## 4. Señal bibliométrica (OpenAlex, artículos de revista)

Crecimiento 2023→2025 dividido por el crecimiento de la base "remote sensing" (×1,49). Detalle, consultas exactas y advertencias en el anexo A.

| Tema | 2023 | 2025 | 2026 parcial | Crecimiento relativo |
|---|---|---|---|---|
| Agentes LLM para GIS | 7 | 144 | 271 | **13,8** |
| Foundation models geoespaciales | 10 | 183 | 428 | **12,3** |
| Reproducibilidad geoespacial | 323 | 1115 | 1933 | 2,3 |
| DEM globales (Copernicus, FABDEM) | 26 | 81 | 128 | 2,1 |
| HAND | 7 | 21 | 40 | 2,0 |
| GLOF | 76 | 189 | 200 | 1,7 |
| Incertidumbre de DEM | 11 | 25 | 36 | 1,5 |
| Runout de flujos de detritos | 45 | 64 | 76 | 1,0 |
| Severidad de incendios | 397 | 552 | 626 | 0,9 |
| Susceptibilidad con deep learning | 61 | 72 | 94 | 0,8 |
| Chi, ksn, knickpoints | 146 | 127 | 150 | 0,6 |

Cómo leerla: el runout de detritos no crece más que la base, pero el aluvión de Taltal es demanda directa de un usuario y no requiere tendencia para justificarse. Chi/ksn y susceptibilidad con deep learning están planos o en baja: no conviene priorizarlos.

---

## 5. Oportunidades priorizadas

### O1. Alcance de valle confinado a cono aluvial (Grfin/LAHARZ corregido)

**Problema.** LAHARZ reparte un área planimétrica B fija. En un valle confinado, las secciones anchas de aguas arriba consumen B y la inundación nunca llega al cono. Además, las secciones perpendiculares sobre un cauce D8 a 30 m generan desbordes laterales en bloques. Es la falla que reportó Tal_Tal en Taltal.

**Evidencia (verificado).**
- Reid et al. 2025, *Grfin Tools — User Guide and Methods*, USGS TM 14-A3, doi:10.3133/tm14A3, sec. 3.5.3: *"A fixed-volume cross-sectional area may limit downstream extent as planimetric area is expended upstream in larger cross sections."*
- Grfin evalúa tres secciones por celda, usa un "círculo promedio" rellenado por BFS en terreno no confinado y hace crecer el volumen aguas abajo según el área aportante. Corre solo en Windows, depende de TauDEM, sigue cauces D8 y no bifurca.
- Reid et al. 2026, *Landslides*, doi:10.1007/s10346-026-02751-8: Grfin reproduce áreas inundadas en Micronesia.
- Rickenmann y Scheidl 2023, doi:10.1051/e3sconf/202341507013: la movilidad depende de la pendiente del cono, lo que respalda coeficientes distintos para valle y cono.
- Cabré et al. 2025, *ESPL*, doi:10.1002/esp.70041: conos de confluencia en valles confinados de Atacama en 2015 y 2017.
- No hay ningún paper que modele el aluvión de Taltal 2015 (búsqueda en OpenAlex y web sin resultados).

**Diseño.**
1. Índice de confinamiento por tramo desde HAND (ancho del fondo de valle).
2. Presupuesto de B en dos partes: el tramo confinado consume solo lo que excede el cauce activo; el ápice del cono recibe el resto.
3. Coeficientes α distintos para valle y cono, por pendiente o confinamiento.
4. Círculo promedio y altura media de flujo en zona no confinada, como Grfin.
5. Secciones sobre un eje suavizado, no sobre la escalera D8.
6. En el cono, reparto del volumen remanente entre varias trayectorias con MFD (Grfin no bifurca).
7. Ensamble Monte Carlo sobre volumen, α y error de DEM: probabilidad de inundación por celda.

**Encaje.** Reutiliza `hazards::laharz`, HAND, MFD/TFGA, la acumulación ponderada (crecimiento del volumen), swath (secciones) y priority-flood.

**Validación.** Taltal con el polígono de Sernageomin que ofreció Tal_Tal (meta: superar Jaccard 0,17); conos de Cabré et al.; las dos quebradas de 2015 modeladas con FLO-2D en Zegers et al. 2020 (doi:10.5194/nhess-20-1919-2020).

**Dificultad**: media. **Entregable**: comando `hazards runout-cone` (nombre provisional) y paper en *C&G*, *NHESS* o *Landslides* (inferido, probable).

**Extensión natural (O1b)**: tamizaje regional de quebradas peligrosas, combinando crecimiento de volumen, alcance H/L, número de Melton y la clasificación de conos dominados por detritos o por crecidas de Bazilova et al. 2025 (doi:10.1029/2024jf008029). Dificultad baja-media una vez hecho O1.

### O2. Embeddings v2: varios productos, STAC y operaciones temporales

**Evidencia.**
- Crecimiento: la subconsulta AlphaEarth/"satellite embeddings" pasa de 1 (2024) a 10 (2025) y 124 (2026) artículos (verificado).
- Fricción: "Harvesting AlphaEarth" (*JAG* 2026, doi:10.1016/j.jag.2026.105258) reclama la configuración que exige desplegarlos; la mayoría de los estudios usa Earth Engine.
- Operaciones que hoy se hacen a mano (abstracts leídos):
  - punto de cambio por píxel entre años (TESSERA en Kenia, doi:10.31223/x5gf88);
  - asignación a la referencia más cercana sin entrenamiento (mismo trabajo);
  - few-shot: con 8 bloques etiquetados en una región nueva se recupera la ventaja de los embeddings en severidad de incendios (arXiv 2609.34602);
  - susceptibilidad a deslizamientos un año adelante con ROC-AUC > 0,95 usando el embedding del año anterior (npj Natural Hazards, doi:10.1038/s44304-026-00268-7).
- Riesgos documentados: fuga espacial (las coordenadas son recuperables desde los embeddings) y *shortcut learning* al cambiar de evento.

**Diseño.**
1. Lector con presets `--product alphaearth|olmoearth|tessera-v2`: la dequantización de OlmoEarth ya es la de AlphaEarth (verificado en `crates/algorithms/src/embeddings/mod.rs`); falta la ruta Zarr v3 proyectada por zona UTM (time, band, y, x), normalización L2 explícita y lectura de las primeras k dimensiones.
2. Leer y escribir la extensión STAC `embeddings` v1.0 y propagar la licencia en la provenance (CC-BY, CC0 y licencia restringida de OlmoEarth no deben mezclarse sin aviso).
3. Comandos nuevos: `embeddings change`, `embeddings prototype`, `embeddings cluster` (en streaming) y `embeddings probe` (lineal o logístico, con validación por buffer espacial obligatoria por defecto).
4. Integración con susceptibilidad: embeddings más variables de terreno, con validación temporal un año adelante.

**Encaje.** Construye sobre D1 (lector COG, similitud, PCA, `extract`, tiles en `serve`). **Dificultad**: baja-media para 1–2, media para 3. **Ventaja**: streaming con RAM acotada y CLI sin Python ni cuenta Google.

### O3. Backend determinista para agentes (MCP nativo en Rust)

**Evidencia.** Crecimiento ×13,8 relativo a la base (verificado). GIS Copilot (*IJDE* 2025) no ejecuta de forma autónoma las tareas avanzadas; GeoCogent (*IJGIS* 2025) identifica las alucinaciones de código como problema central. GeoNatureAgent ya distribuye su benchmark como servidor MCP, así que evaluar es directo. GRA (Google) atribuye la ganancia a herramientas de dominio y planificación. No se encontró ningún MCP GIS con contratos de CRS ni provenance.

**Diseño.**
1. Servidor MCP en Rust con `rmcp` 3.8 que reemplace o envuelva las 42 herramientas Python actuales.
2. JSON Schema por herramienta con CRS, unidades y nodata en entradas y salidas; rechazo explícito de grillas o CRS incompatibles; modo dry-run que valida la grilla sin calcular.
3. Errores con causa y remedio (el estilo que ya usan los arreglos de 1.6.0: "run `surtgis reproject` first").
4. Provenance por llamada y un `AGENTS.md`, como el que publica OlmoEarth.
5. Herramientas compuestas (cuenca completa, susceptibilidad) para reducir los pasos que el agente debe planificar.
6. Evaluación en GeoNatureAgent y, si se puede, en GISAgentBench.

**Dificultad**: media. **Entregable**: nota de software o paper corto con el benchmark (inferido, probable).

### O4. Provenance de inputs remotos: "¿mi composite usó datos corruptos?" (arreglo rápido)

**Evidencia (verificado).** El incidente PC #516 publica una lista de 26.116 ítems afectados. La provenance actual registra, por input, ruta, hash, bytes, rol y forma (`crates/core/src/provenance.rs`), pero no los IDs de ítem STAC, la colección, el ETag ni la fecha de descarga.

**Diseño.**
1. Registrar en la provenance de `stac composite` y `time-series` el ID de cada ítem usado, la colección, el catálogo, el ETag o Last-Modified del asset y la hora de lectura.
2. `surtgis verify --against <lista>` marca los archivos que usaron ítems de la lista dentro de una ventana de tiempo.
3. Mapear la provenance a STAC Processing v1.3 (`processing:software`, enlaces `derived_from` y `source_for`).

**Dificultad**: baja-media. **Valor**: es la mejor demostración disponible de para qué sirve la provenance, con un caso real y reciente.

### O5. HAND con volumen conservado, curvas sintéticas y evaluación integrada

**Evidencia.** HAND crece 2,0 veces la base y fue la variable central del paper de Tal_Tal. TOPFIM reparte un volumen conservado con CSI 0,83 contra HEC-RAS (*WRR* 2025, doi:10.1029/2024wr037851). DHAND acopla HAND con un modelo 1D (*GMD* 2025, doi:10.5194/gmd-18-3387-2025). Las curvas de descarga sintéticas desde HAND mejoran con corrección hidrodinámica (*JFRM* 2025, doi:10.1111/jfr3.70057). La evaluación vive en una herramienta aparte (FIMeval, *EMS* 2025). Demanda chilena: Copiapó, Chañaral y El Salado en 2015 (doi:10.1002/esp.6017).

**Diseño.** Curvas de descarga sintéticas por tramo, asignación de volumen conservado y métricas CSI/F1/Jaccard en el mismo comando. Comparte el asignador de volumen con O1.

**Dificultad**: media.

### O6. Motor común de ensambles (componente transversal)

En lugar de un frente aparte de "ejecución incierta", conviene un ejecutor de ensambles compartido por O1, O5 y `surtgis-flow`: campos de error de DEM con correlación espacial, muestreo LHS de parámetros y acumulación por celda (media, desviación, probabilidad) con RAM acotada.

**Evidencia.** En GLOF modelados con r.avaflow, el DEM es el segundo parámetro más sensible y hicieron falta 78 corridas manuales (*NHESS* 2025, doi:10.5194/nhess-25-1841-2025). Un surrogate gaussiano da probabilidad de inundación aguas abajo (*NHESS* 2026, doi:10.5194/nhess-26-1705-2026). La propagación estricta de error de DEM a derivados casi no tiene literatura (17 artículos en 2026): el hueco es real pero la demanda es chica como producto aislado (inferido, tan probable como no).

**Dificultad**: media. Se construye una vez y lo usan los tres.

### Otras, de menor prioridad

- **GLOF con incertidumbre** para Aysén y Magallanes (Chen et al. 2025, doi:10.5194/hess-29-733-2025; el tipo de presa importa, Veh et al. 2025, doi:10.1038/s44221-025-00388-w). Usa agua con SAR, composites, HAND y `surtgis-flow`. Dificultad media.
- **Cadena post-incendio** dNBR → probabilidad → volumen → inundación. `pfdf` del USGS ya cubre los modelos de probabilidad y volumen en Python; lo que falta es integrar la cadena y recalibrar para Chile. Útil en cursos.
- **Endpoints `/point` y `/timeseries`** en `surtgis serve` sobre composites COG/STAC.
- **Plugin de SurtGIS para GeoLibre**: distribución sin competir de frente, exponiendo lo que Whitebox no tiene (solver de detritos, provenance).

---

## 6. Qué dejar

| Frente | Por qué |
|---|---|
| Toolbox general de terreno en el navegador | whitebox-wasm, geolibre-rust y GeoLibre lo cubren con cientos de herramientas y releases semanales. Mantener el WASM de SurtGIS para nichos (ASI, visor de territorio-digital). |
| Tiling COG en el cliente | Lo cubre cog-tiler-wasm. |
| Tiles sobre datacubes Zarr | Lo cubre xpublish-tiles 0.13. `serve` debe quedarse en COG/STAC/DEM. |
| Escritura GeoZarr multiscales | Convenciones congeladas en v0.1, sin spec v1. Esperar. |
| Inferencia de ViT grandes en el navegador | Sin ONNX oficial; Prithvi-Tiny ajustado no supera a una U-Net en Sen1Floods11. Si se hace inferencia, empezar por modelos destilados de menos de 1 M de parámetros. |
| Chi/ksn y susceptibilidad solo con deep learning como líneas propias | Literatura plana o en baja frente a la base. |
| Unidades de pendiente | `r.slopeunits` de GRASS ya existe. |

---

## 7. Plan sugerido (8 semanas)

1. **Semana 1**: O4 (ítems STAC y fechas en la provenance, `verify --against`, mapeo a STAC Processing 1.3). En paralelo, pedir a Tal_Tal el DEM, la fuente y el polígono de Sernageomin de Taltal.
2. **Semanas 2–4**: O1 con el ejecutor de ensambles mínimo de O6. Validar en Taltal y en las quebradas de Zegers et al. 2020.
3. **Semanas 5–6**: O2, pasos 1–3 (lector multi-producto, STAC `emb:`, `change` y `prototype`).
4. **Semanas 7–8**: O3 (MCP en Rust) y evaluación en GeoNatureAgent.

Después: O5 (HAND con volumen), O1b (tamizaje regional) y GLOF.

**Mantención que no debe esperar**, como patch 1.6.1 agrupado:
- la combinación `--no-default-features --features cloud,projections`, que el README recomienda, no compila desde ≤ 1.5.0;
- actualizar el crate `gdal` y quitar la fijación del job de CI a `ubuntu-24.04`;
- tests de regresión de Planetary Computer: COG sin `GDAL_NODATA`, Landsat sur con EPSG 326xx y northings negativos.

---

## 8. Límites

- Los conteos de OpenAlex se limitan a artículos de revista porque 2026 trae millones de registros de Zenodo; 2026 está probablemente subestimado por el retraso de indexación.
- Varias afirmaciones de los anexos provienen de snippets de búsqueda y están marcadas así. Las que sostienen las recomendaciones de la sección 5 se verificaron en la fuente primaria.
- "No existe" significa "no aparece publicado ni documentado", no "es imposible que exista".
- No se revisaron las licencias del código de Grfin ni de Blackbird.


---

# Anexos: reportes por frente

Cada anexo es el reporte íntegro del agente de investigación correspondiente. Las marcas [V] y [S] son de cada agente: verificado en fuente primaria y solo snippet, respectivamente.

## Anexo A. Bibliometría (OpenAlex)

**Método.** Filtro común: `filter=title_and_abstract.search:<Q>,publication_year:2021-2026,type:article,primary_location.source.type:journal&group_by=publication_year`. OpenAlex trae 35,2 M obras de 2026 contra 14,4 M de 2025, sobre todo datasets de Zenodo; por eso se restringe a artículos de revista (2025: 6,37 M; 2026: 5,47 M). 2026 anualizado = conteo × 12/10,13. Crecimiento relativo = crecimiento del tema / crecimiento de la base "remote sensing" (×1,49 entre 2023 y 2025). La búsqueda aplica stemming; obras sin abstract se buscan solo en título.

| Tema | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 parcial | ×23→25 | rel. base | 2026 anual./2025 rel. |
|---|---|---|---|---|---|---|---|---|---|
| Agentes LLM-GIS | 2 | 2 | 7 | 40 | 144 | 271 | 20,6 | 13,8 | 2,02 |
| Foundation model geoespacial (estricto) | 0 | 2 | 10 | 75 | 183 | 428 | 18,3 | 12,3 | 2,52 |
| Reproducibilidad geoespacial | 325 | 285 | 323 | 444 | 1115 | 1933 | 3,45 | 2,32 | 1,86 |
| DEM globales Copernicus/FABDEM | 7 | 15 | 26 | 57 | 81 | 128 | 3,12 | 2,09 | 1,70 |
| HAND | 12 | 13 | 7 | 20 | 21 | 40 | 3,00 | 2,02 | 2,05 |
| GLOF | 95 | 83 | 76 | 121 | 189 | 200 | 2,49 | 1,67 | 1,14 |
| Incertidumbre de DEM (amplio) | 13 | 19 | 11 | 14 | 25 | 36 | 2,27 | 1,53 | 1,55 |
| FM/embeddings EO (amplio) | 407 | 471 | 526 | 711 | 1163 | 2020 | 2,21 | 1,49 | 1,87 |
| Cloud-native (ruidoso) | 4 | 1 | 10 | 13 | 22 | 72 | 2,20 | 1,48 | 3,52 |
| GPU raster/hidrología (ruidoso) | 28 | 44 | 25 | 34 | 48 | 76 | 1,92 | 1,29 | 1,70 |
| ICESat-2/GEDI y DSM→DTM | 102 | 136 | 152 | 207 | 253 | 262 | 1,66 | 1,12 | 1,11 |
| AlphaEarth/TESSERA/Prithvi/Clay | 67 | 64 | 84 | 88 | 138 | 261 | 1,64 | 1,10 | 2,03 |
| S1/SAR humedad de suelo/sequía | 164 | 153 | 153 | 178 | 235 | 240 | 1,54 | 1,03 | 1,10 |
| BASE remote sensing | 12023 | 13453 | 14561 | 17512 | 21656 | 20134 | 1,49 | 1,00 | 1,00 |
| Runout de flujos de detritos | 51 | 47 | 45 | 61 | 64 | 76 | 1,42 | 0,96 | 1,28 |
| Severidad de incendios / post-incendio | 342 | 407 | 397 | 430 | 552 | 626 | 1,39 | 0,93 | 1,22 |
| BASE "digital elevation model" | 1491 | 1478 | 1423 | 1617 | 1953 | 1977 | 1,37 | 0,92 | 1,09 |
| BASE landslide | 3497 | 3682 | 4063 | 4738 | 5281 | 4894 | 1,30 | 0,87 | 1,00 |
| Susceptibilidad + deep learning | 28 | 51 | 61 | 63 | 72 | 94 | 1,18 | 0,79 | 1,40 |
| Chi/ksn/knickpoints | 130 | 112 | 146 | 143 | 127 | 150 | 0,87 | 0,58 | 1,27 |

Subconsultas: `AlphaEarth OR "satellite embedding(s)"` 2024: 1 → 2025: 10 → 2026: 124. Propagación estricta de error de DEM a derivados 2023: 3 → 2025: 8 → 2026: 17. Runout probabilístico 2023: 4 → 2025: 7 → 2026: 11. Post-incendio con flujo de detritos, erosión o escorrentía 2023: 71 → 2025: 117 → 2026: 121. "Cloud-native" mezcla genómica y "GPU" mezcla UAV: no se usan como evidencia fuerte. "Reproducibilidad" crece en parte por moda léxica (probable).

**Consultas exactas (`<Q>`).** Base RS `"remote sensing"`; base DEM `"digital elevation model"`; base landslide `landslide OR landslides`; LLM-GIS `("large language model" OR LLM OR GPT) AND (GIS OR geospatial OR "spatial analysis")`; FM estricto `"foundation model" AND ("remote sensing" OR "Earth observation" OR geospatial)`; FM amplio `("foundation model" OR embedding OR embeddings) AND ("remote sensing" OR "Earth observation" OR geospatial OR satellite)`; AlphaEarth `AlphaEarth OR "satellite embedding" OR "satellite embeddings"`; reproducibilidad `(reproducibility OR reproducible OR replicability) AND (geospatial OR GIS OR "remote sensing" OR "geographic information")`; DEM globales `"Copernicus DEM" OR "GLO-30" OR FABDEM OR "Copernicus GLO"`; HAND `"height above nearest drainage"`; GLOF `"glacial lake outburst" OR GLOF`; incertidumbre amplia `("DEM uncertainty" OR "DEM error" OR "elevation error" OR "elevation uncertainty") AND (propagation OR "Monte Carlo" OR slope OR derivative OR terrain)`; incertidumbre estricta `("error propagation" OR "uncertainty propagation" OR "Monte Carlo") AND ("digital elevation model" OR DEM) AND (slope OR curvature OR "flow accumulation" OR watershed OR "terrain attributes" OR derivatives)`; runout `("debris flow" OR "debris-flow") AND (runout OR "run-out" OR inundation) AND (model OR simulation)`; runout probabilístico `("debris flow" OR "debris-flow" OR lahar OR "mass flow") AND (runout OR "run-out") AND (probabilistic OR uncertainty OR "Monte Carlo" OR ensemble)`; incendios `("burn severity" OR dNBR OR "fire severity") OR ("post-fire" AND (erosion OR "debris flow"))`; cloud-native `Zarr OR "cloud optimized GeoTIFF" OR "cloud-optimized GeoTIFF" OR "SpatioTemporal Asset Catalog" OR ("cloud-native" AND geospatial)`; GPU `(GPU OR WebGPU OR CUDA) AND (raster OR "digital elevation model" OR "flow accumulation" OR "hydrological model" OR terrain)`; chi `"chi analysis" OR "channel steepness" OR ksn OR knickpoint OR knickpoints`; ICESat `(("ICESat-2" OR GEDI) AND (DEM OR terrain OR elevation)) OR ("DSM" AND "DTM" AND (generation OR conversion OR filtering))`; S1 `("Sentinel-1" OR SAR) AND ("soil moisture" OR drought)`; susceptibilidad DL `"landslide susceptibility" AND ("deep learning" OR "convolutional neural network" OR CNN OR transformer)`. Trabajos clave: misma consulta con `publication_year:2025-2026`, `sort=cited_by_count:desc`, abstract desde `abstract_inverted_index`.

**Trabajos clave 2025–2026 (abstract leído salvo indicación).**
- Agentes: GIS Copilot (*IJDE* 2025, doi:10.1080/17538947.2025.2497489), LLM sobre QGIS y GeoPandas; las tareas avanzadas aún no se ejecutan de forma autónoma. Autonomous GIS agent for data retrieval (*IJDE* 2025, doi:10.1080/17538947.2025.2458688). GeoCogent (*IJGIS* 2025, doi:10.1080/13658816.2025.2549460): "code hallucinations" como problema central, benchmark GeoCodes. Autonomous GIS (*Annals of GIS* 2025, doi:10.1080/19475683.2025.2552161).
- Embeddings: Prithvi-EO-2.0 (*IEEE TGRS* 2025, doi:10.1109/tgrs.2025.3642610). Harvesting AlphaEarth (*JAG* 2026, doi:10.1016/j.jag.2026.105258): desplegar estos modelos "demands extensive configuration, environment setup, inference EO preparation"; falta transferibilidad, interpretabilidad y estabilidad. AlphaEarth para calidad del aire en Quito (*Remote Sensing* 2025, doi:10.3390/rs17203472), vía Earth Engine. Tree species mapping (*Sci. Remote Sens.* 2026, doi:10.1016/j.srs.2026.100466, solo inicio del abstract).
- DEM globales: GEDTM30 (*PeerJ* 2025, doi:10.7717/peerj.19673), 15 parámetros a 6 escalas en ~800 teselas con solape, en Python con GDAL y Whitebox Workflows. FathomDEM (*ERL* 2025, doi:10.1088/1748-9326/ada972). GDEM corregido con ICESat-2 (*ESSD* 2025, doi:10.5194/essd-17-205-2025). GRIT (*WRR* 2025, doi:10.1029/2024wr038308), red hidrográfica con bifurcaciones.
- HAND: HAND-SRC en llanuras (*J. Hydrol.* 2025, doi:10.1016/j.jhydrol.2024.132649), hidroacondicionamiento adaptado; precisión cae con celdas mayores a 1 m. Blackbird/DHAND (*GMD* 2025, doi:10.5194/gmd-18-3387-2025). FIMeval (*EMS* 2025, doi:10.1016/j.envsoft.2025.106786).
- GLOF/flujos: Sikkim 2023 (*Science* 2025, doi:10.1126/science.ads2659). Incertidumbre GLOF con r.avaflow (*NHESS* 2025, doi:10.5194/nhess-25-1841-2025): 78 simulaciones, DEM segundo parámetro más sensible, malla el cuarto. Hit probability Monte Carlo (*JFRM* 2025, doi:10.1111/jfr3.70011).
- Post-incendio: erosión global (*Nat. Geosci.* 2026, doi:10.1038/s41561-025-01876-0). Oregon (*ESPL* 2025, doi:10.1002/esp.70045): el modelo USGS está calibrado solo con el sur de California. RUSLE post-incendio en GEE (*Fire Ecology* 2025, doi:10.1186/s42408-025-00363-6).

**Huecos de software priorizados por el agente.** (1) AlphaEarth y embeddings como raster sin GEE; (2) backend determinista para agentes; (3) HAND completo con hidroacondicionamiento, curvas sintéticas y evaluación; (4) ensembles Monte Carlo de flujos y GLOF con DEM perturbado; (5) terreno multiescala global con tiling seguro; (6) propagación de incertidumbre de DEM (como componente, no producto); (7) cadena post-incendio; (8) reproducibilidad como argumento transversal. Señales negativas: chi/ksn (×0,58), susceptibilidad con DL (×0,79), S1 humedad (×1,03), ICESat/GEDI (×1,12). Supuesto: que los abstracts reflejan qué software usan los autores (solo 12 artículos de AlphaEarth en 2026 nombran GEE en el abstract).

## Anexo B. Ecosistema de herramientas y estándares

| Fecha | Qué | Fuente | Evaluación |
|---|---|---|---|
| 2026-11-04 | Even Rouault deja la presidencia y el PSC de GDAL. [V] | lists.osgeo.org/pipermail/gdal-dev/2026-November/061790.html | Riesgo sistémico; refuerza el valor de SurtGIS sin GDAL. |
| 2026-10-15/16 | PR #15399 propone retirar los drivers Zarr e Icechunk ("AI slop"); cerrado sin merge. [V] | github.com/OSGeo/gdal/pull/15399 | Futuro de Zarr en GDAL incierto. |
| 2026-10-20 | CONTRIBUTING de GDAL: "vibe coding y agentes autónomos no son bienvenidos" (#15415). [V] | github.com/OSGeo/gdal/pull/15415 | Aportes upstream: PRs humanos y chicos. |
| a 2026-11-03 | GDAL 3.14 sin tags beta ni RC (último v3.13.3, 2026-08-18). [V] | API de tags OSGeo/gdal | Muy improbable en noviembre. |
| 2026-10-27→11-04 | PC sentinel-2-l2a: uint16 corrompido vía uint8 desde 10-27 21:00 UTC; 26.116 ítems; backfill completo 11-04. [V] | microsoft/PlanetaryComputer#516 | Afecta composites de usuarios; oportunidad para provenance. |
| 2026-10-23 | PC S2 L2A: COG desde 2026-04-17 sin GDAL_NODATA; fix 10-28. [V] | issue #513 | SurtGIS robusto (descarta 0). |
| 2026-09-24→28 | PC: búsqueda multi-colección rota (corregida) y 429 en `/sas/v1/sign`. [V] | issues #505, #506 | SurtGIS cachea el SAS por contenedor. |
| 2026-10-16 | Landsat C2 L2 sur con EPSG 326xx y northings negativos. [V] | issue #510 | Test de regresión para Chile. |
| 2026-10-28 | STAC Embeddings v1.0.0. [V] | stac-extensions/embeddings | Oportunidad directa. |
| 2026-10-14 | STAC Processing v1.3.0, `source_for`. [V] | stac-extensions/processing | Calza con la provenance BLAKE3. |
| 2026-10-20 | STAC `ortho` v0.1.0; `access` y `maps` en borrador. [V] | stac-extensions | Marginal. |
| 2026-11-02 | Convención Zarr DGGS activa; multiscales, proj y spatial en v0.1 (2026-06-12). [V] | github.com/zarr-conventions | GeoZarr v1 no se movió. |
| 2026-10/11 | OGC API Processes: commits en Part 3 (CWL) y Part 5 (provenance). Sin aprobación de v2. [V repo / S aprobación] | opengeospatial/ogcapi-processes | Part 5 relevante para jobs de `serve`. |
| 2026-10-31 | whitebox-wasm 0.7.0/0.7.2, licencia MIT/Apache. [V] | opengeos/whitebox-wasm | Amenaza en el navegador. |
| 2026-10-11→31 | geolibre-rust 1.6→1.8.1 (295★). [V] | opengeos/geolibre-rust | Actividad real de Whitebox Next Gen. |
| desde 2026-08-31 | whitebox_next_gen sin commits; crates `wb*` ~20k descargas recientes. [V] | GitHub API, crates.io | Upstream quieto. |
| 2026-10-25→11-02 | GeoLibre 3.5→3.6.1: Model Builder con bucles, herramientas MCP `live_*`. [V] | opengeos/GeoLibre | Hub GUI+IA de Whitebox. |
| 2026-10-09/31 | cog-tiler-wasm 0.5/0.6. [V] | opengeos/cog-tiler-wasm | Tiling COG cliente cubierto. |
| 2026-10-25→11-03 | xpublish-tiles 0.12→0.13.1: MVT, WMS de series, HEALPix. [V] | earth-mover/xpublish-tiles | Cierra tiles sobre Zarr. |
| 2026-10-06/27 | icechunk 2.3.0/2.3.1, RC 2.4. [V] | earth-mover/icechunk | Incremental. |
| 2026-10-06→11-02 | rio-tiler 9.4.13; TiTiler sin release desde 2.4.0. [V] | GitHub | Quieto. |
| 2026-10-26 | rustac: stac 0.17.7, stac-server 0.5.9. [V] | stac-utils/rustac | Candidato a reemplazar el cliente STAC propio. |
| 2026-10-18→11-04 | geotiff-reader 0.9→0.11 (roteiro-gis), 17★, un autor. [V] | crates.io | Vigilar; poca adopción. |
| sin cambios | GDAL 3.13.3, GRASS 8.5.0, xDEM 0.2.3, RichDEM 2.3.1, PDAL 2.10.2, zarrs 0.23.14, async-tiff 0.4, geoarrow-rs 0.9; TopoToolbox paper aún preprint. [V / S paper] | GitHub | — |
| 2026-10-30 (programado) | QGIS 4.4.0 en roadmap, sin tag (último 4.2.3). [V] | qgis.org | Atrasado. |
| 2026-07/08 | CDSE restringe View Service de COP-DEM-GLO-30; AWS sigue abierto. [V] | dataspace.copernicus.eu/node/4044, /node/4100 | Solo afecta a usuarios de CDSE. |
| 2026-07-29 | Google "Custom Satellite Embeddings", preview enterprise de pago. Source Cooperative cubre 2017–2025. [V] | blog de Google Maps Platform | Cerrado. |

**Oportunidades según el agente.** (1) `surtgis verify` contra listas de ítems afectados (requiere guardar ETag/Last-Modified); (2) sidecar STAC Embeddings v1.0 y MLM; (3) provenance a STAC Processing 1.3 y seguimiento de OGC Processes Part 5; (4) Zarr/Icechunk sin GDAL como diferenciador; (5) test del caso AlphaEarth "bottom-up" con un tile de Source Coop; (6) endpoints `/point` y `/timeseries` en `serve`; (7) tests de regresión de Planetary Computer; (8) plugin para GeoLibre.

**Ya no vale la pena según el agente.** Toolbox general en el navegador; tiling COG en el cliente; tiles sobre datacubes Zarr; writer GeoZarr multiscales; paridad con TiTiler; publicar el lector COG como crate aparte (async-tiff y geotiff-rust ocupan el espacio; confianza media).

**Límites.** No cargó la página de noticias de OGC (timeout); no verificó cambios del catálogo de Earth Engine en el período.

## Anexo C. GeoAI, inferencia y agentes

**Productos y formatos.**
- [V] OlmoEarth v1.3 (Ai2, 2026-10-22), source.coop/olmoearth/olmoearth-v1-3-embeddings: Zarr v3, un grupo por zona UTM, ejes (time, band, y, x), shards 2048², chunks 256²×64, zstd 3; int8 con `v = sign(q)·(|q|/127.5)²`, −128 nodata; 128 dimensiones Matryoshka (0–63 autónomas), sin normalizar L2; 2017–2025, global solo 2021–2025; procedencia en GeoParquet y `AGENTS.md`; acceso anónimo. Licencia "OlmoEarth Artifact License" con restricciones (uso extractivo según prensa). Modelo en Hugging Face como `weights.pth`, sin ONNX.
- [V] STAC `embeddings` v0.0.1 (10-27) y v1.0.0 (10-28), prefijo `emb:`, estado "Proposal".
- [V metadatos] AlphaEarth 2025 registrado en OpenAlex (doi:10.71646/ghsz7j, 2026-10-23); la ficha de Earth Engine sigue hasta 2025-01-01. Source Cooperative `tge-labs/aef` con índice GeoParquet y CC-BY 4.0 [S].
- [V] TESSERA v2: checkpoints en Hugging Face desde 2026-07-24, CC0, sin ONNX; embeddings v2 en Zarr [S]; el orden de bandas S2 no estándar falla en silencio [S].
- [S] Clay y Major TOM sin versión nueva; Copernicus-Embed a 0,25°.

**Aplicaciones (abstracts leídos).**
- Deslizamientos: Fang et al. 2026 (npj Natural Hazards, doi:10.1038/s44304-026-00268-7), ROC-AUC > 0,95 un año adelante; el 5 % superior capta el 90–95 % del área futura; cae en 2024 por cambio de gatillantes. Hong Kong con PU-learning (doi:10.21203/rs.3.rs-10786241/v1).
- Agua: Sentinel-1 + AlphaEarth sube IoU de 0,77 a 0,85 y supera a OPERA DSWx-S1 en 48 de 53 escenas (arXiv 2610.06704).
- Incendios: TESSERA y AlphaEarth superan a los índices por 0,05–0,13 AUC dentro de un incendio y pierden 0,15–0,18 en otra región; con 8 bloques etiquetados se recupera la ventaja (arXiv 2609.34602).
- Sin entrenamiento: TESSERA en Kenia (doi:10.31223/x5gf88), asignación a la más cercana de 2 referencias y un punto de cambio por píxel que fecha el 92 % del cambio.
- Advertencias: *shortcut learning* (exactitud balanceada 0,807 → 0,638 al dejar fuera un desastre, arXiv 2610.13766); coordenadas recuperables desde los embeddings (arXiv 2609.29151); Prithvi-Tiny (IoU 0,833) no supera a U-Net (0,856) en Sen1Floods11 (arXiv 2610.34904).

**Inferencia.** [V] Sin ONNX oficial en IBM-ESA, Ai2, geotessera ni Clay; solo ONNX comunitario (`nthh/prithvi-eo-tiny-onnx`, 0 descargas). Destilación de Prithvi-300M a EfficientViT de 0,7 M (arXiv 2609.20441). Crates: `ort` 2.0.0-rc.15 (10-19), `tract-onnx` 0.23.9 (10-15), `burn` 0.22.1 (10-20), `candle` 0.11.1 (10-31), `rmcp` 3.8.0 (10-26). En navegador, el precedente es GeoAI.js con ONNX Runtime Web [S].

**Agentes.** [V] GISAgentBench (arXiv 2608.01645): el abstract no menciona CRS; según snippets dominan operaciones omitidas o mal ordenadas. [V] GeoNatureAgent (arXiv 2610.09112): benchmark como MCP de 16 herramientas, 103 tareas; Claude Sonnet 4 61,7 %, DeepSeek V3.2 57,9 %. [V] Google Earth AI GRA (arXiv 2610.15750): ganancia por skills y herramientas de dominio; la planificación previa permite que Flash iguale a Pro. MCP GIS existentes: sparkgeo/geo-mcp-servers (lista, 247★), honua-io/geospatial-mcp, gis-mcp (Python, 92+ herramientas, [S]), gdal-mcp [S]; Esri en beta y CARTO comercial [S]; Earth Engine solo comunitarios [S]; MCP de riesgo Hydrata/ANUGA. Ninguno con contratos de CRS/unidades ni procedencia.

**Oportunidades según el agente.** (1) Lector multi-producto con STAC `emb:` (el dequantizador ya sirve para OlmoEarth; falta Zarr UTM proyectado en `zarr_reader.rs`); (2) primitivas `change`, `prototype`, `cluster`, `probe`; (3) susceptibilidad con embeddings + terreno y validación un año adelante; (4) MCP nativo con `rmcp`, JSON Schema, rechazo de CRS mixtos, dry-run, procedencia y `AGENTS.md`; (5) evaluación en GeoNatureAgent/GISAgentBench; (6) tiles de cambio y clustering en `serve`; (7) inferencia ONNX empezando por modelos destilados de menos de 1 M.

**Riesgos.** Licencias heterogéneas; volumen (~150 TB por año en OlmoEarth, prensa); validez científica (fuga espacial, *shortcut learning*); volatilidad de `ort`, `rmcp` y la extensión STAC; competencia de Esri, CARTO y Google.

## Anexo D. Hazards y geomorfometría con demanda en Chile

**P1. Inundación tipo LAHARZ con zona confinada y cono separadas (Taltal).** [V] Reid et al. 2025, USGS TM 14-A3 (doi:10.3133/tm14A3; software doi:10.5066/P9NVKFE2), sec. 3.5.3: *"A fixed-volume cross-sectional area may limit downstream extent as planimetric area is expended upstream in larger cross sections."* Grfin evalúa tres secciones por celda (perpendicular y ±45°), usa un círculo promedio rellenado por BFS, altura media `hbar` sin paredes y volumen creciente aguas abajo. Limitaciones: cauces D8 de TauDEM, sin bifurcación, solo área máxima potencial, binario C++ para Windows con TauDEM (licencia no revisada). Respaldo: Reid et al. 2026 (*Landslides*, doi:10.1007/s10346-026-02751-8); Rickenmann y Scheidl 2023 (doi:10.1051/e3sconf/202341507013); Cabré et al. 2025 (*ESPL*, doi:10.1002/esp.70041); Herbert et al. 2024 (*Landslides*, doi:10.1007/s10346-023-02207-3, solo título); DFLOWZ (*C&G* 2014, doi:10.1016/j.cageo.2014.02.002). Diseño: confinamiento desde HAND (método VHS, doi:10.5281/zenodo.21685464), presupuesto B en dos partes, α1/α2, círculo promedio, secciones sobre eje suavizado, MFD en el cono, Monte Carlo. Validación: Taltal, Cabré et al., Zegers et al. 2020 (doi:10.5194/nhess-20-1919-2020).

**P2. Tamizaje regional start-to-stop.** Reid et al. 2026 y Grfin; Bazilova, de Haas e Immerzeel 2025 (*JGR-ES*, doi:10.1029/2024jf008029), CatBoost en 1.793 cuencas, Melton como predictor principal; Bloomberg et al. 2025 (*NHESS*, doi:10.5194/nhess-25-647-2025).

**P3. Ejecución probabilística con emulador para `surtgis-flow`.** Spiller et al. 2026 (*NHESS*, doi:10.5194/nhess-26-1705-2026), surrogate gaussiano; Rinzin et al. 2025 (*NHESS*, doi:10.5194/nhess-25-1841-2025); Kumar, Yıldız y Kowalski 2026 (*NPG*, doi:10.5194/npg-33-425-2026), divergencia KL para elegir observaciones de calibración.

**P4. HAND con volumen conservado.** Shi et al. 2025 (*WRR*, doi:10.1029/2024wr037851), TOPFIM, CSI 0,83 contra HEC-RAS en ~10 s; Chlumsky, Craig y Tolson 2025 (*GMD*, doi:10.5194/gmd-18-3387-2025); Yu et al. 2025 (*JFRM*, doi:10.1111/jfr3.70057). Demanda: Copiapó, Chañaral y El Salado 2015 (Contreras et al. 2024, *ESPL*, doi:10.1002/esp.6017).

**P5. GLOF con incertidumbre.** Chen et al. 2025 (*HESS*, doi:10.5194/hess-29-733-2025); Veh et al. 2025 (*Nature Water*, doi:10.1038/s44221-025-00388-w): lagos represados por hielo producen diez veces más GLOF registrados y diez veces mayores. Antecedente chileno: Grosse, Patagonia (*STOTEN* 2025, doi:10.1016/j.scitotenv.2025.178368, solo título).

**P6. Post-incendio dNBR → probabilidad → volumen → inundación.** `pfdf` del USGS ya implementa Staley et al. 2017 y Gartner et al. 2014 en Python [V]. Falta integrar y recalibrar para Chile (Valparaíso 2024, megaincendios 2017 y 2023). Dunne et al. 2025 (*JGR-ES*, doi:10.1029/2025jf008325).

**P7. Unidades de pendiente.** `r.slopeunits` de GRASS existe; SUD-CGHP (*Appl. Sci.* 2025, doi:10.3390/app15094647) es código de un solo paper.

**Descartado por el agente.** Conectividad de sedimentos (SedInConnect existe; variantes RUSLE menores, doi:10.1016/j.catena.2025.109646); emulación neuronal de alcances (Nava et al. 2025, arXiv:2512.16221), fuera del alcance de DEM + imágenes.

**Demanda chilena verificada.** Kmetyko, Mergili y Palma 2024 (doi:10.5194/egusphere-egu24-9844) comparan r.avaflow con LAHARZ en ArcGIS para el Villarrica 2015. Ningún paper modela el aluvión de Taltal 2015 (OpenAlex y web sin resultados).

**Supuestos del agente.** Que el confinamiento desde HAND discrimina bien a 30 m en Taltal; que existen datos de depósitos de Cabré et al.

**Fuentes web.** https://pubs.usgs.gov/publication/tm14A3 · https://www.usgs.gov/software/grfin-tools · https://www.usgs.gov/programs/landslide-hazards/science/grfin-tools-methods-and-software-modeling-landslide-runout-and · https://ghsc.code-pages.usgs.gov/lhp/pfdf · https://www.usgs.gov/programs/landslide-hazards/science/postfire-debris-flow-hazard-assessment-software
