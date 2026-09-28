# SurtGIS — Estado del arte y frentes de vanguardia (septiembre 2026)

**Fecha**: 2026-09-28
**Base evaluada**: SurtGIS v1.4.0 + main @ 0638417 (105 algoritmos, 15 crates, WASM/npm, Python/PyPI, `surtgis serve` M0–M1.5, `surtgis-flow` v1.1 calibrado en Macul, `surtgis-ecw`, paper en EMS 2026).
**Método**: seis revisiones web independientes (formatos cloud-native y ecosistema Rust; GPU/WebGPU/WASM; IA geoespacial; hidrología y hazards; servidores y visores; incertidumbre, diferenciabilidad, DGGS y provenance), cada una con fuentes fechadas, y una síntesis cruzada. Los reportes íntegros van en los anexos A–F. Complementa `sota_review.md` (enero–febrero 2026), que fue una revisión de paridad algorítmica contra WhiteboxTools, SAGA y GRASS; esta revisión no busca paridad sino dónde adelantarse.

---

## 1. Resumen ejecutivo

La revisión de enero concluyó que SurtGIS estaba "a la par o adelante" en algoritmos. Nueve meses después, ese terreno dejó de ser diferenciador y tres hechos externos lo confirman:

1. **Whitebox Next Gen** (junio 2026) es una reescritura Rust modular de ~750 herramientas con LiDAR y vector nativos. La ventaja "motor Rust" ya no es exclusiva.
2. **opengeos** (Qiusheng Wu) atacó el navegador en tres frentes en 2026: `whitebox-wasm` (733 herramientas vía WASI), `cog-tiler-wasm` (tiles desde COG en el cliente) y **GeoLibre**, que el 23 de septiembre integró un plugin de embeddings (AlphaEarth, TESSERA, Clay). La ventaja "WASM" tampoco.
3. **TiTiler 2.x** ya hace "operador como parámetro de URL" para operadores de ventana y está sacando GDAL del camino caliente con crates Rust (`async-tiff`, `rio-tiler` 9). La ventaja "tiles dinámicos" es hoy paridad, no vanguardia.

Lo que ninguna de las seis revisiones encontró en ningún competidor, y que la arquitectura de SurtGIS ya deja a medio camino, se concentra en cinco frentes:

| # | Frente | Por qué nadie lo tiene | Ajuste con lo que ya existe |
|---|---|---|---|
| A | **Ejecución incierta nativa**: cada derivado con media, desviación y probabilidad por celda, Monte Carlo por construcción con RAM acotada | xDEM propaga solo a promedios; WBT solo depresiones; la literatura lo hace con scripts en R y 100 realizaciones | El streaming con presupuesto de RAM y los acumuladores por tile ya existen; falta el generador de campos y Welford por celda |
| B | **Provenance embebida y verificable** en cada archivo de salida + determinismo bit a bit multihilo como contrato | STAC `processing` no lleva hashes; PROV vive en catálogos, no en archivos; ningún motor GIS declara reproducibilidad multihilo | SurtGIS ya valida bit a bit contra GDAL; el tag TIFF y `surtgis verify` son semanas, no meses |
| C | **Híbrido cliente–servidor con un solo motor**: operadores locales en WASM desde COG remoto, operadores globales materializados por jobs, y OGC API Processes v2 sobre esos jobs | GeoLibre hace lo local, TiTiler lo dinámico, ZOO/pygeoapi los procesos; nadie une los tres con el mismo código | `serve` ya tiene jobs, tiles y cache; WASM ya tiene 56 algoritmos; falta el lector COG por rangos en WASM y el mapeo a Processes |
| D | **Embeddings de foundation models como capa de primera clase**: lectura de AlphaEarth/TESSERA como bandas, similitud y PCA por tile, features para susceptibilidad; luego inferencia de modelos tiny (5–44 M) en el binario | Los embeddings viven en GEE o Python+GPU; nadie los sirve ni infiere en Rust ni en navegador | STAC, COG multibanda, `formula` y `extract-patches` ya existen; `inference` tiene el andamiaje por tiles con halo y solo le falta el backend |
| E | **Solver de detritos de vanguardia**: bien puesto matemáticamente, ensambles bayesianos de μ–ξ con huella de referencia, y el mismo kernel en wgpu nativo y WebGPU | r.avaflow v4 sigue en CPU y calibra por prueba y error; los modelos two-phase populares tienen ill-posedness demostrada (JFM 2025); ningún solver de detritos corre en navegador | `surtgis-flow` ya tiene el kernel, la calibración de Macul y el criterio de reposo; GPU y ensambles son extensiones, no reescritura |

Dos frentes más son diferenciadores reales pero de nicho: **geomorfometría esferoidal sin reproyectar** (Florinsky/Guth, ningún motor la tiene; encaja con las 14 curvaturas ya implementadas) y **GeoZarr/Icechunk** (el I/O Rust ya está maduro con `zarrs` e `icechunk`, pero la convención `multiscales` sigue en v0.1 inestable).

Lo que **no** conviene perseguir como vanguardia: hidrología sobre DGGS (compite con un dataset ESSD 2025 ya publicado), solver diferenciable vía Enzyme (`std::autodiff` sigue experimental en nightly), JPEG XL en COG, "COG 2.0" (no existe), GeoPolars/DuckDB raster, DEM super-resolución (solo papers).

---

## 2. Qué cambió desde enero (hechos con fecha)

**Ecosistema Rust y cloud-native**
- Whitebox Next Gen anunciado 2026-06-27; su crate público `wbraster` 0.2.1 es solo I/O, los algoritmos no se publican como librería. No hay hoy ningún motor Rust publicado que combine algoritmos validados, streaming HTTP con presupuesto de RAM y bindings Python/WASM. `oxigeo` (75 crates, 814K LOC, 145 estrellas) tiene señales de volumen generado y no se considera validado.
- `zarrs` 0.23 (sharding, async, WASM), `icechunk` 2.2.2 (crate Rust con API pública), `async-tiff` Rust 0.4 (2026-09-18), `geoarrow-rs` 0.9, `rustac` 0.2.16. kerchunk deprecado; VirtualiZarr exige Icechunk 2.
- GeoZarr es ahora tres convenciones (`proj`, `spatial`, `multiscales`); `multiscales` v0.1 con cambios rompedores esperados antes de fin de 2026. Ninguna implementación Rust. GDAL 3.13 lee y escribe; GDAL 3.14 (noviembre 2026) tendrá Icechunk solo lectura.

**GPU, WebGPU y WASM**
- WebGPU: Chrome, Safari 26 y Firefox en Windows/macOS. En Linux, Chrome 147 solo con NVIDIA ≥535 y Wayland; Firefox Linux sigue en Nightly. wgpu v30. Wasm 3.0 (2025-09-17) oficializa memory64, pero wasm-bindgen rechaza threads + memory64: hoy se elige Rayon (<4 GB) o >4 GB, no ambos.
- cuSpatial archivado (2025-07-28). FastFlow (Pacific Graphics 2024) lleva flow routing y depression routing a GPU; priority-flood exacto sigue en CPU. r.avaflow v4 (GMD 2025-12) sigue en CPU.
- WGSL no garantiza bit a bit CPU↔GPU (FMA, subnormales, sin/cos con error 2⁻¹¹). Cualquier puerto GPU debe publicar tolerancias, no identidad.

**IA geoespacial**
- Modelos tiny con pesos abiertos: Prithvi-EO-2.0 tiny (5 M), TerraMind tiny (5 M), TESSERA v2 (44 M, 16-d Matryoshka retienen 92 %). AlphaEarth sigue cerrado, pero sus embeddings 64-d están en GCS como COG y en Source Cooperative. Ninguno publica ONNX oficial.
- `ort` 2.0.0-rc.13, `tract` (WASM, producción), `burn-onnx` con panics en WebGPU. Nadie infiere embeddings satelitales en Rust ni en navegador.
- GISAgentBench (agosto 2026): el mejor modelo resuelve 32,7 % de 349 tareas reales; la desalineación de CRS es la falla más costosa. Los contratos tipados (CRS, unidades, nodata, fallos tipados) son el diseño que funciona.

**Hidrología y hazards**
- TopoToolbox 3 (preprint mayo 2026): núcleo C compartido, GraphFlood (SWE aproximadas sobre DAG) y property-based testing. GRASS 8.5 (mayo 2026) suma `r.hand`, `r.slopeunits`, `r.lfp`, `r.hydroflatten`. NOAA FIM 4 cubre casi toda la población de EE. UU. con HAND.
- Langham et al. (JFM 2025) demuestran ill-posedness en Pitman-Le y Pudasaini; "two-phase" sin términos difusivos es riesgo, no vanguardia. Kestrel (JOSS 2024) es la referencia two-phase con erosión.
- xDEM v0.2 es el referente de incertidumbre de DEM (heterocedasticidad + variograma), pero propaga solo a promedios. GEDTM30 y FathomDEM renuevan los DTM globales.

**Servidores y visores**
- TiTiler 2.4.0 (2026-09-21), rio-tiler 9.4 con readers async Zarr y GeoZarr. xpublish-tiles sirve Zarr/Icechunk con coarsening dinámico. Martin tiene COG solo bajo feature inestable, sin 3857. GeoServer 3.0 no aporta a tiles dinámicos.
- OGC API Tiles 1.0 es ISO 19177-1:2026; Processes v2 (Part 1+2) cerró comentario público el 2026-09-05; Part 3 ("collection output", el request de un tile dispara el proceso) sigue borrador. DGGS Part 1 aprobado con casi cero implementaciones.
- MapLibre 6.x tiene hillshade y color-relief nativos, sin WebGPU. deck.gl 9.4 mantiene WebGPU experimental.

**Ciencia reproducible**
- SERRA (agosto 2026): 100 realizaciones de error de DEM suben el NSE de profundidad de inundación de 0,18 a 0,72. Sigue siendo script ad hoc.
- STAC `processing` v1.2.0 no lleva hashes de inputs ni parámetros. GeoPROV (IJGI 2026) opera a nivel de catálogo.
- Yang et al. (arXiv 2026-09-10): el orden de reducción es la causa primaria de no determinismo; orden fijo cuesta ~20 %.

---

## 3. Los cinco frentes, en detalle

### A. Ejecución incierta nativa

**Qué es.** Un flag `--uncertainty rmse=<m>,range=<m>,n=<N>` (o su equivalente en Python y en el server) que convierte cualquier derivado en tres salidas: media, desviación estándar y, para productos binarios (cauce, depresión, alcance), probabilidad por celda. El campo de error se genera por simulación gaussiana secuencial o turning bands con autocorrelación espacial, y las N realizaciones se acumulan por celda con Welford, de modo que la RAM es la de una realización más dos acumuladores.

**Por qué es vanguardia.** El método es de 1997 y sigue vigente; la novedad es ofrecerlo como primitiva del motor y no como script. WBT lo tiene solo para depresiones; xDEM estima el error pero remite a GSTools para propagarlo; GRASS da la superficie aleatoria y deja el loop al usuario. Ningún motor devuelve `slope_sd` o `P(cauce)` de fábrica.

**Ajuste.** El streaming por tiles con presupuesto de RAM (R9) y el modelo de error de xDEM (heterocedástico + variograma) son compatibles; el variograma puede venir del módulo de kriging ya existente. El costo N× se acota con el `MemoryBudget`.

**Riesgos.** Comunicar bien el modelo de error (RMSE global vs heterocedástico); validar el generador contra gstat/GSTools; en flow accumulation el costo es N ejecuciones completas.

**Entregable publicable.** Paper metodológico corto (C&G o EMS): "uncertainty-aware geomorphometry by construction", con el caso de HAND/FIM y el de susceptibilidad.

### B. Provenance embebida + determinismo como contrato

**Qué es.** Cada GeoTIFF/COG que SurtGIS escribe lleva un tag TIFF privado con: hash BLAKE3 de cada input, parámetros efectivos, versión del motor, semilla, número de hilos y hash del resultado. `surtgis verify <archivo>` recomputa y compara. Exportable a STAC `processing` y PROV-O. Los jobs del server emiten además un item STAC por producto.

**Determinismo.** Fijar el orden de las reducciones globales en Rayon y añadir un test de CI que exija igualdad bit a bit entre 1 y N hilos para cada algoritmo. Donde el orden fijo cueste, documentarlo como excepción.

**Por qué es vanguardia.** STAC no lleva hashes; PROV vive en catálogos; nadie en GIS declara reproducibilidad multihilo verificada. Es barato y convierte lo que SurtGIS ya hace (validación bit a bit contra GDAL) en un contrato visible.

**Riesgos.** Definir un esquema estable del tag antes de publicarlo; ~20 % de costo en reducciones globales.

### C. Híbrido cliente–servidor con un solo motor

**Qué es.** El visor decide por operador: pendiente, hillshade, curvaturas e índices se calculan en el navegador (WASM) leyendo el COG remoto por rangos; fill, acumulación, HAND y TWI se piden al server como job y vuelven como TileJSON. Los jobs se exponen además como OGC API Processes v2 (`/processes`, `/jobs/{id}/results`), y el server ofrece tiles f32 de datos (no PNG) para que el cliente siga analizando.

**Por qué es vanguardia.** GeoLibre y cog-tiler-wasm hacen lo local; TiTiler lo dinámico; ZOO y pygeoapi los procesos; OGC Processes Part 3 describe en borrador la idea de que un tile dispare un proceso. Nadie une las tres capas con el mismo código.

**Ajuste.** `serve` ya tiene jobs, cache, allowlist y métricas; WASM ya tiene 56 algoritmos y demo 3D. Faltan: lector COG por rangos dentro de WASM, política de decisión local/global, mapeo de `POST /jobs` a Processes v2 y el formato de tiles f32 con metadatos.

**Riesgos.** Límite de 4 GiB con Rayon en WASM; COOP/COEP para threads; Processes v2 puede cambiar tras el comentario público; opengeos itera rápido.

**Prerrequisito.** El benchmark M3 pendiente, hecho con la plantilla de tile-benchmarking de DevSeed, contra TiTiler 2.4, xpublish-tiles y tileserver-rs.

### D. Embeddings como capa de primera clase

**Fase 1 (baja dificultad, alto impacto visible).** Leer AlphaEarth (COG en GCS), TESSERA y Major TOM (GeoParquet) como fuentes multibanda; servir similitud coseno y PCA por tile con `formula` sobre 64 bandas; agregar embeddings como features en `extract-patches` y `predict_raster` y replicar en Chile el +0,04–0,11 AUC en susceptibilidad reportado en 2026.

**Fase 2 (media-alta).** Inferencia de modelos tiny (Prithvi/TerraMind 5 M, TESSERA 44 M) con `tract` en el binario y en WASM, usando el andamiaje `inference::run_tiled` que ya existe. Exige exportar ViT a ONNX y verificar paridad con la normalización de bandas.

**Por qué es vanguardia.** "AlphaEarth local, abierto y sin cuenta Google" no existe; nadie infiere embeddings satelitales en Rust ni en navegador. La deuda técnica de los productos de embeddings (tiling propio, sin CRS) es una oportunidad para un lector que los normalice.

**Riesgos.** Volumen (decenas de TB por continente); cambios de tiling upstream; sin ONNX oficial.

**Adyacente.** Contratos tipados para las 42 herramientas MCP (CRS, unidades, nodata, fallos tipados, prevalidación) y un mini-benchmark de terreno/hidrología con datos ajenos: es exactamente donde GISAgentBench muestra que fallan los agentes.

### E. Solver de detritos de vanguardia

Tres extensiones escalonadas sobre `surtgis-flow`:

1. **Ensamble bayesiano ligero** (LHS o ABC sobre μ–ξ–entrainment contra huella IoU): mapas de probabilidad de alcance; costo controlado; cierra el N4 de Macul con posterior en vez de un par de valores.
2. **Posedness explícita**: documentar y testear que el modelo monofásico con Voellmy y entrainment acotado está bien puesto; si se agrega segunda fase, hacerlo con los términos difusivos de Langham et al. (2025), no con Pudasaini crudo.
3. **Kernel en wgpu**: el mismo WGSL en GPU nativa y en WebGPU; publicar tolerancias (masa, alcance, frente) respecto de la referencia CPU. Sería el primer solver de remoción en masa que corre en el navegador.

**Por qué es vanguardia.** r.avaflow v4 y RAMMS siguen en CPU y calibran a mano; los emuladores neuronales (FNO, 2026) solo se validan contra el solver que los entrenó; el solver diferenciable (Inunda, 2026) existe para agua, no detritos, y en Rust depende de Enzyme experimental.

**Riesgos.** Sin bit a bit CPU↔GPU; Linux/Firefox parcial en WebGPU; validación reducida a un caso hasta conseguir más huellas.

---

## 4. Matriz consolidada

| Frente | Dificultad | Impacto | Riesgo | Costo estimado | Dependencias |
|---|---|---|---|---|---|
| B. Provenance + determinismo | Baja | Alto (credibilidad) | Bajo | 2–4 semanas | Ninguna |
| D1. Embeddings como bandas + features | Baja-media | Alto (visible) | Medio (volumen) | 3–6 semanas | STAC/COG existentes |
| Benchmark M3 (prerrequisito de C) | Baja | Alto (paper) | Resultado desfavorable en COG frío | 1–2 semanas, máquina tranquila | Ninguna |
| A. Ejecución incierta | Media | Muy alto (científico) | Costo N× | 2–3 meses | Kriging/variograma, R9 |
| C. Híbrido + Processes v2 + tiles f32 | Media-alta | Muy alto (tesis analysis-first) | opengeos, límite 4 GiB | 3–4 meses | M3, WASM threads |
| E1. Ensamble bayesiano en flow | Media | Alto | Cómputo | 1–2 meses | N4 Macul |
| Geomorfometría esferoidal | Baja-media | Medio-alto (DEM globales) | Validación vs GEE | 3–5 semanas | Curvaturas Florinsky |
| D2. Inferencia tiny en binario/WASM | Media-alta | Muy alto | Sin ONNX oficial | 2–3 meses | `inference` + tract |
| E3. Kernel wgpu/WebGPU | Alta | Alto | Sin bit a bit | 3–4 meses | wgpu, GPU Linux |
| GeoZarr sharded lectura + multiscales escritura | Media | Medio-alto | Spec v0.1 inestable | 1–2 meses, esperar v1 | zarrs |
| Icechunk escritura desde jobs | Media-alta | Alto | API sin estabilidad declarada | Esperar GDAL 3.14 | icechunk 2.x |
| E2. Solver diferenciable | Muy alta | Muy alto (paper) | Enzyme experimental | No en 2026 | E1 primero |
| Hidrología DGGS | Media-alta | Medio | Nicho, ESSD 2025 ya publicado | Descartado | — |

---

## 5. Secuencia sugerida

**Q4 2026 (bajo costo, alta señal).** B completo; D1 (lector AlphaEarth/TESSERA + similitud por tile + features); benchmark M3 cuando la máquina esté libre; nota de software del server. Resultado: el motor declara contratos que nadie más declara y muestra GeoAI sin GPU.

**H1 2027 (científico).** A como paper metodológico con HAND/FIM y susceptibilidad; E1 cerrando Macul con posterior; geomorfometría esferoidal si aparece un usuario de DEM global.

**H2 2027 (plataforma).** C completo, apoyado en el benchmark; D2 si tract soporta el ViT exportado; E3 si WebGPU en Linux/Firefox se estabiliza.

**Reevaluar en 2027.** GeoZarr `multiscales` v1, Icechunk en GDAL 3.14, `std::autodiff` estable, Processes v2 aprobado.

---

## 6. Límites de esta revisión

- Cada agente agotó su presupuesto de búsqueda; quedaron sin verificar: priority-flood sobre COG streaming en terceros, TWI multiescala, Iber+/GLOF, releases de TauDEM 2024–2026, tres consultas menores de UQ.
- MDPI y ScienceDirect devolvieron 403; esas afirmaciones se apoyan en snippets.
- Las cifras de rendimiento de tileserver-rs, xpublish-tiles y oxigeo son de README o blog, no de benchmark reproducible.
- No se leyó código de terceros; "no lo tiene" significa "no aparece documentado ni publicado", no "es imposible que exista".


---

# Anexos: reportes por frente

Cada anexo es el reporte íntegro del agente de investigación correspondiente, con sus fuentes. Se demota un nivel de encabezado para encajar en este documento.


## Anexo A. Formatos cloud-native y ecosistema Rust

Nota metodológica: todo lo de abajo viene de la web (búsquedas y fetch directo de repos, changelogs y blogs). Donde la fuente es un anuncio y no un release, se indica.

### 1. Zarr, GeoZarr, Icechunk, VirtualiZarr, Xarray

**GeoZarr** dejó de ser una spec monolítica: ahora son tres convenciones componibles (`proj`, `spatial`, `multiscales`) en la organización `zarr-conventions`, con un OGC SWG formado y meta de revisión por el Architecture Board "summer 2026" ([geozarr.org](https://geozarr.org/), consultado 2026-09-28). Ojo con la madurez: la convención `multiscales` está en **v0.1 "pre-stable"** con cambios rompedores esperados antes de v1 "antes de fin de 2026" ([zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales)). Implementaciones listadas ([geozarr.org/implementations](https://geozarr.org/implementations)): GDAL (lectura/escritura completa desde 3.13), eopf data-model, zarr-cm, rioxarray (solo lectura), TiTiler-EOPF, plugin QGIS GeoZarr 0.2.0 (2026-03-05, exige GDAL 3.13+ para sharded), OpenLayers, zarr-layer, deck.gl-raster. **Ninguna implementación en Rust.** El caso de producción real es el EOPF Sentinel Zarr Explorer de ESA (lanzado 2026-02-13, Development Seed + EOX; [blog](https://developmentseed.org/blog/2026-02-13-eopf-explorer-launch/)).

**Zarr v3 / sharding**: zarr-python 3.4.0 salió el 2026-09-15 (mejoras al codec de sharding, arrays numpy como shards) ([releases](https://github.com/zarr-developers/zarr-python/releases)); xarray 2026.04.0 trae lectura sharded y DataTree oficial ([whats-new](https://docs.xarray.dev/en/v2026.04.0/whats-new.html)), pero escribir shards con Dask sigue obligando a que los chunks de Dask coincidan con los shards ([discusión #9938](https://github.com/pydata/xarray/discussions/9938)). Extensiones nuevas: chunks de largo variable (Earthmover, 2026-05-05) y chunks rectilíneos experimentales. GDAL: sharding lectura/escritura desde 3.13, decodificación multihilo, `ZARR_V3` por defecto desde 3.14 ([driver Zarr](https://gdal.org/en/latest/drivers/raster/zarr.html)).

**Icechunk**: v2 anunciado 2026-04-09; crate Rust `icechunk` **2.2.2 (2026-09-17)** con API pública `Repository`/`Session`/`Store` ([docs.rs](https://docs.rs/icechunk/latest/icechunk/)); adoptado por el NWS para la plataforma CIRRUS (2026-06-04, [blog](https://icechunk.io/en/latest/blog-posts/)). En GDAL solo hay lector `/vsiicechunk/` **read-only**, milestoneado para 3.14 en noviembre 2026, "no en release oficial todavía" ([hypertidy, 2026-06-26](https://www.hypertidy.org/posts/2026-06-26_icechunk-is-coming/)). Frente nuevo: "hybrid stores" (referencias virtuales + pirámides GeoZarr) renderizados en el navegador con icechunk-js + zarr-layer + topozarr ([CNG, 2026-08-10](https://cloudnativegeo.org/blog/2026/08/virtual-icechunk-multiscale/)); Earthmover hace lo mismo en Arraylake/Flux (2026-06-16, [blog](https://www.earthmover.io/blog/multiscales-in-al/)). Todo Python/JS.

**VirtualiZarr** 2.6.2 (2026-05-18) exige icechunk 2.x; **kerchunk está oficialmente deprecado** (fsspec/kerchunk PR #589; [issue #1099, 2026-09-16](https://github.com/zarr-developers/VirtualiZarr/issues/1099)). GDAL lee referencias kerchunk JSON/Parquet desde 3.11. **Cubed** existe como reemplazo serverless de Dask con memoria acotada, pero ~4x más lento en 1.5 TB ([xarray blog](https://xarray.dev/blog/cubed-xarray)); es nicho.

**Rust**: `zarrs` 0.23.14 (2026-08-15): sharding siempre disponible, API async, metadata consolidada v3, zero-copy ([changelog](https://github.com/zarrs/zarrs/blob/main/CHANGELOG.md)). Development Seed lanzó **zarrista** (2026-08-13), Python sobre zarrs, 1.9–2.7x más rápido que zarr-python ([blog](https://developmentseed.org/zarrista/latest/blog/archive/2026/)). Es decir, el I/O Zarr en Rust ya está maduro; lo que falta es la capa geoespacial encima.

### 2. COG y GeoTIFF

No existe un "COG 2.0": el estándar OGC sigue en 1.0 (julio 2023, [OGC 21-026](https://docs.ogc.org/is/21-026/21-026.html)). El "multi-resolution" nuevo se está resolviendo en GeoZarr multiscales, no en TIFF. GDAL 3.13.0 (2026-05-08) añadió `Create()` con escritura aleatoria para COG y `gdal raster … validate` para COG ([releases](https://github.com/OSGeo/gdal/releases)); 3.13.3 es de 2026-08-18. JPEG XL sigue restringido a libtiff interno + libjxl y ≤4 bandas; LERC y ZSTD son estándar ([driver COG](https://gdal.org/en/stable/drivers/raster/cog.html)).

Lectores Rust asíncronos: **async-tiff** (Development Seed): Python 0.7.2 (mayo 2026), Rust 0.4 (PR mergeado 2026-09-18); JPEG, JPEG2000, ZSTD, LERC(+deflate/zstd), Deflate, LZW, WebP, LZMA; `object_store`; caché read-ahead exponencial ([changelog](https://developmentseed.org/async-tiff/latest/CHANGELOG/)). Es solo lector, tiled, sin análisis. **cog3pio** 0.1.0 (2026-07-27): CPU + CUDA vía nvtiff ([docs.rs](https://docs.rs/cog3pio/latest/cog3pio/)). geotiff.js 3.0.5 con 3.1.0-beta ([npm](https://www.npmjs.com/package/geotiff)). Lo que ninguno de estos ofrece es presupuesto de RAM ni overviews de análisis: son decodificadores.

### 3. GeoParquet 2.0 / GeoArrow y motores raster en Rust

Parquet adoptó tipos lógicos GEOMETRY/GEOGRAPHY en 2025; GeoParquet 2.0.0 se apoya en ellos y los soportan parquet-java, arrow-cpp, **arrow-rs**, DuckDB, hyparquet ([parquet.apache.org, 2026-02-13](https://parquet.apache.org/blog/2026/02/13/native-geospatial-types-in-apache-parquet/)). Overture publica en GeoParquet (release 2026-09-23.1, [docs](https://docs.overturemaps.org/getting-data/duckdb/)). **geoarrow-rs** rust-v0.9.0 (2026-09-11), py-v0.6.3 con wheels Emscripten (2026-06-11) ([releases](https://github.com/geoarrow/geoarrow-rs/releases)); geozero 0.15.1; crate `gdal` 0.18.

¿Motor raster Rust con API de librería comparable? Escéptico:
- **WhiteboxTools** 1.5.0 es de 2021; la continuación es **Whitebox Next Gen** (700+ herramientas, open core `wbtools_oss`, MIT/Apache) ([repo](https://github.com/jblindsay/whitebox_next_gen)). Su crate público `wbraster` 0.2.1 (2026-07-30) es **solo I/O** ("not a raster processing or analysis library"), Zarr solo local ([lib.rs](https://lib.rs/crates/wbraster)). Los algoritmos no están publicados como crate.
- **RichDEM** original sin mantención; fork `richdem2` (giswqs) en conda 2.4.3 (2026-05-20) ([repo](https://github.com/giswqs/richdem2)).
- **DuckDB**: extensión comunitaria `raster` v1.0.0, basada en GDAL ([duckdb.org](https://duckdb.org/community_extensions/extensions/raster)); **RaQuet** (CARTO): raster en Parquet con celdas QUADBIN, DuckDB/BigQuery/Snowflake ([repo](https://github.com/CartoDB/raquet)). Álgebra de bandas, no terreno.
- **GeoPolars**: desbloqueado en nov 2025 por extension types de Polars, sigue siendo "prototype, not production-ready" ([repo](https://github.com/geopolars/geopolars)).
- **oxigeo** 0.2.4 (2026-08-18) declara 75 crates, 814K LOC, COG por HTTP, Zarr v3 sharded, hillshade/watershed; 145 estrellas, sin adopción verificable ([repo](https://github.com/cool-japan/oxigeo)). Sospecha de volumen generado; tratar como no validado. `geonative-*` es otra familia pure-Rust incipiente ([docs.rs](https://docs.rs/geonative-geotiff/latest/geonative_geotiff/)).

Conclusión: no hay hoy un motor Rust publicado que combine algoritmos de terreno validados, streaming HTTP con presupuesto de RAM y bindings Python/WASM. La competencia real sigue siendo GDAL + xarray/rioxarray en Python.

### 4. STAC

STAC 1.1.0 y STAC API 1.0.0 son OGC Community Standards (docs 25-004 y 25-005; [ogc.org](https://www.ogc.org/standards/stac/)). **stac-geoparquet** se separó a `radiantearth/stac-geoparquet-spec` en octubre 2025 y "no ha sido liberado como v1 estable"; 1.1.0 deprecó `collection` por `collections` ([spec](https://radiantearth.github.io/stac-geoparquet-spec/latest/)). La tesis "STAC sin API" (Development Seed, [Right-sizing STAC, 2025-05-07](https://developmentseed.org/blog/2025-05-07-stac-geoparquet/)) recomienda geoparquet + rustac para catálogos pequeños y medianos. **rustac** 0.2.16 (2026-09-04) con `stac` 0.17.6, `stac-duckdb`, `stac-server` ([lib.rs](https://lib.rs/crates/rustac)). **pgstac** eliminó pypgstac y lo reemplazó por `pgstac-migrate` + un CLI en Rust; stac-fastapi-pgstac v7 (septiembre 2026) ([release notes](https://stac-utils.github.io/pgstac/release-notes/)).

### 5. Oportunidades para adelantarse (no solo igualar)

1. **Primer motor de terreno que lee GeoZarr sharded por HTTP con presupuesto de RAM.** Hoy: GDAL 3.13 (C++) y rioxarray leen conventions; en Rust solo `zarrs` (I/O). Nadie hace slope/hillshade/hidrología directo sobre EOPF/GeoZarr sharded con RAM acotada. Dificultad media: zarrs resuelve sharding y consolidated metadata; falta parsear `proj`/`spatial`/`multiscales` (v0.1, inestable) y elegir nivel de pirámide según ventana.
2. **Escritura de salidas como GeoZarr multiscales** (equivalente Rust de topozarr/ndpyramid). Hoy: topozarr, ndpyramid, GDAL BuildOverviews, todo Python/C++. Con esto un DEM derivado se visualiza en OpenLayers/zarr-layer sin servidor de tiles. Dificultad media.
3. **Escribir Icechunk nativamente desde el motor (jobs transaccionales con snapshots).** Hoy: Icechunk es Rust pero solo se escribe desde Python/xarray; GDAL será read-only en 3.14 (noviembre 2026). Un job de `surtgis serve` que hace commit de un derivado versionado sería el primer motor de análisis en escribirlo sin Python. Dificultad media-alta: la API del crate no declara estabilidad y hay módulo de migraciones de formato.
4. **Tiler dinámico Rust sobre Zarr/Icechunk con OGC Tiles + multiscales.** Hoy: xpublish-tiles (Python, sin soporte multiscale al 2025-09-29, [blog](https://www.earthmover.io/blog/dynamic-map-tile-rendering-icechunk-zarr-data-xpublish-tiles)), TiTiler-xarray, Flux (propietario). `surtgis serve` ya tiene XYZ+TileJSON sobre COG; extenderlo a GeoZarr con fórmula ASI sería único. Dificultad media.
5. **Resolver referencias virtuales (VirtualiZarr manifests / Icechunk virtual chunks) en Rust.** Hoy: solo Python; GDAL lee refs kerchunk (deprecado). SurtGIS ya decodifica NetCDF/GRIB/TIFF, así que puede materializar chunks virtuales sin duplicar datos. Dificultad alta (formato de manifest en evolución con Icechunk 2).
6. **Análisis en el navegador sobre datos crudos GeoZarr/Icechunk (WASM).** La visión "hybrid stores" del CNG streamea datos, no imágenes; icechunk-js y zarr-layer solo pintan. zarrs compila a WASM (desde 0.22). SurtGIS ya tiene binding npm: sería el primero con hillshade/pendiente/índices sobre Zarr client-side. Dificultad media-alta (tamaño del bundle, fetch por rangos en navegador).
7. **stac-geoparquet como catálogo directo del motor.** Hoy: rustac + DuckDB. SurtGIS lee GeoParquet; consumir stac-geoparquet sin API ni DuckDB para armar composites es barato y encaja con "right-sizing STAC". Dificultad baja; impacto medio (la spec no es v1).

| Oportunidad | Estado del arte hoy | Quién | Dificultad | Impacto |
|---|---|---|---|---|
| Lector GeoZarr sharded HTTP + RAM acotada en motor de terreno | GDAL 3.13, rioxarray, TiTiler-EOPF (C++/Python); en Rust solo I/O (zarrs) | ESA/EOPF, DevSeed, GDAL | Media | Alto |
| Escritura GeoZarr multiscales | topozarr, ndpyramid, GDAL BuildOverviews | CarbonPlan, Earthmover | Media | Alto |
| Escritura Icechunk nativa (jobs versionados) | Solo Python; GDAL read-only en 3.14 (nov 2026) | Earthmover, NWS | Media-alta | Alto |
| Tiler Rust Zarr/Icechunk con multiscales + fórmulas | xpublish-tiles, TiTiler-xarray (Python), Flux (propietario) | Earthmover, DevSeed | Media | Alto |
| Chunks virtuales (VirtualiZarr/Icechunk) en Rust | Solo Python; kerchunk deprecado | zarr-developers, Earthmover | Alta | Medio-alto |
| Análisis WASM sobre GeoZarr/Icechunk crudo | icechunk-js, zarr-layer solo visualizan | CNG, CarbonPlan | Media-alta | Alto (diferenciador) |
| stac-geoparquet directo sin API/DuckDB | rustac + stac-duckdb | Development Seed, Radiant Earth | Baja | Medio |

**Lo que no se recomienda perseguir como vanguardia**: JPEG XL en COG (sigue nicho, ≤4 bandas, sin encoder pure-Rust maduro), "COG 2.0" (no existe) y GeoPolars/DuckDB raster (no compiten en terreno).


## Anexo B. GPU, WebGPU y WebAssembly

### Estado del arte: GPU, WebGPU y WebAssembly para análisis raster/DEM (septiembre 2026)

#### 1. WebGPU en el navegador

**Soporte.** Chrome lo envía desde la 113 (abril 2023; Android 121+; Linux recién: Intel Gen12+ en 144, NVIDIA en 147 solo con driver ≥535 y Wayland). Firefox: Windows en 141 (julio 2025), macOS Apple Silicon en 145/147 (enero 2026); **Linux y Android siguen en Nightly**, "esperado 2026". Safari 26 (septiembre 2025) en macOS/iOS/iPadOS/visionOS. Fuente: [gpuweb Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [web.dev, nov. 2025](https://web.dev/blog/webgpu-supported-major-browsers). Lectura escéptica: "todos los navegadores" es cierto para Windows/macOS/iOS; en Linux la cobertura es parcial y dependiente del driver.

**wgpu/WGSL.** wgpu está en v30 (v29 marzo 2026, v30 julio 2026, v30.0.1 agosto 2026; [changelog](https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md)). Es la implementación de Firefox, así que un shader WGSL corriendo en wgpu nativo tiene alta probabilidad de correr igual en Firefox; Chrome usa Dawn (otro compilador), fuente de diferencias sutiles.

**Geoprocesamiento raster con WebGPU en cliente (lo que existe):**
- [deluge-flood-sim](https://github.com/rkottomt/deluge-flood-sim): SWE local-inercial sobre DEM USGS, 1024² a 60 fps (~100× tiempo real), con dam-break analítico, conservación de masa ~1e-7 y convergencia de malla. MIT, autor anónimo, sin paper. Es lo más cercano a "hidrodinámica científica en WebGPU", pero es un proyecto personal.
- [WebGPU-Erosion-Simulation](https://github.com/GPU-Gang/WebGPU-Erosion-Simulation) (stream power + área de drenaje aproximada en paralelo) y [webgpu-shallow-water](https://github.com/lisyarus/webgpu-shallow-water) (virtual pipes): gráfica, no ciencia.
- [Waveshed](https://waveshed.io/) (viewshed/cobertura RF con WebGPU y fallback CPU) y [viewshed-lab](https://github.com/johnny-salz/viewshed-lab): viewshed en compute shader, nicho RF.
- Visualización: [deck.gl 9.4 (5 sep. 2026)](https://deck.gl/docs/whats-new) mantiene WebGPU "experimental, no para producción"; [deck.gl-raster](https://developmentseed.org/deck.gl-raster/docs/intro/) hace band math cliente pero en **WebGL2**; MapLibre y Cesium siguen en WebGL2 con backends WebGPU en desarrollo o forks comunitarios.

**Conclusión:** nadie hace análisis de terreno científico (hillshade + hidrología + viewshed validado) con WebGPU en cliente como producto; hay demos aisladas de una rutina cada una.

#### 2. WebAssembly

[Wasm 3.0 (17 sep. 2025)](https://webassembly.org/news/2025-09-17-wasm-3.0/) oficializa memory64, multi-memory, GC, relaxed SIMD y tail calls; navegadores limitan memory64 a ~16 GB. Threads siguen vía SharedArrayBuffer (exige COOP/COEP). **Gotcha central para SurtGIS:** la transformación de threads de wasm-bindgen rechaza memory64 ([issue #5330](https://github.com/wasm-bindgen/wasm-bindgen/issues/5330)), así que hoy eliges Rayon (wasm-bindgen-rayon, <4 GB) **o** >4 GB, no ambos.

GIS en WASM: qgis-js ya no requiere parches y `wasm32-emscripten` es target oficial de QGIS ([blog QGIS, mar. 2026](https://blog.qgis.org/2026/03/15/reports-from-the-winning-grant-proposals-2025/)); Processing en el navegador está en "exploración". [gdal3.js](https://github.com/bugra9/gdal3.js/) porta utilidades (translate/warp/ogr2ogr), no algoritmos de terreno. geotiff.js está en serie 3.x. DuckDB-Wasm carga `spatial` (vector). Competencia directa nueva: [whitebox-wasm](https://github.com/opengeos/whitebox-wasm) (opengeos, Rust puro, 733 herramientas pero solo un subconjunto corre en navegador, el resto exige WASI; sin threads ni GPU documentados; límite 4 GiB) y [terrano](https://github.com/GeoLang/terrano) (Rust/WASM, AGPL, 49 commits, hillshade/D8/watershed/viewshed). Ninguno tiene GPU ni paralelismo en cliente. "Terreno completo sin servidor" está a distancia de RAM (4 GB) y de un pipeline COG→WASM→GPU integrado, no de algoritmos.

#### 3. GPU nativo en ciencia del terreno

- cuSpatial fue **archivado el 28 jul. 2025** ([RSN 45](https://docs.nvidia.com/datascience/notices/rsn0045/index.html)); era vectorial, no raster. No hay "RAPIDS para DEM".
- Hidrología: [FastFlow (Pacific Graphics 2024, best paper)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15243) resuelve flow routing en O(log n) iteraciones y depression routing en O(log² n): 5× y 34-52× sobre trabajo previo GPU en 1024². [Kotyra 2025, C&G](https://www.sciencedirect.com/science/article/pii/S0098300425001116) longest flow path en GPU. Priority-flood exacto sigue siendo secuencial/CPU; en GPU se usan variantes iterativas por convergencia.
- Hidrodinámica: [LISFLOOD-FP 8.2 (GMD 2025)](https://gmd.copernicus.org/articles/18/9827/2025/) da 2.5-4× vs 16 cores solo con >1M celdas; [SERGHEI](https://gmd.copernicus.org/articles/16/977/2023/) (Kokkos) escala a 1024 GPUs ([arXiv 2511.01001, nov. 2025](https://arxiv.org/abs/2511.01001)), memoria-bound; TRITON multi-GPU; [CaMa-Flood-GPU (GMD 2026)](https://gmd.copernicus.org/articles/19/5623/2026/). Detritos: [r.avaflow v4 (GMD, sep. 2025)](https://gmd.copernicus.org/articles/18/9879/2025/) **sigue en CPU**; RAMMS sin GPU; hay GPU en [MoSES_2PDF](https://arxiv.org/pdf/2104.06784) y en EST (Martínez-Aranda, Eng. Geol. 2021). Ningún solver de detritos corre en navegador.
- Viewshed: ganancias GPU maduras desde 2011; lo nuevo (IJDE 2024, multi-viewpoint) suma solo 6-9%. Viewshed total/acumulado sí escala en multi-GPU (arXiv 2003.02200).
- Radiación: [HORAYZON](https://github.com/ChristianSteger/HORAYZON) es ray tracing **CPU** (Embree); [RS 2026](https://doi.org/10.3390/rs18173044) hace irradiancia sobre DSM en GPU con viewsheds comprimidos.

**Regla empírica:** ganan en GPU los stencils (hillshade, slope, curvaturas, focal), SWE explícitas, viewshed y horizontes; siguen en CPU priority-flood exacto, D-inf con orden topológico, streams/Strahler y todo lo que pide fillado exacto o bitwise.

#### 4. Compute portable

wgpu: un WGSL corre en Vulkan/Metal/DX12 nativo y en WebGPU vía WASM. [rust-gpu](https://github.com/Rust-GPU/rust-gpu) (EmbarkStudios archivado 31 oct. 2025, continúa en org Rust-GPU) es riesgo alto; [CUDA Rust](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/) (NVIDIA, sep. 2026) es solo NVIDIA. [CubeCL](https://github.com/tracel-ai/cubecl) (alpha, usado por Burn) compila un kernel Rust a wgpu/CUDA/HIP/CPU-SIMD y corre en navegador; es el **único precedente** de "misma rutina en CPU, GPU nativa y WebGPU desde un Rust", pero en ML. [Mojo 1.0 (ago. 2026)](https://arxiv.org/abs/2509.21039) es portable NVIDIA/AMD, sin navegador. En geociencia el portable es Kokkos (SERGHEI), sin ruta web.

**Riesgo numérico:** WGSL declara la reasociación/fusión FMA un "portability hazard", permite flush de subnormales y acota división a 2.5 ULP y sin/cos a error absoluto 2⁻¹¹ ([WGSL §15.7](https://www.w3.org/TR/WGSL/#floating-point-evaluation)); reproducibilidad bit a bit CPU↔GPU no es alcanzable sin reducciones ordenadas y sin funciones transcendentales propias ([arXiv 2609.11356](https://arxiv.org/abs/2609.11356)).

#### 5. Oportunidades para SurtGIS

| Oportunidad | Estado del arte hoy | Quién | Dificultad | Impacto | Riesgo |
|---|---|---|---|---|---|
| **Solver de detritos (SWE+Voellmy) idéntico en wgpu nativo y WebGPU** | Detritos GPU solo CUDA (MoSES, EST); r.avaflow/RAMMS en CPU; SWE en navegador solo deluge (agua, sin paper) | Nadie en detritos | Alta | Primer solver de remoción en masa que corre en el navegador del municipio y en la GPU del laboratorio con el mismo WGSL; caso Macul en vivo | Sin bit a bit vs CPU: publicar tolerancia (masa, alcance) no identidad; Linux/Firefox parcial |
| **Tiles dinámicos calculados en GPU en `surtgis serve`** | titiler/CPU; tu p50 9 ms ya compite | Nadie en Rust | Media | Fórmulas/índices sobre COG multibanda en wgpu; misma ruta sirve a WASM cliente | Ganancia solo en tiles grandes o multi-tile; latencia de upload GPU |
| **Viewshed total/acumulado en wgpu (nativo+web)** | CUDA académico 2011-2024; Waveshed WebGPU en RF | Waveshed (nicho) | Media | R2/XDraw en compute shader; visibilidad de vertederos/turbinas en cliente | Concordancia con GRASS r.viewshed ±1 celda |
| **FastFlow-style flow routing/depression en wgpu** | Solo CUDA (Inria/UCT), 2024 | Jain et al. | Alta | Hidrología GPU portable (AMD/Apple/web) inexistente | Resultados difieren de priority-flood exacto; hay que ofrecer ambos |
| **Radiación solar anual + horizontes en GPU** | HORAYZON CPU; RS 2026 GPU DSM | Steger; grupo RS 2026 | Media | Tu solar-annual (67 s/45M) → segundos; sombras por hora en navegador | Trig 2⁻¹¹ en WGSL: reimplementar sin/cos si se exige paridad |
| **Terreno completo sin servidor: COG→WASM(Rayon)→WebGPU** | whitebox-wasm/terrano sin threads ni GPU; deck.gl-raster WebGL2 | opengeos, GeoLang, DevSeed | Media | SurtGIS ya tiene WASM+demo 3D; falta hilos y GPU integrados | Rayon excluye memory64 (4 GB); COOP/COEP en hosting |
| **Test-suite de paridad CPU/GPU/Web publicable** | Papers de determinismo solo en ML | Nadie en GIS | Baja-media | Paper corto (C&G/EMS) definiendo tolerancias por algoritmo | Requiere GPU Linux con WebGPU (NVIDIA 147+/Wayland) |

Ventana real: 12-18 meses, antes de que whitebox-wasm o deck.gl/luma v10 integren compute; los que ya están (FastFlow, SERGHEI) no tienen ruta web ni Rust.


## Anexo C. IA geoespacial: foundation models, inferencia y agentes

Nota metodológica: presupuesto de búsquedas web agotado (200); todo viene de esas búsquedas más lecturas directas de arXiv, Hugging Face, GitHub y blogs. Se marca [paper] vs [herramienta] donde importa.

### 1. Foundation models geoespaciales

**Modelos con pesos abiertos (Apache-2.0 salvo indicación):**
- **Prithvi-EO-2.0** (IBM/NASA): 300M y 600M (dic-2024, [HF](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)); variantes **tiny (5M) y 100M-TL (87M)** desde oct-2025, con caída de solo 4 %/1 % en benchmark ([IBM Research, 10-oct-2025](https://research.ibm.com/blog/terramind-prithvi-tiny-small-models-geospatial)). Tiny pesa 144 MB en `.pt`; corre a 329 fps en el procesador espacial Unibap iX10 y en un iPhone 16 Pro. Formato: checkpoints PyTorch vía TerraTorch; **no hay ONNX oficial**.
- **TerraMind 1.0** (ESA/IBM, abr-2025): tiny 5M / small 20M / base / large ([HF](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)); tiny a 325 fps con 12 bandas 224×224.
- **Clay v1.5** (nov-2024): 632M totales, encoder 311M, checkpoint `.ckpt` de 1,25 GB ([spec](https://clay-foundation.github.io/model/release-notes/specification.html)). Sin ONNX documentado.
- **Galileo** (NASA Harvest): tiny 5,3M ([GitHub](https://github.com/nasaharvest/galileo)); **DOFA, Panopticon, Copernicus-FM** en TorchGeo 0.7 ([release](https://github.com/microsoft/torchgeo/releases/tag/v0.7.0)); **OlmoEarth** (AI2, nov-2025, pesos y datos abiertos, [arXiv](https://arxiv.org/abs/2511.13655)).
- **TESSERA** (Cambridge): pixel-wise, 128-d, 10 m, pesos y código abiertos, embeddings CC0. v1.1 jun-2026 (S3 + HF); **v2 (jul-2026)** destila teachers de 0,5–2B a un **estudiante de 44M con representaciones Matryoshka: 16-d retienen 92 % del rendimiento** ([arXiv 2607.03949](https://arxiv.org/abs/2607.03949)). CDSE publicó un ejemplo openEO para generarlos el 25-sep-2026 ([CDSE](https://dataspace.copernicus.eu/news/2026-9-25-new-openeo-community-example-generate-tessera-pixel-embeddings-sentinel-1-and)).

**AlphaEarth Foundations** (DeepMind): **pesos y entrenamiento cerrados**; solo embeddings 64-d int8, 10 m, anuales 2017–2025, en GEE, en GCS como COG y en Source Cooperative desde nov-2025 ([Medium GCS](https://medium.com/google-earth/alphaearth-foundations-satellite-embeddings-now-available-on-google-cloud-storage-f9ab0f7252d6); [arXiv](https://arxiv.org/abs/2507.22291)). Major TOM los regrilló a GeoParquet ([HF](https://huggingface.co/datasets/Major-TOM/Core-AlphaEarth-Embeddings)). **Major TOM/ESA Φ-lab**: 170M embeddings (SSL4EO, DINOv2, SigLIP, MMEarth) en GeoParquet en CDSE desde nov-2025 ([CDSE](https://dataspace.copernicus.eu/news/2025-11-14-global-embedding-dataset-now-available-cdse)).

**Escepticismo útil:** el post de CNG "The technical debt of Earth embedding products" (28-feb-2026, [link](https://cloudnativegeo.org/blog/2026/02/the-technical-debt-of-earth-embedding-products/)) muestra que solo África en AlphaEarth son 19,2 TB y en TESSERA 38,4 TB; que TESSERA v1 entregaba "numpy sin CRS" y que cada producto inventa su tiling. Y una evaluación de 89 GFMs (ago-2026, [arXiv](https://arxiv.org/abs/2608.03804)) concluye que **un tercio no ofrece nada más que código fuente**. Lo que sí corre en CPU hoy: los tiny de 5M y el estudiante TESSERA de 44M; los de 300M+ son GPU en la práctica.

### 2. Inferencia en Rust / borde

- **ort** 2.0.0-rc.13 (28-jul-2026) ata ONNX Runtime 1.30, MSRV 1.88; aún sin 2.0 estable ([docs.rs](https://docs.rs/crate/ort/latest), [GitHub](https://github.com/pykeio/ort)). No compila a WASM por sí solo: para navegador delega en backends puros Rust.
- **tract** (Sonos): ONNX/NNEF, WASM, crates `tract-metal`/`tract-cuda`, producción real ([GitHub](https://github.com/sonos/tract)). **burn-onnx** convierte ONNX a código Rust con backend wgpu/WebGPU, pero en abr-2026 aún había panics con ResNet/MobileNet en WebGPU ([issue #325](https://github.com/tracel-ai/burn-onnx/issues/325)). **candle** es el más liviano para inferencia ([comparativa abr-2026](https://dasroot.net/posts/2026/04/rust-machine-learning-burn-vs-candle-framework-comparison/)). **wonnx** (WebGPU puro Rust) parece abandonado.
- En navegador, ONNX Runtime Web + WebGPU es el camino maduro (SAM2 en el browser, ~85 % soporte WebGPU en 2026, [SitePoint](https://www.sitepoint.com/webgpu-browser-ai-javascript-inference/)).
- **Borde real:** destilación Prithvi 300M → EfficientViT-B0 de **0,7M, INT8, 1,5 MB, 5,57 ms por tile 512² en Jetson**, IoU 0,787 vs 0,822 del teacher (17-sep-2026, [arXiv 2609.20441](https://arxiv.org/abs/2609.20441)). Φsat-2 ya infiere en órbita ([eoPortal](https://www.eoportal.org/satellite-missions/phisat-1)).
- **Vacío encontrado:** no se halló a nadie haciendo embeddings o segmentación de imágenes satelitales en un binario Rust sin Python ni en WASM geoespacial. samgeo es Python+GPU ([samgeo](https://samgeo.gishub.org/)); los demos de similitud AlphaEarth consultan GEE en vivo, no infieren localmente ([similar-earth](https://github.com/pariosur/similar-earth)).

### 3. Terreno + ML

- **DEM super-resolución:** solo [papers] (TGFSR, IJAEOG 2026; normalizing flow, Sci Rep 2025; multimodal con Depth Anything, 2025; ViT con priors, [arXiv 2507.09681](https://arxiv.org/pdf/2507.09681)). Ninguna herramienta usable.
- **DSM→DTM global:** **FathomDEM** (ERL, ene-2025, ViT híbrido; Américas en [Zenodo](https://zenodo.org/records/14523356)), **GEDTM30** (PeerJ 2025, abierto, OpenGeoHub, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12296579/)), FathomDEM+ comercial 2026, DeltaDTM/DiluviumDEM costeros. Generación de terreno por difusión llegó a SIGGRAPH 2026 ([terrain-diffusion](https://github.com/xandergos/terrain-diffusion)).
- **Susceptibilidad:** embeddings AlphaEarth superan a los factores condicionantes clásicos en +0,04–0,11 AUC en Taiwán, Hong Kong y Emilia-Romagna ([arXiv 2601.07268](https://arxiv.org/abs/2601.07268), rev. sep-2026); Clay como contexto auxiliar de U-Net sube F1 59,9→64,5 en Landslide4Sense, pero Clay solo cae a 55,2 ([arXiv 2606.14081](https://arxiv.org/abs/2606.14081)). Review NHESS ene-2026 de 400+ estudios ([NHESS](https://nhess.copernicus.org/articles/26/487/2026/)).
- **Emuladores:** U-Net patch-predict-stitch para flujos post-incendio (C&G, abr-2025, [arXiv](https://arxiv.org/abs/2504.07736)); FNO 1D y **FNO-3D 2D para detritos entrenado con un solver FV propio** (Italia, ene y abr-2026, [Geosciences](https://doi.org/10.3390/geosciences16020055), [Land](https://doi.org/10.3390/land15050759)); inundación FNO + "Magnifier" **10.800× más rápido que ANUGA**, pero los revisores señalan que exige el hidrograma completo y solo se valida contra el solver ([EGUsphere, may-2026](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1982/)). Todos [paper]; nadie entrega binario.
- **GIS diferenciable:** **Inunda** (jul-2026): solver de inundación local-inercial, GPU-nativo, diferenciable, calibra conductividad por gradiente contra aforos, NSE 0,72 vs 0,31 del operacional ([arXiv 2607.09614](https://arxiv.org/abs/2607.09614)). HydroModels.jl (EMS 2025) y δHydro (WRR 2026) son hidrología conceptual. No existe una librería de análisis de terreno (pendiente, acumulación) con autodiff.

### 4. Agentes GIS y LLM

- **Herramientas:** QGIS MCP con 125 tools ([plugin](https://plugins.qgis.org/plugins/qgis_mcp_plugin/)), gdal-mcp con middleware que obliga al agente a justificar el método ([GitHub](https://github.com/JordanGunn/gdal-mcp), [post feb-2026](https://bertt.wordpress.com/2026/02/09/gdal-powered-ai-agents/)), **Esri MCP beta desde 29-jun-2026** ([blog](https://www.esri.com/arcgis-blog/products/platform/developers/mcp-support-beta-and-arcgis-static-maps-service-in-arcgis-location-platform-release)), GEE MCP, y 77+ servidores catalogados por Sparkgeo ([link](https://sparkgeo.com/blog/geospatial-mcp-servers-mapped-and-categorized/)).
- **Benchmarks:** GIS Copilot 86 % en 110 tareas QGIS (IJDE 2025); **GISclaw** (may-2026) llega a 97–100 % en GeoAnalystBench generando código Python en sandbox, no con tool-calling ([arXiv 2603.26845](https://arxiv.org/html/2603.26845)); **GeoAgentBench** (abr-2026, 117 tools) concluye que "la configuración precisa de parámetros es el determinante primario del éxito" ([arXiv 2604.13888](https://arxiv.org/abs/2604.13888)); **GISAgentBench** (ago-2026, 349 tareas reales de GIS StackExchange, 63 de terreno/hidrología): **mejor modelo 32,7 % estricto**; fallas por operaciones faltantes 28 %, secuencia 18 %, y **desalineación de CRS baja el éxito a 0,22** ([arXiv 2608.01645](https://arxiv.org/html/2608.01645v1)). ANASSA (sep-2026) formaliza contratos con CRS, incertidumbre y estados de fallo tipados ([arXiv 2609.14824](https://arxiv.org/html/2609.14824v1)).
- **Diseño que funciona:** APIs tipadas con contrato de retorno documentado; validación de CRS/topología/unidades antes de ejecutar; fallos tipados (no strings); pocas herramientas de granularidad media con parámetros acotados; y para modelos capaces, un sandbox de código supera a 100+ tools atómicas. Los benchmarks fáciles (50 tareas de libro) están saturados; los reales, no.

### 5. Oportunidades para SurtGIS

| Oportunidad | Estado del arte hoy | Quién | Dificultad | Impacto | Riesgo |
|---|---|---|---|---|---|
| **Embeddings de FM tiny en el binario y en WASM, servidos como tiles dinámicos** (Prithvi/TerraMind 5M, TESSERA 44M/16-d) vía `ort`/`tract` sobre COG remoto | Solo Python+GPU (TerraTorch) o GEE; nadie en Rust ni en navegador | IBM, Cambridge, Google | Media-alta: exportar ViT con patch 3D a ONNX, paridad bit a bit con la normalización de bandas | Muy alto: "AlphaEarth local, abierto y sin cuenta Google" | Sin ONNX oficial; ViT en tract/candle poco probado; latencia CPU |
| **Lector/servidor de embeddings precomputados** (AlphaEarth COG, TESSERA, Major TOM GeoParquet) como bandas + similitud coseno local | Formatos incompatibles, "deuda técnica" documentada por CNG (feb-2026) | Element 84, Earth Genome, CNG | Baja-media | Alto y rápido | Volumen (TB); cambios de tiling upstream |
| **Contrato de herramientas para agentes** sobre las 42 tools MCP: CRS/unidades/nodata tipados, prevalidación, fallos tipados, + mini-benchmark terreno/hidrología reproducible | GISAgentBench 32,7 %; CRS es la falla más costosa | PSU (GIS Copilot), GISclaw, ANASSA | Media | Alto: es exactamente donde fallan los agentes | Benchmarks propios parecen autopromoción; hay que publicar con datos ajenos |
| **Emulador neuronal de `surtgis-flow`** entrenado con su propio solver (FNO/U-Net) e inferido en el binario | Papers italianos 2026 hacen lo mismo con solver propio; sin herramienta | Univ. italianas, USGS | Alta | Alto: runout en segundos, ensembles μ/ξ | Generalización a topografía nueva; se valida solo contra el solver |
| **Solver de detritos diferenciable** (adjoint/dual numbers) para calibrar μ, ξ contra huella (IoU) | Inunda (jul-2026) lo hace para inundación; ninguno para detritos | Inunda, δHydro | Muy alta | Alto científicamente (paper) | Gradientes a través de frentes secos/fricción no suave; alternativa barata: sensibilidad paralela por diferencias finitas |
| **DSM→DTM y SR local con modelos pequeños in-binary** (`dtm-correct`) | FathomDEM cerrado/ViT; GEDTM30 abierto pero estático; SR solo papers | Fathom, OpenGeoHub | Media | Medio-alto para Chile (sin LiDAR nacional) | Sin LiDAR de referencia local, el modelo se importa a ciegas |
| **Susceptibilidad con embeddings como features** en `extract-patches`/`predict_raster` (replicar el +0,04–0,11 AUC en Chile) | Demostrado en 3 regiones (2026) | arXiv 2601.07268 | Baja | Medio; buen paper de validación regional | Embeddings anuales: no capturan el evento, sí el contexto |
| **Segmentación destilada INT8 (<2 MB) en WASM** (agua, depósitos) | 0,7M/1,5 MB en Jetson (sep-2026); nada en browser geoespacial | arXiv 2609.20441 | Media | Medio: demo viral, uso en terreno sin red | Precisión cae 3–4 pts; entrenamiento aún requiere GPU |

Recomendación escéptica: las filas 1–3 son las únicas donde SurtGIS tiene una ventaja estructural real (binario único, COG/STAC nativo, WASM y MCP ya existentes) y donde el estado del arte son papers o Python-con-GPU, no herramientas. Las 4–5 dan papers, no usuarios, en 2027.


## Anexo D. Hidrología digital, geomorfometría y hazards

### 1. Herramientas 2024-2026: qué hay nuevo y qué hacen que SurtGIS no

- **TopoToolbox 3** — preprint EGUsphere 2026-05-26 (Kearney, Schwanghart et al.): núcleo C `libtopotoolbox` compartido por MATLAB/Python/R, integra **GraphFlood** (Gailleton et al., ESurf 2024: SWE 2D aproximadas sobre el DAG de flujo, ~10× más rápido que River.lab, escala ~lineal a 10⁶–10⁸ celdas), acoplamiento bidireccional con Landlab y **property-based/metamorphic testing**. pytopotoolbox v0.0.7 (2025-09-29). Lo que SurtGIS no tiene: hidráulica aproximada estacionaria sobre DEM (profundidad/caudal 2D sin correr un SWE completo) y un solver de red fluvial completo (chi/ksn/knickpoints maduros). https://egusphere.copernicus.org/preprints/2026/egusphere-2026-2478/ · https://esurf.copernicus.org/articles/12/1295/2024/
- **Whitebox Workflows Next Gen (WbW-NG)** — anunciado 2026-06-27, reescritura Rust modular (`wbraster`, `wbvector`, `wblidar`, `wbprojection`, `wbtopology`), ~750 tools, WhiteboxTools marcado legacy. Es el competidor directo en el mismo lenguaje: I/O LiDAR/LAS nativo, vector completo y `StochasticDepressionAnalysis`/`BreachDepressionsLeastCost` que SurtGIS no ofrece. https://github.com/jblindsay/whitebox_next_gen · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7161204
- **GRASS 8.5.0** (2026-05-08): API Python nueva, JSON en decenas de tools, y addons hidrológicos OpenMP: `r.hand`, `r.hydrobasin`, `r.lfp`, `r.hydroflatten`, `r.slopeunits`, `r.runoff` (SCS-CN), `r.timeofconcentration`. Brecha vs SurtGIS: hidroaplanado, slope units, longest flow path, CN/runoff. https://grass.osgeo.org/news/2026_05_08_grass_8_5_0_released/
- **SAGA 9.8→9.13** (abr-2025 → jul-2026): Multi-Scale Roughness, hillshading extendido, cloth simulation filter LiDAR (9.12), descargadores OpenTopography/SoilGrids/CHELSA, Awesome Spectral Indices (9.8). Brecha: filtrado de nube de puntos y roughness multiescala. https://sourceforge.net/p/saga-gis/news/
- **xDEM v0.2** (GlacioHack): coregistro 3D, corrección de sesgos, **heterocedasticidad + correlación espacial de errores + propagación a derivados** (Hugonnet et al. 2022). Es el referente de incertidumbre; SurtGIS no tiene nada equivalente. https://xdem.readthedocs.io/en/stable/uncertainty.html
- **Landlab 2.11.0** (2026-04-06) con `ConcentrationTracker` (GMD 2026-02-13); **fastscapelib** reescrito (grillas raster/triangulares); ambos son LEM, no motores GIS. https://gmd.copernicus.org/articles/19/1387/2026/
- **pysheds 0.5** (última 2025-08-14), **pyflwdir 0.5.11** (2026-04-21, base de HydroMT), **RichDEM 2.3.1** (feb-2026, activo pero sin novedades algorítmicas). https://pypi.org/project/pysheds/ · https://libraries.io/pypi/pyflwdir
- **HydroSHEDS v2 Américas**: TanDEM-X 1", ~7.700 sumideros naturales, ~40 M líneas de impronta, ~200 k correcciones manuales; resto del globo sin fecha. MERIT-Hydro sigue en 3" (2019). https://www.hydrosheds.org/news/hydrosheds-v2-for-the-americas
- **TauDEM**: no encontré release 2024-2026; sigue siendo el MPI de referencia.

### 2. Algoritmos de frontera

- **Condicionamiento LiDAR 1 m**: el problema activo no es fill vs breach sino **alcantarillas/puentes**. DEMend (Water Resour. Manag. 2025) automatiza detección/corrección; Frontiers AI 2025 clasifica cruces de drenaje con EfficientNetV2 sobre openness/curvatura/TPI a 1 m; USGS (Stanislawski) hace *road breaching* por DL. https://link.springer.com/article/10.1007/s11269-025-04226-2 · https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1561281/full · https://pubs.usgs.gov/publication/70200636
- **Priority-flood paralelo/GPU**: Barnes 2016 (tiles, trillón de celdas) sigue siendo el estándar en CPU; **FastFlow** (Jain et al., CGF 2024) lleva flow + depression routing a GPU. https://arxiv.org/pdf/1606.06204 · https://onlinelibrary.wiley.com/doi/10.1111/cgf.15243
- **HAND operacional**: NOAA OWP FIM 4 cubre ~100 % de la población de EE. UU. (2026-09-24), con **ras2fim 1.18** híbrido HAND + HEC-RAS; FIMserv v1.0 lo empaqueta. https://www.noaa.gov/news-release/noaa-flood-mapping-tool-now-covers-nearly-100-of-us · https://github.com/NOAA-OWP/ras2fim
- **Landforms/DL**: GeomorPM (2025, modelo preentrenado conv+Transformer sobre DEM), revisión ESR 2025 de unidades aluviales; no hallé un "geomorphons 2.0" formal, solo aplicaciones. https://onlinelibrary.wiley.com/doi/10.1002/esp.70295
- **Conectividad de sedimentos**: SedInConnect 3.0 modular (GitHub), sin paper nuevo 2024-2026. https://github.com/HydrogeomorphologyTools/SedInConnect_3.0
- TWI multiescala: sin verificar (presupuesto agotado).

### 3. Hazards

- **r.avaflow v4** (GMD 2025-12): modelo por capas, control de deformación, Voellmy por fase, slow-flow, VR. https://gmd.copernicus.org/articles/18/9879/2025/
- **Kestrel** (JOSS 2024-01): dos fases líquido/sólido, erosión-deposición morfodinámica, arrastre que transita fluido turbulento↔granular; aplicado a Cotopaxi 2024-2025. https://github.com/jakelangham/kestrel/
- **Advertencia clave**: Langham et al., *JFM* 2025 demuestran **ill-posedness** en Pitman-Le, Pudasaini 2012, Pudasaini-Mergili 2019 (3 fases) y otros; se cura con términos difusivos pequeños. "Two-phase Pudasaini" sin ese cuidado no es vanguardia, es un riesgo. https://arxiv.org/abs/2505.21254
- **Calibración**: back-analysis bayesiano y ensambles (Landslides 2024 doi:10.1007/s10346-024-02423-5; Landslides 2026 doi:10.1007/s10346-026-02746-5; NPG 2026 "Bayesian data selection", Zhao); USGS propaga lluvia→volumen→runout probabilístico (NHESS 2024). https://nhess.copernicus.org/articles/24/2359/2024/ · https://npg.copernicus.org/articles/33/425/2026/
- **HEC-RAS**: no seguirá desarrollando mud/debris en la serie 6; DebrisLib (Geosciences 2025) como librería modular no-newtoniana. https://doi.org/10.3390/geosciences15070240
- **Inundación rápida**: LISFLOOD-FP 8.2 (GMD 2025, DG multiwavelet adaptativo GPU), SERGHEI escalado a 2048 GPUs (arXiv 2511.01001), FastFlood Global (WASM, 2025-11-10, 1500× vs modelos clásicos). https://gmd.copernicus.org/articles/18/9827/2025/ · https://fastflood.org/
- **Susceptibilidad**: revisión NHESS 2026 (>400 estudios), embeddings AlphaEarth (arXiv 2601.07268), adaptadores sobre vision foundation models (arXiv 2608.09325). https://nhess.copernicus.org/articles/26/487/2026/
- **Gemelos digitales**: slope digital twin (Eng. Geol. 2025), plataforma criosférica (NSR 2024), tsunami bayesiano en tiempo real (arXiv 2504.16344). Lo que define un "solver de vanguardia 2026": bien puesto matemáticamente, erosión acotada, ensambles con posterior calibrada, y latencia compatible con pronóstico.

### 4. Incertidumbre y DEMs

- Copernicus GLO-30 2023_1 (2024-07-23) y 2024_1 en GEE; TanDEM-X 30 m EDEM; **GEDTM30** (PeerJ 2025: fusión Copernicus+AW3D+alturas de objetos, 30 mil millones de puntos ICESat-2/GEDI, COG+STAC en OpenLandMap); **FathomDEM** (2025, remoción de objetos por visión computacional); GDEMM2024; corrección de GDEMs con GEDI/ICESat-2 (IJDE 2024). https://peerj.com/articles/19673/ · https://opentopography.org/news/updated-copernicus-30m-DEM-available
- Propagación Monte Carlo con autocorrelación espacial (Geo-spatial Inf. Sci. 2024) sigue siendo el método; ningún motor GIS lo ofrece como primitiva salvo xDEM en Python. https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2324921
- Grillas esferoidales: sigue siendo Florinsky 2017 y Guth 2021 (Trans. GIS); GEE ya lo implementa a escala global (IJGI 2020). Ningún motor Rust lo hace.

### 5. Oportunidades para adelantarse

| Oportunidad | Estado del arte hoy | Quién | Dificultad | Impacto | Riesgo |
|---|---|---|---|---|---|
| Priority-flood/acumulación **por tiles sobre COG remoto** sin cargar el DEM | Barnes 2016 en disco local; WbW-NG y GRASS 8.5 en RAM/OpenMP | Nadie sobre COG streaming (no verificado por búsqueda) | Alta | Alto: hidrología continental desde `surtgis serve` | Correctitud en costuras de tiles; latencia HTTP |
| **Incertidumbre Monte Carlo como primitiva** de todos los derivados (error heterocedástico + variograma) | xDEM (Python, offline) | GlacioHack | Media | Alto: único motor nativo con σ por celda | Costo N×; comunicar bien el modelo de error |
| Solver de detritos **bien puesto** + ensambles bayesianos | r.avaflow v4, Kestrel; ill-posedness demostrada (JFM 2025) | Mergili, Langham/Woodhouse, Zhao | Alta | Alto: Macul calibrado con posterior | Reinventar física; validación reducida a un caso |
| Condicionamiento LiDAR con **detección de alcantarillas** (DL o geomorfométrica) | DEMend 2025, USGS DL | USGS, U. Cincinnati | Media | Medio-alto (municipal, flash flood) | Requiere datos de entrenamiento locales |
| **GraphFlood-like** (SWE aproximado sobre DAG) en Rust/WASM | TopoToolbox 3, GraphFlood 1.0 | Gailleton | Media | Alto: inundación en navegador tipo FastFlood | Duplicar TT3; aprobación de referencias |
| Grillas **esferoidales (Florinsky/Guth)** para COG globales sin reproyectar | GEE; ninguna librería nativa | Florinsky, Guth | Baja-media | Medio: derivados directos de GEDTM30/Copernicus | Nicho; validación contra GEE |
| Slope units + longest flow path + SCS-CN en `serve` | GRASS 8.5 addons | Comunidad GRASS | Baja | Medio: paridad para susceptibilidad | Poco diferenciador |
| **Property-based/metamorphic testing** publicable del motor | TT3 lo declara como novedad | Kearney et al. | Baja | Medio: credencial de calidad | Ninguno relevante |

**Lectura escéptica**: el terreno donde SurtGIS ya compite (derivados, D8/D-inf, fill/breach) está saturado y Whitebox Next Gen elimina la ventaja "Rust". Las brechas defendibles son tres: streaming remoto de hidrología global, incertidumbre nativa, y un solver de detritos con posedness demostrada y ensambles, que ningún paquete open source combina hoy.

No verificado por agotamiento del presupuesto de búsqueda: priority-flood sobre COG, TWI multiescala, Iber+/GLOF, releases de TauDEM.


## Anexo E. Servidores de tiles, visores y análisis en cliente

Nota metodológica: todo sale de la web. Donde una fuente es un README o un blog de vendedor se señala; "en producción" significa release etiquetado o despliegue público verificable.

### 1. Servidores dinámicos 2025-2026

**TiTiler no tiene un "1.0" reciente: va en 2.x.** 2.0.0 salió el 2026-03-16 (exige rio-tiler ≥9, elimina el tilesize 256 por defecto y los sufijos de escala), y la última es 2.4.0 del 2026-09-21 (endpoint de coordenadas xarray) — [CHANGES.md](https://github.com/developmentseed/titiler/blob/main/CHANGES.md). Su motor rio-tiler pasó por 8.0 (2025-11-20, `ZarrReader` experimental) y 9.0 (2026-03-11, reader **asíncrono experimental sobre obstore + async-geotiff**, es decir, sin GDAL en el camino caliente); 9.4 (2026-09-17) agrega readers zarr async, `GeoZarrReader` y `AsyncSTACReader` — [rio-tiler CHANGES](https://github.com/cogeotiff/rio-tiler/blob/main/CHANGES.md). Esto es lo más cercano a "TiTiler en Rust": no existe un port, pero DevSeed está sacando GDAL del path de lectura con crates Rust envueltos en Python.

**Derivados de terreno al vuelo:** TiTiler los tiene desde 0.8 como `algorithm=` (hillshade, slope, contours, terrarium, terrainrgb, normalizedIndex, cast, etc.), con un parámetro `buffer` explícito para el borde del tile; no documenta encadenamiento de algoritmos — [Algorithms](https://developmentseed.org/titiler/user_guide/algorithms/). Es decir, la idea "operador como parámetro de URL" ya está en producción en TiTiler para operadores locales de ventana. Nadie tilea operadores globales: la respuesta estándar es materializar (TiTiler no lo hace; se delega a pipelines externos). El único marco que lo conceptualiza es OGC API Processes Part 3 ("collection output": el request de un tile dispara el proceso), pero es **borrador** — [21-009 DRAFT](https://docs.ogc.org/DRAFTS/21-009.html).

**Ecosistema DevSeed:** titiler-pgstac 3.1.0 (2026-08-04), eoapi-cdk 11.7.0 (2026-09-10) y eoapi-k8s 0.16.3 (2026-09-17) — todos en producción — [eoapi-cdk](https://github.com/developmentseed/eoapi-cdk/releases/tag/v11.7.0). titiler-cmr corre en NASA Worldview y FIRMS desde 2025 — [NASA Earthdata](https://www.earthdata.nasa.gov/news/feature-articles/tailoring-view-titiler-cmr-customizes-hls-imagery-worldview-firms). titiler-eopf 0.11.1 sirve GeoZarr de Sentinel en el EOPF Explorer (ESA/EOX) — [EOX, 2026-02-27](https://eox.at/2026/02/visualizing-geozarr/).

**Zarr/cubos:** xpublish-tiles (Earthmover, blog 2025-09-29) sirve OGC Tiles desde Zarr/Icechunk con Datashader+Numba, coarsening dinámico para zooms bajos y reclama 10× vs TiTiler en ciertas configuraciones (30 ms para un array 2048² en un M2); soporta grillas curvilíneas y HEALPix, algo que ningún tiler COG hace; releases 0.9.x en septiembre 2026 — [Earthmover](https://www.earthmover.io/blog/dynamic-map-tile-rendering-icechunk-zarr-data-xpublish-tiles), [PyPI](https://pypi.org/project/xpublish-tiles/). openEO/CDSE migró su backend a STAC (2026-03-02); no hay backend Rust de openEO — [CDSE](https://dataspace.copernicus.eu/news/2026-3-2-openeo-boosts-performance-cdse-new-stac-backend).

**Rust:** Martin tiene COG solo bajo `--features=unstable-cog`, no en el build por defecto, **sin EPSG:3857 aún**, solo RGB 8-bit, sin rescale/colormap — [docs Martin](https://maplibre.org/martin/sources-cog-files/). tileserver-rs (un desarrollador, 57 estrellas, MIT) promete un binario para PMTiles/MBTiles/PostGIS/COG/GeoParquet/DuckDB/STAC con Terrarium/Mapbox-RGB y hillshade al vuelo, "4-10× vs titiler" (130 vs 28 req/s) y ~100 ms cache caliente / ~800 ms frío; es marketing de README, sin benchmark reproducible — [tileserver.app](https://tileserver.app/), [GitHub](https://github.com/vinayakkulkarni/tileserver-rs). Nadie en Rust combina fórmula espectral + derivados de terreno + jobs de materialización en un binario.

**Java:** GeoServer 3.0.0 (2026-06-12) es modernización de plataforma (Spring 7, ImageN reemplaza JAI), nada nuevo en tiles dinámicos — [OSGeo](https://www.osgeo.org/community-news/geoserver-3-0-0-released/). MapServer 8.6 (dic 2025) — [mapserver.org](https://mapserver.org/development/announce/8-4.html).

### 2. Estándares

- **OGC API Tiles 1.0**: aprobado; ahora también EN ISO 19177-1:2026 — [ogcapi.ogc.org/tiles](https://ogcapi.ogc.org/tiles/), [iTeh](https://standards.iteh.ai/catalog/standards/cen/04980f6a-f98f-40b9-8b85-51a3292014c8/en-iso-19177-1-2026).
- **OGC API DGGS Part 1 v1.0.0 (21-038r1)**: aprobado — [ogcapi.ogc.org/dggs](https://ogcapi.ogc.org/dggs/).
- **OGC API Coverages**: la portada de ogcapi.ogc.org lo lista "approved", pero su página propia y docs.ogc.org siguen en DRAFTS (19-087); tratarlo como recién aprobado o en trámite, no como maduro — [DRAFT 19-087](https://docs.ogc.org/DRAFTS/19-087.html).
- **OGC API Processes v2 (Part 1 + Part 2 Deploy/Replace/Undeploy)**: comentario público 2026-08-06 → 2026-09-05; Part 3 (workflows) y Part 4 (job management) siguen borrador — [OGC](https://www.ogc.org/requests/ogc-api-processes-standard-version-2-public-comment/). ZOO-Project ya implementa Part 2; pygeoapi 0.25 en octubre 2026 — [pygeoapi FOSS4G 2026](https://pygeoapi.io/presentations/foss4g2026/).
- **GeoZarr**: apunta a revisión del OGC Architecture Board "verano 2026"; multiscales vía convención `zarr-conventions/multiscales`; ndpyramid genera pirámides — [geozarr.org/faq](https://geozarr.org/faq), [multiscales](https://github.com/zarr-conventions/multiscales).
- **TileJSON 3.0**: solo metadatos; no modela tiles de datos ni codificaciones tipo terrain-rgb — [spec](https://github.com/mapbox/tilejson-spec/blob/master/3.0.0/README.md).
- **CNG Forum 2026**: 6-9 octubre, Snowbird; "menos charlas, más conversación"; programa no publicado en detalle — [CNG](https://cloudnativegeo.org/blog/2025/12/join-us-at-cng-forum-2026-building-the-future-of-cloud-native-geospatial/).

### 3. Cliente sin servidor

MapLibre GL JS ya está en 6.x (v6.11.2, septiembre 2026); tiene `raster-dem`, hillshade nativo con varios métodos (desde v5.5.0), capa `color-relief` (ejemplo publicado 2025-06-25) y `texelFetch` para lecturas exactas del DEM; **sin WebGPU** — [releases](https://github.com/maplibre/maplibre-gl-js/releases), [color-relief](https://maplibre.org/maplibre-gl-js/docs/examples/add-a-color-relief-layer/). deck.gl 9 tiene TerrainLayer (ahora en GlobeView) y `@developmentseed/deck.gl-geotiff` lee COG en navegador con `@developmentseed/geotiff` + `@cogeotiff/core` — [deck.gl](https://deck.gl/docs/whats-new), [npm](https://www.npmjs.com/package/@developmentseed/deck.gl-geotiff).

Lo más agresivo viene de opengeos: **cog-tiler-wasm** (tiles XYZ estilo TiTiler generados en el navegador desde range requests, colormaps, rescale, estadísticas; sin band math, solo 3857) — [GitHub](https://github.com/opengeos/cog-tiler-wasm); **whitebox-wasm** (733 herramientas vía WASI, `CogStream` por rangos, tope 4 GiB) — [GitHub](https://github.com/opengeos/whitebox-wasm); y **GeoLibre**, app con hillshade/slope/aspect/zonal/focal en WASM y plugin de embeddings (AlphaEarth, Tessera, Earth Index, Clay) mergeado 2026-09-23 — [GeoLibre](https://geolibre.app/features/), [PR #2602](https://github.com/opengeos/GeoLibre/pull/2602). terrano (Rust, 2026) añade flow accumulation y watershed compilables a WASM — [GitHub](https://github.com/GeoLang/terrano).

Distancia real: pendiente/hillshade/colormap en cliente desde COG remoto es **producción** hoy (cog-tiler-wasm, GeoLibre, MapLibre). Curvatura y TWI no: TWI necesita acumulación global, y el límite de 4 GiB + ausencia de compute shaders en MapLibre lo confinan a rasters chicos. Nadie ofrece un modelo híbrido donde el cliente calcule lo local y el servidor materialice lo global con el mismo código.

### 4. Colaborativo y notebooks

lonboard (GeoArrow, bump a deck.gl-geoarrow 0.4 el 2026-09-25) domina vector; raster en notebooks sigue pasando por TiTiler o leafmap/cog-tiler-wasm — [lonboard PR](https://github.com/developmentseed/lonboard/pull/1195). Felt lanzó "AI raster analysis" (zonal stats, perfiles, enriquecimiento) el 2026-09-15, expuesto vía MCP; sin cifras — [Felt](https://www.felt.com/blog/raster-analysis). Cuellos de botella: latencia fría del COG remoto (tileserver-rs admite ~800 ms; los 0.4-2.4 s de SurtGIS son comparables), egress US$0.08-0.12/GB (AWS 0.09, GCP 0.12) y el hecho de que un cache miss cuesta cómputo y egress a la vez — [Spendark 2026](https://spendark.com/blog/cloud-egress-costs-guide/), [dev.to](https://dev.to/beefedai/cost-and-performance-tradeoffs-between-pre-generated-and-dynamic-tiles-2c2d). DevSeed mantiene un sitio de tile-benchmarking (COG vs Zarr, chunk sizes, cache) que sirve de plantilla para el benchmark pendiente — [tile-benchmarking](https://developmentseed.org/tile-benchmarking/).

### 5. Oportunidades para SurtGIS

| Oportunidad | Estado del arte hoy | Quién | Dificultad | Impacto | Riesgo |
|---|---|---|---|---|---|
| **Tiles + OGC API Processes v2 en un binario**: `POST /jobs` → `/processes`, `/jobs/{id}/results` como TileJSON | Nadie une tiler y Processes; ZOO/pygeoapi son Python sin tiler; Part 3 "collection output" es borrador | ZOO-Project, pygeoapi, Ecere | Media (mapear jobs existentes; conformance classes) | Alto: primer tiler con procesos globales estandarizados | Processes v2 puede cambiar tras el comentario público |
| **Tiles f32 de datos** (`.npy`/Arrow/raw f32 con máscara) para análisis en cliente | TiTiler ya sirve NumpyTile; MapLibre no consume f32; sin estándar de metadatos (TileJSON 3 no lo modela) | Planet + DevSeed | Baja | Medio-alto si se acompaña de consumidor WASM propio | Formato huérfano sin cliente |
| **Híbrido mismo código WASM + servidor**: el cliente calcula slope/hillshade/índices desde COG; el servidor materializa fill/flow/TWI | cog-tiler-wasm y GeoLibre hacen lo local; nadie hace el handoff local↔global con un solo motor | opengeos, terrano | Alta (I/O por rangos en WASM, límite 4 GiB, política de decisión) | Muy alto: es la tesis "analysis-first" completa | Complejidad; competir con opengeos en velocidad de iteración |
| **Benchmark reproducible vs TiTiler 2.4 / xpublish-tiles / tileserver-rs** | Solo claims de README (10×, 4-10×) sin scripts | DevSeed (tile-benchmarking) | Baja | Alto para el paper y credibilidad | Resultado desfavorable en COG remoto frío |
| **Cache de tiles compartido con CDN** (ETag por hash de fuente+operador, `Cache-Control`, warm por job) | Todos delegan al CDN; ninguno emite claves canónicas por operador | — | Baja-media | Medio: reduce egress y cold-start | Invalidación con fuentes mutables |
| **Tiles de embeddings/inferencia** (PCA/similaridad de AlphaEarth/Tessera por tile, `formula` sobre 64 bandas) | GeoLibre visualiza 3 bandas de 64 y descarga; nadie sirve PCA/cosine al vuelo | opengeos, Google, zephr | Media | Alto y visible (GeoAI) | Datasets pesados; egress alto |
| **GeoZarr/Zarr como fuente** (multiscales) | xpublish-tiles y rio-tiler 9.4 ya lo hacen; SurtGIS lee Zarr desde 0.6 | Earthmover, DevSeed | Media | Medio: paridad, no ventaja | GeoZarr aún sin aprobación OGC |
| **OGC API Tiles + DGGS** como segunda malla | Tiles 1.0 aprobado; DGGS Part 1 aprobado con casi cero implementaciones | Ecere | Media-alta | Medio (nicho científico) | Demanda incierta |

Lectura escéptica final: la ventaja de SurtGIS no está en "tiles dinámicos" (TiTiler lo resolvió y ahora acelera con Rust por debajo), ni en "un binario Rust" (Martin y tileserver-rs lo intentan, inmaduros), sino en la combinación **operador global materializado + operador local al vuelo + mismo motor en WASM**, que nadie tiene y que OGC Processes Part 3 apenas describe en borrador.


## Anexo F. Incertidumbre, diferenciabilidad, DGGS y provenance

Nota metodológica: MDPI y ScienceDirect devolvieron 403 y el presupuesto de búsqueda se agotó; las afirmaciones de esas fuentes se apoyan en snippets de búsqueda (vol./año inferido del DOI).

### 1. Cuantificación de incertidumbre (UQ) en terreno e hidrología

**Lo que existe.** El método canónico sigue siendo Monte Carlo con campos de error espacialmente autocorrelados (Hunter & Goodchild 1997; Wechsler & Kroll 2006; capítulo de Temme et al. en *Geomorphometry*, 2008). En 2023-2026 la literatura aplicada lo usa, pero como *script ad hoc*, no como primitiva:

- Jesna, Bhallamudi & Sudheer, *SERRA*, 19-ago-2026: 100 realizaciones de error SRTM por simulación gaussiana condicional (gstat/R), ensamble en HEC-RAS 1D; el NSE de profundidad sube de 0,18 (determinista) a 0,72 (media del ensamble). https://link.springer.com/article/10.1007/s00477-026-03337-5
- Nguyen et al., *WRR* 2025: Monte Carlo sobre alineación de grilla en LISFLOOD-FP, mostrando que hasta la convención de la grilla es fuente de incertidumbre. https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR038919
- Zhao et al., *NHESS* 2020 (aún referencia): DEM como campo aleatorio + ensamble de realizaciones para modelos de flujo tipo deslizamiento. https://nhess.copernicus.org/articles/20/1441/2020/

**Herramientas.** Ninguna propaga por construcción a derivados:
- **xDEM** (Hugonnet et al. 2022): estima heterocedasticidad y variograma multi-rango del error, y propaga solo a *promedios espaciales*; su propia documentación remite a GSTools para simular campos y derivar atributos de terreno. https://xdem.readthedocs.io/en/stable/uncertainty.html
- **WhiteboxTools** `StochasticDepressionAnalysis`: turning bands + RMSE + rango → probabilidad de depresión por celda. Es la única primitiva Monte Carlo "de fábrica" encontrada, y cubre solo depresiones. https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html
- **GRASS** `r.random.surface`: genera superficies aleatorias con dependencia espacial "para determinar cómo el error inherente afecta los análisis", pero el usuario arma el loop. https://grass.osgeo.org/grass-stable/manuals/r.random.surface.html
- **spup** (R) hace propagación genérica; RichDEM y Landlab no tienen nada equivalente.

**Brecha:** no hay motor que entregue `slope_mean + slope_sd` o `P(celda es cauce)` como salida nativa con RAM acotada. Los papers lo hacen con 100 realizaciones en R/Python y HEC-RAS.

### 2. Diferenciabilidad

**Hidrología diferenciable** está madura en Python: δHBV-globe1.0-hydroDL (*GMD* 17, 2024, https://gmd.copernicus.org/articles/17/7181/2024/), δHBV1.1p (Zenodo 2026) y Song et al., *WRR* 2026 (https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR040414); un framework Python para modelado hidrológico diferenciable apareció en *EMS* 2026 (https://www.sciencedirect.com/science/article/abs/pii/S1364815226000423). Todo es hidrología de cuenca agregada, no ráster.

**Aguas someras diferenciables:** Liu & Song, *WRR* 2025 (Hydrograd, SWE "universales", inversión de Manning n por AD; arXiv 18-feb-2025, https://arxiv.org/abs/2502.12396); **Inunda** (Zhi Li, arXiv 10-jul-2026): SWE inercial-local en PyTorch, GPU, millones de celdas, calibra rugosidad, lecho y conductividad por descenso de gradiente. https://arxiv.org/abs/2607.09614. Prototipos, no herramientas de producción.

**Flujos de detritos:** cero AD real. r.avaflow v4 (*GMD*, 10-dic-2025) calibra por "prueba y error iterativa" y no aborda UQ (https://gmd.copernicus.org/articles/18/9879/2025/). La "inversión full-flowfield" de Voellmy (*Geosciences* 16(9):378, sep-2026) usa algoritmo genético + gradiente por diferencias finitas proyectadas, no adjunto. JAX-MPM (arXiv 2025) y los GNN diferenciables para flujos granulares son surrogates, no solvers 2D de terreno. **Fastscape/fastscapelib no tiene versión diferenciable publicada** (búsqueda negativa).

**Rust/Enzyme:** `std::autodiff` está "recién habilitado en nightly" según la meta de proyecto Rust 2026 (https://goals.rust-lang.org/2026/high-level-ml.html), pero el tracking issue #124509 sigue experimental, sin RFC, y el propio equipo advierte que Enzyme es un incubador LLVM con más bugs de lo habitual; el flag `-Zautodiff=Enable` requiere toolchain con Enzyme (https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/autodiff.html; GSoC 2025 TypeTrees, 5-sep-2025). Conclusión: no es base para un producto en 2026; sí para un experimento.

### 3. Grillas globales discretas (DGGS) y geometría esferoidal

- **OGC API – DGGS Part 1: Core** aprobado (21-038r1, 2025); **DGGAL** implementa 16+ DGGRS (ISEA/IVEA 3H/7H/4R/9R, rHEALPix, HEALPix, GNOSIS) con bindings C, Python, **Rust** y WASM, pero sin operaciones de análisis (solo navegación de zonas). https://dggal.org/ · h3o (H3 puro Rust) v0.10.0, may-2026. https://crates.io/crates/h3o
- **Hidrología en DGGS sí existe:** Liao et al., *ESSD* 17, 13-may-2025: elevación, pendiente, dirección de flujo (modelo "mesh-independent"), área drenada y distancia calculadas nativamente en ISEA3H niveles 10-13 con HexWatershed v3 + DGGRID, Amazonas y Yukón. https://essd.copernicus.org/articles/17/2035/2025/ · hydrohex (Python, v0.2.0, D6 y D∞ en H3, "early-stage research software"). https://github.com/ChocopieKewpie/hydrohex · Tesis U. Calgary sobre ISEA3H con flujo y acumulación.
- **Esferoidal:** Florinsky (Trans. GIS 2017; *Digital Terrain Analysis* 3ª ed., Elsevier 2025) sigue siendo el único cuerpo teórico; Guth 2021 (Trans. GIS) da fórmulas para celdas no cuadradas. Ningún motor open source (GDAL, GRASS, WBT, SAGA) implementa pendiente/curvatura sobre grilla equiangular con métrica esferoidal; GEE lo hace "adaptado" (Safanelli 2020).

### 4. Reproducibilidad y provenance

- **PROV/RO-Crate:** Workflow Run RO-Crate (*PLOS One* 2024, alineado con PROV) https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0309210; **GeoPROV** (IJGI 15(6):272, 2026) perfila PROV + GeoSPARQL para cadenas de datos espaciales; ESDPKI (IJGI 15(5):182, 2026) con IRIs versionados. Todo a nivel de catálogo/knowledge graph, no dentro del archivo.
- **STAC:** la extensión `processing` v1.2.0 (candidate) tiene `lineage`, `software`, `expression`, `datetime`, `version`, pero **no hashes de inputs ni valores de parámetros**. https://github.com/stac-extensions/processing · STACD (PROPL/SPLASH 2025) agrega DAGs y recomputación selectiva sobre Airflow. https://dl.acm.org/doi/10.1145/3759536.3763803
- **Icechunk 2.0** (Rust, Earthmover): almacenamiento transaccional/versionado para Zarr; provenance de datos, no de cómputo. https://icechunk.io/
- **Determinismo numérico:** Yang et al., arXiv 10-sep-2026: orden de reducción como causa primaria de no-determinismo bit a bit, ~20 % de costo por orden fijo. https://arxiv.org/abs/2609.11356. En GIS nadie declara reproducibilidad bit a bit multi-hilo como propiedad verificada.
- **Software papers:** JOSS 2025-26 publica librerías Python (GeoWombat, Solshade); *GMD* publica r.avaflow v4; *EMS* publica frameworks diferenciables. Lo que falta: motores que declaren *contratos* verificables (nodata, CRS, determinismo, provenance embebida) y validación cruzada sistemática contra referencias, que es justo lo que SurtGIS ya hace.

### 5. Oportunidades para adelantarse

1. **Ejecución incierta nativa** (`--uncertainty rmse=,range=,n=` en slope/TWI/flow-acc/HAND/stream): SGS/turning bands en streaming, acumuladores de Welford por celda → media, sd y probabilidad con RAM O(1 realización). Hoy: scripts R + HEC-RAS (SERRA 2026), WBT solo para depresiones. Riesgo: costo n×; validar contra gstat.
2. **Provenance embebida y verificable en cada GeoTIFF/COG**: tag TIFF con hash BLAKE3 de inputs, parámetros, versión, semilla y `surtgis verify` que recompute; exportable a STAC `processing` + PROV-O. Hoy nadie lo hace a nivel de archivo (STAC no tiene hashes). Dificultad baja; alto impacto reputacional.
3. **Pendiente/curvatura/hillshade esferoidales sin reproyectar** (Florinsky 2017 + Guth 2021) para GLO-30/NASADEM globales, validadas contra reproyección local UTM. Nadie lo ofrece en un motor; riesgo bajo, impacto en usuarios de DEM globales.
4. **Hidrología sobre H3/ISEA vía h3o o DGGAL-Rust**: D6/D∞ y acumulación nativas, compatibles con OGC API DGGS. Competencia: HexWatershed (ESSD 2025), hydrohex (0.2). Dificultad media (rellenado/breaching en hexágonos).
5. **Determinismo bit a bit declarado y testeado**: reducciones con orden fijo en Rayon y test CI de igualdad 1 vs N hilos; SurtGIS ya valida bit a bit contra GDAL, esto lo convierte en contrato. Riesgo: ~20 % en reducciones globales.
6. **Solver de detritos con gradiente** (calibración de μ, ξ, entrainment): primero adjunto discreto manual o diferencias finitas paralelas con checkpointing (como *Geosciences* 2026) y luego experimento con `std::autodiff`. Hoy r.avaflow es prueba y error; Inunda es agua, no detritos. Riesgo alto: Enzyme inestable, no-diferenciabilidad de frentes secos.
7. **Ensamble Bayesiano ligero en flow** (LHS/ABC sobre μ-ξ con huella IoU) como puente entre 1 y 6: entrega mapas de probabilidad de alcance con costo controlado.
8. **Contrato STAC/PROV de salida del servidor de tiles y jobs**: cada job materializa COG + item STAC con `processing` + hash, cerrando el ciclo con 2.

| Oportunidad | Estado del arte hoy | Quién | Dificultad | Impacto | Riesgo |
|---|---|---|---|---|---|
| Ejecución incierta (MC por construcción, RAM acotada) | Scripts R/Python ad hoc; WBT solo depresiones; xDEM solo promedios | SERRA 2026, xDEM, WBT | Media | Muy alto (primer motor UQ-nativo) | Costo n×; calibrar error espacial |
| Provenance embebida + `verify` | STAC `processing` sin hashes; PROV en catálogos | GeoPROV 2026, STACD 2025 | Baja | Alto | Bajo; definir esquema estable |
| Geomorfometría esferoidal sin reproyectar | Teoría Florinsky/Guth; ningún motor OSS | Florinsky 2025 (libro) | Media | Alto para DEM globales | Validación de curvaturas |
| Hidrología en DGGS (H3/ISEA) | HexWatershed (ESSD 2025), hydrohex 0.2 | Liao et al.; DGGAL Rust | Media-alta | Medio-alto | Nicho aún pequeño |
| Determinismo bit a bit multi-hilo verificado | Nadie en GIS lo declara | Yang et al. 2026 (GPU) | Baja-media | Medio (credibilidad científica) | ~20 % en reducciones |
| Solver de detritos diferenciable | r.avaflow prueba y error; FD proyectada 2026; Inunda (agua) | Mergili; Zhi Li | Alta | Muy alto (calibración N4) | Enzyme experimental; frentes secos |
| Ensamble Bayesiano en flow | Back-análisis manual | r.avaflow, RAMMS | Media | Alto | Costo de cómputo |
| Jobs del servidor con STAC/PROV | Sin equivalente en tile servers | — | Baja | Medio | Bajo |

**Lectura escéptica:** las tres primeras son diferenciadores reales y alcanzables en 2026-27 con el motor actual; DGGS es apuesta de nicho que compite con un dataset ESSD ya publicado; el solver diferenciable es el titular más atractivo y el de mayor probabilidad de fracaso técnico si se depende de Enzyme.
