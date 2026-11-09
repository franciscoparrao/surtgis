# SurtGIS Server — documento de diseño

**Fecha**: 2026-09-07 · **Estado**: propuesta, pre-implementación ·
**Decisión pendiente**: aprobar alcance M0/M1 y nombre

## 1. Tesis

Un servidor de tiles **analysis-first** en Rust: el motor SurtGIS (127
algoritmos, lectura COG/GeoTIFF/ECW, reproyección, colormaps) expuesto
detrás de endpoints HTTP estándar, de modo que un tile no es un byte
almacenado sino **el resultado de un cómputo sobre datos remotos, calculado
al momento de la petición**.

No es "GeoServer en Rust". Es la capa que ningún servidor —en ningún
lenguaje— ofrece hoy: análisis de terreno, hidrología e índices espectrales
como parámetros de una URL de tile.

El argumento arquitectónico que lo distingue del resto del ecosistema: **el
mismo motor ya corre en el navegador** (surtgis-wasm, 57 funciones, visor de
territorio-digital). Cliente y servidor comparten binario de cómputo; el
servidor existe para cuando el cliente no puede (móvil, datos masivos,
interop con QGIS/ArcGIS, cacheo compartido). Ningún stack tiene esa
propiedad: TiTiler no corre en browser, GeoServer menos.

## 2. Posicionamiento

| Proyecto | Lenguaje | Qué sirve | Computa | Nicho |
|---|---|---|---|---|
| GeoServer | Java | WMS/WFS/WCS/WPS completo | WPS (pesado, torpe) | El estándar regulatorio/enterprise |
| TiTiler | Python | Tiles dinámicos desde COG | Álgebra de bandas, hillshade básico | Cloud-native raster |
| [Martin](https://maplibre.org/martin/) | Rust | Tiles vectoriales (PostGIS/PMTiles/MBTiles) | No | Vector tiles, maduro (MapLibre) |
| [BBOX](https://www.bbox.earth/) | Rust | OGC API modular | Delega a MapServer/QGIS Server | OGC API generalista |
| tileserver-rs | Rust | PMTiles/MBTiles/PostGIS/COG | No | Servir bytes rápido |
| **SurtGIS Server** | Rust | Tiles raster dinámicos | **127 algoritmos + fórmulas ASI** | **Análisis al vuelo** |

Capacidades que nadie más tiene (en ningún lenguaje):

1. **Terreno/hidrología al vuelo**: hillshade multidireccional, curvaturas
   Florinsky, TPI, geomorphons, SVF… como parámetro de query sobre un COG
   remoto arbitrario.
2. **Fórmulas del catálogo ASI por URL**: `?formula=(N-R)/(N+R)` — el
   evaluador de `index_builder` (commit e1d03c5) directo al endpoint. Los
   197 índices sin escribir código.
3. **Tiles desde ECW**: con `surtgis-ecw` (commit 0acdd6b) seríamos el único
   servidor del mundo que sirve ECW sin el SDK de Hexagon. Caso de uso
   inmediato: los ortomosaicos Geomag de territorio-digital/GeoPatrol.
4. **Salidas de simulación como tiles**: frames de `surtgis-flow`/hydroflux
   servidos por HTTP estándar → GEODEO (Unreal) y visores web los consumen
   sin formato propietario.

## 3. Principio de diseño rector: dinámico vs materializado

La restricción técnica central, declarada desde el día uno para no prometer
lo imposible:

**Solo los operadores locales y focales tilean al vuelo.** Un tile de
hillshade se computa leyendo su ventana + un buffer perimetral (gutter) y
nada más. Un tile de flow accumulation NO: el valor de una celda depende de
toda la cuenca aguas arriba — no existe descomposición por tiles
independientes.

| Clase | Ejemplos | Estrategia |
|---|---|---|
| Local (por celda) | índices espectrales, fórmula ASI, reclasificación, band math | **Dinámico**, gutter 0 |
| Focal (ventana fija) | slope, aspect, hillshade, curvaturas, TPI/TRI, focal stats | **Dinámico**, gutter = radio del kernel |
| Focal de radio grande | SVF, openness, DEV, geomorphons (radio 10–50) | **Dinámico con tope de radio**; sobre el tope, materializar |
| Global | fill sinks, flow accumulation, cuencas, HAND, streams | **Materializado**: job de precómputo → COG derivado → se sirve como fuente normal |

Los globales entran igual al producto, pero como **pipeline de
materialización** (`POST /jobs` → corre el algoritmo con la infraestructura
streaming existente → escribe COG → queda disponible como `?url=`), no como
mentira de "hidrología en tiempo real".

## 4. Arquitectura

### 4.1 Ubicación y empaquetado

- **Crate nuevo `crates/server`** (`surtgis-server`), `publish = false`
  inicial — mismo patrón que `flow` y `ecw`.
- **CLI**: subcomando `surtgis serve` tras feature opt-in `server` (patrón
  path-only establecido). Un solo binario para todo el motor, como Martin.
- **Stack**: `axum` + `tokio` (ya hay tokio en `cloud` para el COG reader
  asíncrono), `tower-http` para compresión/CORS/trazas.

### 4.2 Pipeline de un tile (todo con APIs que ya existen)

```
GET /tiles/{z}/{x}/{y}.png?url=<COG>&alg=hillshade&cmap=terrain
 1. Validar fuente contra la allowlist (§7)                    [nuevo]
 2. Tile XYZ → bbox WebMercator + gutter del algoritmo         [nuevo, aritmética]
 3. CogReader::open(url) + read_bbox::<f64>(bbox∪gutter)       [crates/cloud, existe]
    · fuentes .ecw → EcwReader::read_region                    [crates/ecw, existe]
 4. Reproyectar CRS fuente → EPSG:3857 si difiere              [proj4rs; mover el
    (inverse mapping bilinear, el de reproject.rs)              núcleo del handler
                                                                CLI a un módulo
                                                                compartido]
 5. Algoritmo sobre la ventana (slope, hillshade, formula…)    [crates/algorithms]
 6. Renderer::new(scheme) → RGBA (+ over/multiply para          [crates/colormap,
    fusión hillshade+color)                                     existe]
 7. rgba_to_png_bytes → respuesta con Cache-Control/ETag       [existe + headers]
```

La pieza 4 es la única refactorización real: extraer el inverse-mapping de
`crates/cli/src/handlers/reproject.rs` a un módulo reutilizable
(`core::warp` o similar) con API por-ventana.

### 4.3 Modelo de tiles

- **TileMatrixSet**: `WebMercatorQuad` (EPSG:3857), tiles 256×256. Es lo que
  consume MapLibre/Leaflet/OpenLayers/QGIS-XYZ. `WorldCRS84Quad` queda para
  M2 si OGC API lo pide.
- **Gutter**: cada algoritmo declara su radio de kernel; la lectura pide
  `bbox + radio·resolución` y el resultado se recorta. Los algoritmos
  focales de SurtGIS ya son puros sobre `Raster` → sin cambios.
- **Overviews**: `z` bajo → `read_bbox` ya elige el overview del COG
  (`output_shape_for`/`overviews()` existen). Para ECW, la pirámide wavelet
  hace lo mismo vía `read_region` con `number_x/y` reducidos.

## 5. API v0

### Núcleo (M1)

```
GET /tiles/{z}/{x}/{y}.png
    ?url=...            fuente (COG http(s), archivo local permitido)
    &alg=<nombre>       hillshade | slope | aspect | tpi | curvature:profile | …
    &formula=(N-R)/(N+R)  evaluador ASI (excluyente con alg)
    &bands=4,3          mapeo de nombres de fórmula → bandas de la fuente
    &cmap=terrain       esquema de crates/colormap
    &rescale=0,3000     dominio del colormap
    &params=azimuth:315 parámetros del algoritmo, k:v separados por coma

GET /tilejson?url=...&alg=...     TileJSON 3.0 (bounds/minzoom/maxzoom del COG)
GET /info?url=...                 metadata de la fuente (CogMetadata / EcwHeader)
GET /statistics?url=...&alg=...   stats de una ventana (para rescale automático)
GET /algorithms                   catálogo: nombre, clase, gutter, parámetros
GET /healthz
```

### Materialización (M1.5)

```
POST /jobs   {url, pipeline: [fill_sinks, flow_accumulation], output}
GET  /jobs/{id}          estado
→ el COG resultante se sirve con los endpoints normales
```

Sincrónico y de un worker al principio; sin colas distribuidas.

### OGC API (M2)

Subconjunto moderno, cero XML de la era SOAP:

- OGC API — Tiles (landing, conformance, tileset metadata) sobre los
  mismos handlers.
- OGC API — Processes como fachada de `/jobs`.
- **No** WMS/WFS/SLD clásicos. Interop con QGIS/ArcGIS vía XYZ (ambos lo
  consumen nativo) — WMS legacy solo si un usuario regulatorio lo exige, y
  entonces se evalúa delegar a BBOX en vez de implementarlo.

## 6. Cache

- **L1**: LRU en memoria de tiles renderizados, clave = hash(url, z/x/y,
  alg, params). Configurable en MB.
- **L2** (opcional, flag): disco, layout `{hash}/{z}/{x}/{y}.png` — sirve
  también como seed para pre-generar pirámides estáticas
  (`surtgis serve --pregen`).
- **Fuentes**: el `CogReader` ya cachea headers/tiles por conexión;
  mantener un pool de readers abiertos por URL (TTL corto).
- ETag = la clave L1; `Cache-Control: public, max-age` configurable.

## 7. Seguridad y operación (mínimos no negociables)

- **SSRF**: `?url=` es un vector clásico. Allowlist de prefijos en config
  (`--allow https://sentinel-cogs.s3..., /data/local/`), deny por defecto
  a IPs privadas/metadata endpoints. Sin allowlist configurada, solo
  archivos bajo `--root`.
- **Límites**: timeout por tile, tope de bytes leídos por petición, tope de
  gutter, semáforo de peticiones concurrentes al mismo origen.
- **Auth**: M1 = ninguna o bearer token estático (flag). OIDC/API keys es
  problema del reverse proxy del usuario, no nuestro — documentarlo así.
- **Observabilidad**: `tracing` + `/metrics` Prometheus (counter de tiles,
  histograma de latencia por algoritmo, hit-rate de cache).

## 8. Qué NO es (anti-alcance explícito)

- No WMS/WFS/WCS/SLD legacy, no GUI de administración, no catálogo de capas
  con estado: la "capa" es la URL de la fuente + parámetros.
- No tiles vectoriales (Martin lo resuelve; interoperar, no competir).
- No multi-tenancy ni gestión de usuarios.
- No reimplementación de STAC server (existe stac-rs; el nuestro consume
  STAC, no lo sirve).

## 9. Rendimiento: objetivo y plan de verificación

Objetivo M1 (hardware de referencia i7-1270P, fuente COG local):

| Operación | Objetivo p50 |
|---|---|
| Tile índice espectral (local) | < 15 ms |
| Tile hillshade/slope (focal 3×3) | < 25 ms |
| Tile primera petición COG remoto (frío) | < 300 ms |

Plan de benchmark honesto (estilo del paper EMS): mismo tile, misma fuente,
SurtGIS Server vs TiTiler (uvicorn, mismos workers) vs GeoServer WMS+SLD
hillshade, p50/p95/p99 bajo `oha`/`wrk`, frío y caliente. Números que
sobrevivan revisión — la ventaja esperable es grande justamente porque el
baseline Python paga serialización y GIL, pero hay que medirla, no
declamarla.

## 10. Roadmap

| Hito | Contenido | Estimación |
|---|---|---|
| **M0 PoC** | `surtgis serve` + `/tiles` con slope/hillshade/formula sobre COG local y remoto + cmap + demo MapLibre | 2–3 días |
| **M1 MVP** | ECW como fuente, gutter por algoritmo declarado, cache L1/L2, allowlist, límites, TileJSON, /info, /statistics, métricas | 1–2 semanas |
| **M1.5** | Jobs de materialización (hidrología) | 3–4 días |
| **M2** | OGC API Tiles/Processes, WorldCRS84Quad, conformance tests (ETS de OGC) | 1–2 semanas |
| **M3** | Benchmark formal vs TiTiler/GeoServer + nota de software (SoftwareX/JOSS/ESIN) | según venue |

Dependencia transversal: extraer el warp de `reproject.rs` a módulo
compartido (sirve además al propio CLI y a WASM a futuro).

## 11. Riesgos y preguntas abiertas

1. **Superficie de operación**: un servidor implica CVEs, deploys, soporte.
   Mitigación: alcance chico, deps auditadas (axum/tower son el estándar),
   y un usuario real desde el día uno (territorio-digital sirviendo los
   ECW de Geomag) para que no sea infraestructura huérfana.
2. **Foco del postdoc**: es tangencial a la línea hydroflux/papers. A favor:
   GEODEO y GeoPatrol lo consumen, y la nota de software es publicable. En
   contra: M2 (OGC conformance) es trabajo de estándar, no de ciencia.
   Recomendación: M0/M1 sí; M2 solo si aparece demanda concreta.
3. **TiTiler se mueve**: ya tiene `algorithms` (hillshade, contours). El
   wedge defendible no es "tenemos hillshade" sino la *profundidad* del
   catálogo (Florinsky, geomorphons, ASI completo), ECW, y el motor
   compartido browser/server. Posicionar así desde el README.
4. **Tokio en el árbol del CLI**: la feature `server` engorda el build.
   Aceptable por ser opt-in; medir el impacto en tiempo de compilación de
   CI (`--all-features`).
5. **Abierta**: ¿tiles f32 (formato `png16`/WebP lossless/COG-por-tile)
   para clientes que quieren datos, no imágenes? Postergarlo a demanda;
   el endpoint `/statistics` cubre el caso analítico simple.

## 12. Nombre

Opciones, en orden de preferencia:

1. **`surtgis-server`, comando `surtgis serve`** — descubrible, sin marca
   nueva que mantener, consistente con Martin (`martin`), y el SEO de
   "surtgis" capitaliza el paper EMS.
2. Nombre propio del portafolio (estilo Smelt/Anvil/Cantus): candidato
   natural **Bifrost** (el puente entre mundos — el motor y la web), pero
   colisiona con varios proyectos existentes del mismo nombre.

Recomendación: opción 1. El marketing vive en el tagline ("dynamic
terrain-analysis tile server"), no en el nombre.
