# WASM + Web Demo + npm Package — Implementation Log

**Fecha**: 2026-02-02
**Branch**: `feature/wasm`

---

## Plan Ejecutado

### Objetivo
Completar la web demo WASM y preparar el npm package con todas las funciones exportadas, Web Worker integration, y CI/CD para GitHub Pages.

### Estado Previo

| Componente | Estado |
|---|---|
| WASM crate (`crates/wasm/src/lib.rs`) | 100% — 33 funciones con `#[wasm_bindgen]` |
| pkg/ (wasm-pack output) | Incompleto — solo 5 funciones exportadas por build anterior con `cargo build` |
| Web demo (`web/`) | 60% — solo 5 algoritmos en UI |
| Web Worker | No existia |
| npm publish config | No existia |
| GitHub Pages deploy | No existia |

---

## Pasos Implementados

### Paso 1: Rebuild WASM con wasm-pack

```bash
wasm-pack build crates/wasm --target bundler --release --out-dir ../../pkg
```

**Resultado**: 33 funciones exportadas en `pkg/surtgis_wasm.d.ts`.

**Nota importante**: wasm-pack 0.14+ con target `bundler` genera modulos que se auto-inicializan al importar (no hay `default` export ni funcion `init()`). El WASM se carga sincrono via `import * as wasm from "./surtgis_wasm_bg.wasm"`.

WASM binary: 572 KB (195 KB gzipped).

### Paso 2: Script post-build para npm metadata

**Archivo creado**: `scripts/post_wasm_build.sh`

El script parchea `pkg/package.json` despues de cada `wasm-pack build` para agregar:
- `name: "surtgis"` (en vez de `surtgis-wasm`)
- keywords, repository, homepage
- files extra (wrapper, worker, README)

### Paso 3: API Wrapper Ergonomica

**Archivos creados**:
- `pkg/surtgis.js` — clase `SurtGIS` con metodos tipados
- `pkg/surtgis.d.ts` — TypeScript declarations

La clase agrupa las 33 funciones en namespaces logicos:
- **Terrain** (17): slope, aspect, hillshade, multidirectionalHillshade, curvature, tpi, tri, twi, geomorphons, northness, eastness, dev, shapeIndex, curvedness, skyViewFactor, uncertaintySlope, ssa2dDenoise
- **Hydrology** (5): fillSinks, priorityFlood, flowDirectionD8, flowAccumulationD8, hand
- **Imagery** (4): ndvi, ndwi, savi, normalizedDifference
- **Morphology** (4): erode, dilate, opening, closing
- **Statistics** (3): focalMean, focalStd, focalRange

Uso:
```js
import { SurtGIS } from "surtgis/surtgis.js";
const gis = await SurtGIS.init();
const result = gis.slope(demBytes, { units: "degrees" });
```

### Paso 4: Web Worker Integration

**Archivos creados**:
- `pkg/worker.js` — script que corre en el Worker thread
- `pkg/surtgis-worker.js` — clase `SurtGISWorker` para el main thread
- `pkg/surtgis-worker.d.ts` — types

Protocolo: request/response con IDs, zero-copy via Transferable buffers.

Uso:
```js
import { SurtGISWorker } from "surtgis/surtgis-worker.js";
const gis = await SurtGISWorker.init();
const result = await gis.slope(demBytes); // Promise<Uint8Array>
gis.terminate();
```

### Paso 5: Expansion Web Demo (5 → 33 algoritmos)

**Archivos modificados**:

| Archivo | Cambios |
|---|---|
| `web/src/lib/wasm.js` | Import 33 funciones, switch con 33 cases |
| `web/src/components/AlgoPanel.svelte` | 5 grupos colapsables, 33 algoritmos con controles de parametros |
| `web/src/App.svelte` | Layout side-by-side (input + resultado), DEM preview al cargar, SCHEME_MAP para 33 algos, soporte dual-band |
| `web/src/lib/colormap.js` | Nuevos esquemas: `blue_white_red`, `geomorphons`, `water`, `accumulation` |

**Features nuevos de la UI**:
- Vista dual: DEM original a la izquierda, resultado a la derecha
- Preview inmediato del DEM al cargar (sin necesidad de correr algoritmo)
- Grupos colapsables para organizar 33 algoritmos
- Badge con conteo de algoritmos por grupo
- Esquemas de color especializados (divergente para curvatura/TPI, geomorphons para landforms, agua para TWI/HAND)
- Upload de segunda banda para algoritmos de imagery (NDVI, NDWI, SAVI, Normalized Diff)
- Parametros individuales: curvature type, geomorphons flatness/radius, SVF directions, SSA window/components, etc.
- Parametros compartidos: morphology kernel radius, focal window radius

### Paso 6: GitHub Pages Deploy

**Archivo creado**: `.github/workflows/deploy.yml`

Workflow:
1. wasm-pack build
2. npm ci + vite build en `web/`
3. Deploy a GitHub Pages via `actions/deploy-pages@v4`

Trigger: push a `main` o manual dispatch.

Soporte `BASE_URL` en `vite.config.js` para paths relativos en GH Pages.

### Paso 7: README npm

**Archivo creado**: `pkg/README.md`

Incluye:
- Quick start (browser, Web Worker, low-level)
- API reference completa (tabla con 33 funciones)
- Bundle size info
- License

### Paso 8: CI actualizado

**Archivo modificado**: `.github/workflows/ci.yml`

Cambios:
- Job `wasm`: `cargo build` → `wasm-pack build` + verificacion de conteo de funciones (>=29)
- Job `web-build` (nuevo): wasm-pack build + npm ci + vite build

---

## Verificacion

| Check | Resultado |
|---|---|
| `wasm-pack build` | OK — 33 funciones exportadas |
| `grep -c "export function" pkg/surtgis_wasm.d.ts` | 33 |
| `npm run build` (web) | OK — dist/ generado sin errores |
| WASM binary size | 572 KB (195 KB gzip) |
| Total JS bundle | 157 KB (40 KB gzip) |
| Switch cases en wasm.js | 33 |

---

## Resumen de Archivos

### Archivos Creados

| Archivo | Descripcion |
|---|---|
| `scripts/post_wasm_build.sh` | Parchea package.json post wasm-pack |
| `pkg/surtgis.js` | API wrapper ergonomica (clase SurtGIS) |
| `pkg/surtgis.d.ts` | TypeScript types para wrapper |
| `pkg/worker.js` | Web Worker script |
| `pkg/surtgis-worker.js` | Clase SurtGISWorker (main thread) |
| `pkg/surtgis-worker.d.ts` | TypeScript types para worker |
| `pkg/README.md` | Documentacion npm |
| `.github/workflows/deploy.yml` | GitHub Pages deploy |

### Archivos Modificados

| Archivo | Cambios |
|---|---|
| `web/src/lib/wasm.js` | 5 → 33 algoritmos |
| `web/src/components/AlgoPanel.svelte` | UI completa con 5 grupos y 33 algos |
| `web/src/App.svelte` | Side-by-side layout, DEM preview, dual-band support |
| `web/src/lib/colormap.js` | 4 esquemas nuevos |
| `web/vite.config.js` | Soporte BASE_URL |
| `.github/workflows/ci.yml` | wasm-pack + web-build jobs |

### Archivos Regenerados (wasm-pack)

| Archivo | Descripcion |
|---|---|
| `pkg/surtgis_wasm_bg.wasm` | Binary WASM (572 KB) |
| `pkg/surtgis_wasm.js` | JS glue auto-generado |
| `pkg/surtgis_wasm.d.ts` | Types auto-generados (33 funciones) |
| `pkg/surtgis_wasm_bg.js` | Background JS bindings |
| `pkg/package.json` | npm metadata (parcheado) |

---

## Pendiente

- [ ] npm publish (`npm publish` requiere credenciales)
- [ ] Test manual en browser con DEM real
- [ ] Habilitar GitHub Pages en settings del repo
- [ ] Agregar `package-lock.json` al repo para `npm ci` en CI
