# surtgis

High-performance geospatial analysis in the browser via WebAssembly.

SurtGIS compiles 33 terrain, hydrology, imagery, morphology and statistics algorithms to WASM, giving you native-speed raster processing with zero server dependencies.

**[Live Demo](https://franciscoparrao.github.io/surtgis)** | **[GitHub](https://github.com/franciscoparrao/surtgis)**

## Install

```bash
npm install surtgis
```

## Quick Start

### Browser (with bundler)

```js
import { SurtGIS } from "surtgis/surtgis.js";

const gis = await SurtGIS.init();

// Load a GeoTIFF as Uint8Array
const dem = new Uint8Array(await fetch("dem.tif").then(r => r.arrayBuffer()));

// Compute slope
const slopeResult = gis.slope(dem, { units: "degrees" });

// Compute hillshade
const hillshade = gis.hillshade(dem, { azimuth: 315, altitude: 45 });
```

### Web Worker (off main thread)

```js
import { SurtGISWorker } from "surtgis/surtgis-worker.js";

const gis = await SurtGISWorker.init();

// All methods return Promise<Uint8Array>
const result = await gis.slope(dem, { units: "degrees" });

gis.terminate(); // when done
```

### Low-level bindings

```js
import { slope, hillshade_compute } from "surtgis";

const slopeBytes = slope(demBytes, "degrees");
const hillshadeBytes = hillshade_compute(demBytes, 315, 45);
```

## API Reference

All functions accept and return `Uint8Array` (GeoTIFF bytes).

### Terrain (17)

| Method | Options | Description |
|--------|---------|-------------|
| `slope(dem, opts?)` | `{ units: "degrees"\|"percent" }` | Slope gradient |
| `aspect(dem)` | — | Aspect in degrees (0-360) |
| `hillshade(dem, opts?)` | `{ azimuth: 315, altitude: 45 }` | Analytical hillshade |
| `multidirectionalHillshade(dem)` | — | 6-azimuth combined hillshade |
| `curvature(dem, opts?)` | `{ type: "general"\|"profile"\|"plan" }` | Surface curvature |
| `tpi(dem, opts?)` | `{ radius: 3 }` | Topographic Position Index |
| `tri(dem)` | — | Terrain Ruggedness Index |
| `twi(dem)` | — | Topographic Wetness Index |
| `geomorphons(dem, opts?)` | `{ flatness: 1.0, radius: 10 }` | Landform classification |
| `northness(dem)` | — | Cosine of aspect |
| `eastness(dem)` | — | Sine of aspect |
| `dev(dem, opts?)` | `{ radius: 10 }` | Deviation from Mean Elevation |
| `shapeIndex(dem)` | — | Shape index |
| `curvedness(dem)` | — | Surface curvedness |
| `skyViewFactor(dem, opts?)` | `{ directions: 16, radius: 10 }` | Sky View Factor |
| `uncertaintySlope(dem, opts?)` | `{ demRmse: 1.0 }` | Slope uncertainty (RMSE) |
| `ssa2dDenoise(dem, opts?)` | `{ window: 10, components: 3 }` | 2D-SSA denoising |

### Hydrology (5)

| Method | Options | Description |
|--------|---------|-------------|
| `fillSinks(dem)` | — | Fill depressions |
| `priorityFlood(dem)` | — | Priority-flood filling (Barnes 2014) |
| `flowDirectionD8(dem)` | — | D8 flow direction |
| `flowAccumulationD8(fdir)` | — | Flow accumulation from D8 directions |
| `hand(dem, opts?)` | `{ streamThreshold: 1000 }` | Height Above Nearest Drainage |

### Imagery (4)

| Method | Options | Description |
|--------|---------|-------------|
| `ndvi(nir, red)` | — | Normalized Difference Vegetation Index |
| `ndwi(green, nir)` | — | Normalized Difference Water Index |
| `savi(nir, red, opts?)` | `{ lFactor: 0.5 }` | Soil-Adjusted Vegetation Index |
| `normalizedDifference(a, b)` | — | Generic (A-B)/(A+B) |

### Morphology (4)

| Method | Options | Description |
|--------|---------|-------------|
| `erode(dem, opts?)` | `{ radius: 1 }` | Morphological erosion |
| `dilate(dem, opts?)` | `{ radius: 1 }` | Morphological dilation |
| `opening(dem, opts?)` | `{ radius: 1 }` | Opening (erode + dilate) |
| `closing(dem, opts?)` | `{ radius: 1 }` | Closing (dilate + erode) |

### Statistics (3)

| Method | Options | Description |
|--------|---------|-------------|
| `focalMean(dem, opts?)` | `{ radius: 3 }` | Focal mean |
| `focalStd(dem, opts?)` | `{ radius: 3 }` | Focal standard deviation |
| `focalRange(dem, opts?)` | `{ radius: 3 }` | Focal range (max - min) |

## Bundle Size

| Component | Size | Gzipped |
|-----------|------|---------|
| WASM binary | ~570 KB | ~195 KB |
| JS glue | ~3 KB | ~1 KB |

## License

MIT OR Apache-2.0
