/**
 * Web Worker wrapper — same API as SurtGIS but all methods return Promise<Uint8Array>.
 */

import type {
  SlopeOptions,
  HillshadeOptions,
  CurvatureOptions,
  RadiusOptions,
  GeomorphonsOptions,
  SvfOptions,
  UncertaintyOptions,
  Ssa2dOptions,
  HandOptions,
  SaviOptions,
} from "./surtgis.js";

export class SurtGISWorker {
  static init(workerUrl?: string | URL): Promise<SurtGISWorker>;
  terminate(): void;

  // Terrain
  slope(dem: Uint8Array, opts?: SlopeOptions): Promise<Uint8Array>;
  aspect(dem: Uint8Array): Promise<Uint8Array>;
  hillshade(dem: Uint8Array, opts?: HillshadeOptions): Promise<Uint8Array>;
  multidirectionalHillshade(dem: Uint8Array): Promise<Uint8Array>;
  curvature(dem: Uint8Array, opts?: CurvatureOptions): Promise<Uint8Array>;
  tpi(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  tri(dem: Uint8Array): Promise<Uint8Array>;
  twi(dem: Uint8Array): Promise<Uint8Array>;
  geomorphons(dem: Uint8Array, opts?: GeomorphonsOptions): Promise<Uint8Array>;
  northness(dem: Uint8Array): Promise<Uint8Array>;
  eastness(dem: Uint8Array): Promise<Uint8Array>;
  dev(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  shapeIndex(dem: Uint8Array): Promise<Uint8Array>;
  curvedness(dem: Uint8Array): Promise<Uint8Array>;
  skyViewFactor(dem: Uint8Array, opts?: SvfOptions): Promise<Uint8Array>;
  uncertaintySlope(dem: Uint8Array, opts?: UncertaintyOptions): Promise<Uint8Array>;
  ssa2dDenoise(dem: Uint8Array, opts?: Ssa2dOptions): Promise<Uint8Array>;

  // Hydrology
  fillSinks(dem: Uint8Array): Promise<Uint8Array>;
  priorityFlood(dem: Uint8Array): Promise<Uint8Array>;
  flowDirectionD8(dem: Uint8Array): Promise<Uint8Array>;
  flowAccumulationD8(fdir: Uint8Array): Promise<Uint8Array>;
  hand(dem: Uint8Array, opts?: HandOptions): Promise<Uint8Array>;

  // Imagery
  ndvi(nir: Uint8Array, red: Uint8Array): Promise<Uint8Array>;
  ndwi(green: Uint8Array, nir: Uint8Array): Promise<Uint8Array>;
  savi(nir: Uint8Array, red: Uint8Array, opts?: SaviOptions): Promise<Uint8Array>;
  normalizedDifference(a: Uint8Array, b: Uint8Array): Promise<Uint8Array>;

  // Morphology
  erode(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  dilate(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  opening(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  closing(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;

  // Statistics
  focalMean(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  focalStd(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
  focalRange(dem: Uint8Array, opts?: RadiusOptions): Promise<Uint8Array>;
}

export default SurtGISWorker;
