/**
 * High-level ergonomic wrapper for SurtGIS WASM bindings.
 */

export interface SlopeOptions {
  units?: "degrees" | "percent";
}
export interface HillshadeOptions {
  azimuth?: number;
  altitude?: number;
}
export interface CurvatureOptions {
  type?: "general" | "profile" | "plan";
}
export interface RadiusOptions {
  radius?: number;
}
export interface GeomorphonsOptions {
  flatness?: number;
  radius?: number;
}
export interface SvfOptions {
  directions?: number;
  radius?: number;
}
export interface UncertaintyOptions {
  demRmse?: number;
}
export interface Ssa2dOptions {
  window?: number;
  components?: number;
}
export interface HandOptions {
  streamThreshold?: number;
}
export interface SaviOptions {
  lFactor?: number;
}

export class SurtGIS {
  /** Initialize the WASM module. */
  static init(): Promise<SurtGIS>;

  // Terrain
  slope(dem: Uint8Array, opts?: SlopeOptions): Uint8Array;
  aspect(dem: Uint8Array): Uint8Array;
  hillshade(dem: Uint8Array, opts?: HillshadeOptions): Uint8Array;
  multidirectionalHillshade(dem: Uint8Array): Uint8Array;
  curvature(dem: Uint8Array, opts?: CurvatureOptions): Uint8Array;
  tpi(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  tri(dem: Uint8Array): Uint8Array;
  twi(dem: Uint8Array): Uint8Array;
  geomorphons(dem: Uint8Array, opts?: GeomorphonsOptions): Uint8Array;
  northness(dem: Uint8Array): Uint8Array;
  eastness(dem: Uint8Array): Uint8Array;
  dev(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  shapeIndex(dem: Uint8Array): Uint8Array;
  curvedness(dem: Uint8Array): Uint8Array;
  skyViewFactor(dem: Uint8Array, opts?: SvfOptions): Uint8Array;
  uncertaintySlope(dem: Uint8Array, opts?: UncertaintyOptions): Uint8Array;
  ssa2dDenoise(dem: Uint8Array, opts?: Ssa2dOptions): Uint8Array;

  // Hydrology
  fillSinks(dem: Uint8Array): Uint8Array;
  priorityFlood(dem: Uint8Array): Uint8Array;
  flowDirectionD8(dem: Uint8Array): Uint8Array;
  flowAccumulationD8(fdir: Uint8Array): Uint8Array;
  hand(dem: Uint8Array, opts?: HandOptions): Uint8Array;

  // Imagery
  ndvi(nir: Uint8Array, red: Uint8Array): Uint8Array;
  ndwi(green: Uint8Array, nir: Uint8Array): Uint8Array;
  savi(nir: Uint8Array, red: Uint8Array, opts?: SaviOptions): Uint8Array;
  normalizedDifference(a: Uint8Array, b: Uint8Array): Uint8Array;

  // Morphology
  erode(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  dilate(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  opening(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  closing(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;

  // Statistics
  focalMean(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  focalStd(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
  focalRange(dem: Uint8Array, opts?: RadiusOptions): Uint8Array;
}

export default SurtGIS;
