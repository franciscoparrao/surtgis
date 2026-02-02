/**
 * SurtGIS — ergonomic high-level wrapper over the auto-generated WASM bindings.
 *
 * Usage:
 *   import { SurtGIS } from "surtgis/surtgis.js";
 *   const gis = await SurtGIS.init();
 *   const result = gis.slope(demBytes, { units: "degrees" });
 *
 * Every function accepts Uint8Array (GeoTIFF bytes) and returns Uint8Array.
 */

import {
  slope as _slope,
  aspect_degrees as _aspect_degrees,
  hillshade_compute as _hillshade_compute,
  multidirectional_hillshade as _multidirectional_hillshade,
  curvature_compute as _curvature_compute,
  tpi_compute as _tpi_compute,
  tri_compute as _tri_compute,
  twi_compute as _twi_compute,
  geomorphons_compute as _geomorphons_compute,
  northness_compute as _northness_compute,
  eastness_compute as _eastness_compute,
  dev_compute as _dev_compute,
  shape_index as _shape_index,
  curvedness as _curvedness,
  sky_view_factor as _sky_view_factor,
  uncertainty_slope as _uncertainty_slope,
  ssa_2d_denoise as _ssa_2d_denoise,
  fill_depressions as _fill_depressions,
  priority_flood_fill as _priority_flood_fill,
  flow_direction_d8 as _flow_direction_d8,
  flow_accumulation_d8 as _flow_accumulation_d8,
  hand_compute as _hand_compute,
  ndvi as _ndvi,
  ndwi as _ndwi,
  savi as _savi,
  normalized_diff as _normalized_diff,
  morph_erode as _morph_erode,
  morph_dilate as _morph_dilate,
  morph_opening as _morph_opening,
  morph_closing as _morph_closing,
  focal_mean as _focal_mean,
  focal_std as _focal_std,
  focal_range as _focal_range,
} from "./surtgis_wasm.js";

export class SurtGIS {
  #ready = false;

  /** Initialize the WASM module. WASM auto-initialises on import (bundler target). */
  static async init() {
    const instance = new SurtGIS();
    // WASM is auto-initialised at import time for bundler target.
    instance.#ready = true;
    return instance;
  }

  #check() {
    if (!this.#ready) throw new Error("SurtGIS not initialised. Call SurtGIS.init() first.");
  }

  // ── Terrain ──────────────────────────────────────────────────────────

  /** Compute slope. Options: { units: "degrees"|"percent" } */
  slope(dem, opts = {}) {
    this.#check();
    return _slope(dem, opts.units ?? "degrees");
  }

  /** Compute aspect in degrees (0-360, 0=North). */
  aspect(dem) {
    this.#check();
    return _aspect_degrees(dem);
  }

  /** Compute hillshade. Options: { azimuth: 315, altitude: 45 } */
  hillshade(dem, opts = {}) {
    this.#check();
    return _hillshade_compute(dem, opts.azimuth ?? 315, opts.altitude ?? 45);
  }

  /** Compute multidirectional hillshade (6 azimuths combined). */
  multidirectionalHillshade(dem) {
    this.#check();
    return _multidirectional_hillshade(dem);
  }

  /** Compute curvature. Options: { type: "general"|"profile"|"plan" } */
  curvature(dem, opts = {}) {
    this.#check();
    return _curvature_compute(dem, opts.type ?? "general");
  }

  /** Compute TPI (Topographic Position Index). Options: { radius: 3 } */
  tpi(dem, opts = {}) {
    this.#check();
    return _tpi_compute(dem, opts.radius ?? 3);
  }

  /** Compute TRI (Terrain Ruggedness Index). */
  tri(dem) {
    this.#check();
    return _tri_compute(dem);
  }

  /** Compute TWI (Topographic Wetness Index). */
  twi(dem) {
    this.#check();
    return _twi_compute(dem);
  }

  /** Compute geomorphons. Options: { flatness: 1.0, radius: 10 } */
  geomorphons(dem, opts = {}) {
    this.#check();
    return _geomorphons_compute(dem, opts.flatness ?? 1.0, opts.radius ?? 10);
  }

  /** Compute northness (cosine of aspect). */
  northness(dem) {
    this.#check();
    return _northness_compute(dem);
  }

  /** Compute eastness (sine of aspect). */
  eastness(dem) {
    this.#check();
    return _eastness_compute(dem);
  }

  /** Compute DEV (Deviation from Mean Elevation). Options: { radius: 10 } */
  dev(dem, opts = {}) {
    this.#check();
    return _dev_compute(dem, opts.radius ?? 10);
  }

  /** Compute shape index. */
  shapeIndex(dem) {
    this.#check();
    return _shape_index(dem);
  }

  /** Compute curvedness. */
  curvedness(dem) {
    this.#check();
    return _curvedness(dem);
  }

  /** Compute sky view factor. Options: { directions: 16, radius: 10 } */
  skyViewFactor(dem, opts = {}) {
    this.#check();
    return _sky_view_factor(dem, opts.directions ?? 16, opts.radius ?? 10);
  }

  /** Compute slope uncertainty (RMSE). Options: { demRmse: 1.0 } */
  uncertaintySlope(dem, opts = {}) {
    this.#check();
    return _uncertainty_slope(dem, opts.demRmse ?? 1.0);
  }

  /** Denoise DEM using 2D-SSA. Options: { window: 10, components: 3 } */
  ssa2dDenoise(dem, opts = {}) {
    this.#check();
    return _ssa_2d_denoise(dem, opts.window ?? 10, opts.components ?? 3);
  }

  // ── Hydrology ────────────────────────────────────────────────────────

  /** Fill sinks (depressions) in a DEM. */
  fillSinks(dem) {
    this.#check();
    return _fill_depressions(dem);
  }

  /** Priority-flood depression filling (Barnes 2014). */
  priorityFlood(dem) {
    this.#check();
    return _priority_flood_fill(dem);
  }

  /** Compute D8 flow direction from a filled DEM. */
  flowDirectionD8(dem) {
    this.#check();
    return _flow_direction_d8(dem);
  }

  /** Compute flow accumulation from a D8 flow direction raster. */
  flowAccumulationD8(fdir) {
    this.#check();
    return _flow_accumulation_d8(fdir);
  }

  /** Compute HAND (Height Above Nearest Drainage). Options: { streamThreshold: 1000 } */
  hand(dem, opts = {}) {
    this.#check();
    return _hand_compute(dem, opts.streamThreshold ?? 1000);
  }

  // ── Imagery ──────────────────────────────────────────────────────────

  /** Compute NDVI from NIR and Red bands. */
  ndvi(nir, red) {
    this.#check();
    return _ndvi(nir, red);
  }

  /** Compute NDWI from Green and NIR bands. */
  ndwi(green, nir) {
    this.#check();
    return _ndwi(green, nir);
  }

  /** Compute SAVI. Options: { lFactor: 0.5 } */
  savi(nir, red, opts = {}) {
    this.#check();
    return _savi(nir, red, opts.lFactor ?? 0.5);
  }

  /** Generic normalized difference: (A - B) / (A + B). */
  normalizedDifference(a, b) {
    this.#check();
    return _normalized_diff(a, b);
  }

  // ── Morphology ───────────────────────────────────────────────────────

  /** Morphological erosion. Options: { radius: 1 } */
  erode(dem, opts = {}) {
    this.#check();
    return _morph_erode(dem, opts.radius ?? 1);
  }

  /** Morphological dilation. Options: { radius: 1 } */
  dilate(dem, opts = {}) {
    this.#check();
    return _morph_dilate(dem, opts.radius ?? 1);
  }

  /** Morphological opening (erosion + dilation). Options: { radius: 1 } */
  opening(dem, opts = {}) {
    this.#check();
    return _morph_opening(dem, opts.radius ?? 1);
  }

  /** Morphological closing (dilation + erosion). Options: { radius: 1 } */
  closing(dem, opts = {}) {
    this.#check();
    return _morph_closing(dem, opts.radius ?? 1);
  }

  // ── Statistics ───────────────────────────────────────────────────────

  /** Focal mean. Options: { radius: 3 } */
  focalMean(dem, opts = {}) {
    this.#check();
    return _focal_mean(dem, opts.radius ?? 3);
  }

  /** Focal standard deviation. Options: { radius: 3 } */
  focalStd(dem, opts = {}) {
    this.#check();
    return _focal_std(dem, opts.radius ?? 3);
  }

  /** Focal range (max - min). Options: { radius: 3 } */
  focalRange(dem, opts = {}) {
    this.#check();
    return _focal_range(dem, opts.radius ?? 3);
  }
}

export default SurtGIS;
