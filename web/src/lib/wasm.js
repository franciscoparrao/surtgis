import {
  slope,
  aspect_degrees,
  hillshade_compute,
  multidirectional_hillshade,
  curvature_compute,
  tpi_compute,
  tri_compute,
  twi_compute,
  geomorphons_compute,
  northness_compute,
  eastness_compute,
  dev_compute,
  shape_index,
  curvedness,
  sky_view_factor,
  uncertainty_slope,
  ssa_2d_denoise,
  fill_depressions,
  priority_flood_fill,
  flow_direction_d8,
  flow_accumulation_d8,
  hand_compute,
  ndvi,
  ndwi,
  savi,
  normalized_diff,
  morph_erode,
  morph_dilate,
  morph_opening,
  morph_closing,
  focal_mean,
  focal_std,
  focal_range,
} from "surtgis-wasm";

/** WASM module auto-initialises on import (bundler target). This is a no-op kept for API compat. */
export async function initWasm() {}

/**
 * @param {string} name
 * @param {Uint8Array} demBytes
 * @param {object} params
 * @param {Uint8Array} [secondBand] - second band for dual-band algorithms
 * @returns {Promise<Uint8Array>}
 */
export async function runAlgorithm(name, demBytes, params = {}, secondBand) {
  await initWasm();

  switch (name) {
    // ── Terrain ──
    case "slope":
      return slope(demBytes, params.units ?? "degrees");
    case "aspect":
      return aspect_degrees(demBytes);
    case "hillshade":
      return hillshade_compute(
        demBytes,
        params.azimuth ?? 315,
        params.altitude ?? 45,
      );
    case "multidirectional_hillshade":
      return multidirectional_hillshade(demBytes);
    case "curvature":
      return curvature_compute(demBytes, params.type ?? "general");
    case "tpi":
      return tpi_compute(demBytes, params.radius ?? 3);
    case "tri":
      return tri_compute(demBytes);
    case "twi":
      return twi_compute(demBytes);
    case "geomorphons":
      return geomorphons_compute(
        demBytes,
        params.flatness ?? 1.0,
        params.radius ?? 10,
      );
    case "northness":
      return northness_compute(demBytes);
    case "eastness":
      return eastness_compute(demBytes);
    case "dev":
      return dev_compute(demBytes, params.radius ?? 10);
    case "shape_index":
      return shape_index(demBytes);
    case "curvedness":
      return curvedness(demBytes);
    case "sky_view_factor":
      return sky_view_factor(
        demBytes,
        params.directions ?? 16,
        params.radius ?? 10,
      );
    case "uncertainty_slope":
      return uncertainty_slope(demBytes, params.demRmse ?? 1.0);
    case "ssa_2d_denoise":
      return ssa_2d_denoise(
        demBytes,
        params.window ?? 10,
        params.components ?? 3,
      );

    // ── Hydrology ──
    case "fill_sinks":
      return fill_depressions(demBytes);
    case "priority_flood":
      return priority_flood_fill(demBytes);
    case "flow_direction_d8":
      return flow_direction_d8(demBytes);
    case "flow_accumulation_d8":
      return flow_accumulation_d8(demBytes);
    case "hand":
      return hand_compute(demBytes, params.streamThreshold ?? 1000);

    // ── Imagery ──
    case "ndvi":
      if (!secondBand) throw new Error("NDVI requires NIR and Red bands");
      return ndvi(demBytes, secondBand);
    case "ndwi":
      if (!secondBand) throw new Error("NDWI requires Green and NIR bands");
      return ndwi(demBytes, secondBand);
    case "savi":
      if (!secondBand) throw new Error("SAVI requires NIR and Red bands");
      return savi(demBytes, secondBand, params.lFactor ?? 0.5);
    case "normalized_diff":
      if (!secondBand) throw new Error("Normalized Difference requires two bands");
      return normalized_diff(demBytes, secondBand);

    // ── Morphology ──
    case "morph_erode":
      return morph_erode(demBytes, params.radius ?? 1);
    case "morph_dilate":
      return morph_dilate(demBytes, params.radius ?? 1);
    case "morph_opening":
      return morph_opening(demBytes, params.radius ?? 1);
    case "morph_closing":
      return morph_closing(demBytes, params.radius ?? 1);

    // ── Statistics ──
    case "focal_mean":
      return focal_mean(demBytes, params.radius ?? 3);
    case "focal_std":
      return focal_std(demBytes, params.radius ?? 3);
    case "focal_range":
      return focal_range(demBytes, params.radius ?? 3);

    default:
      throw new Error(`Unknown algorithm: ${name}`);
  }
}
