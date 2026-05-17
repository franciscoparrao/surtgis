import {
  // Terrain
  slope,
  aspect_degrees,
  hillshade_compute,
  multidirectional_hillshade,
  curvature_compute,
  advanced_curvature,
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
  openness_positive,
  openness_negative,
  mrvbf,
  uncertainty_slope,
  ssa_2d_denoise,
  // Hydrology
  fill_depressions,
  priority_flood_fill,
  flow_direction_d8,
  flow_accumulation_d8,
  flow_accumulation_mfd_compute,
  flow_direction_dinf_compute,
  flow_accumulation_dinf_compute,
  hand_compute,
  // Imagery
  ndvi,
  ndwi,
  savi,
  mndwi,
  nbr,
  ndre,
  gndvi,
  ndbi,
  ndmi,
  msavi,
  evi2,
  evi,
  bsi,
  normalized_diff,
  // Morphology
  morph_erode,
  morph_dilate,
  morph_opening,
  morph_closing,
  // Statistics
  focal_mean,
  focal_std,
  focal_range,
  focal_min,
  focal_max,
  focal_sum,
  focal_median,
  focal_majority,
  focal_percentile,
} from "surtgis-wasm";

/** WASM module auto-initialises on import (bundler target). This is a no-op kept for API compat. */
export async function initWasm() {}

/**
 * Run a WASM algorithm on a primary raster + optional extra band buffers.
 *
 * @param {string} name
 * @param {Uint8Array} demBytes  Primary raster (DEM, or first band for spectral indices)
 * @param {object} params        Algorithm-specific parameters
 * @param {Uint8Array[]} [extraBands]  Additional band buffers for multi-band indices
 *   (e.g. NDVI: [red], EVI: [red, blue], BSI: [red, nir, blue])
 * @returns {Promise<Uint8Array>} Result GeoTIFF bytes
 */
export async function runAlgorithm(name, demBytes, params = {}, extraBands = []) {
  await initWasm();
  const b1 = extraBands[0];
  const b2 = extraBands[1];
  const b3 = extraBands[2];

  switch (name) {
    // ── Terrain ──
    case "slope":
      return slope(demBytes, params.units ?? "degrees");
    case "aspect":
      return aspect_degrees(demBytes);
    case "hillshade":
      return hillshade_compute(demBytes, params.azimuth ?? 315, params.altitude ?? 45);
    case "multidirectional_hillshade":
      return multidirectional_hillshade(demBytes);
    case "curvature":
      return curvature_compute(demBytes, params.type ?? "general");
    case "advanced_curvature":
      return advanced_curvature(demBytes, params.ctype ?? "mean_h");
    case "tpi":
      return tpi_compute(demBytes, params.radius ?? 3);
    case "tri":
      return tri_compute(demBytes);
    case "twi":
      return twi_compute(demBytes);
    case "geomorphons":
      return geomorphons_compute(demBytes, params.flatness ?? 1.0, params.radius ?? 10);
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
      return sky_view_factor(demBytes, params.directions ?? 16, params.radius ?? 10);
    case "openness_positive":
      return openness_positive(demBytes, params.radius ?? 10, params.directions ?? 8);
    case "openness_negative":
      return openness_negative(demBytes, params.radius ?? 10, params.directions ?? 8);
    case "mrvbf":
      return mrvbf(demBytes);
    case "uncertainty_slope":
      return uncertainty_slope(demBytes, params.demRmse ?? 1.0);
    case "ssa_2d_denoise":
      return ssa_2d_denoise(demBytes, params.window ?? 10, params.components ?? 3);

    // ── Hydrology ──
    case "fill_sinks":
      return fill_depressions(demBytes);
    case "priority_flood":
      return priority_flood_fill(demBytes);
    case "flow_direction_d8":
      return flow_direction_d8(demBytes);
    case "flow_accumulation_d8":
      return flow_accumulation_d8(demBytes);
    case "flow_accumulation_mfd":
      return flow_accumulation_mfd_compute(demBytes);
    case "flow_direction_dinf":
      return flow_direction_dinf_compute(demBytes);
    case "flow_accumulation_dinf":
      return flow_accumulation_dinf_compute(demBytes);
    case "hand":
      return hand_compute(demBytes, params.streamThreshold ?? 1000);

    // ── Imagery (2-band) ──
    case "ndvi":
      if (!b1) throw new Error("NDVI requires NIR (primary) and Red (band 2)");
      return ndvi(demBytes, b1);
    case "ndwi":
      if (!b1) throw new Error("NDWI requires Green (primary) and NIR (band 2)");
      return ndwi(demBytes, b1);
    case "savi":
      if (!b1) throw new Error("SAVI requires NIR (primary) and Red (band 2)");
      return savi(demBytes, b1, params.lFactor ?? 0.5);
    case "mndwi":
      if (!b1) throw new Error("MNDWI requires Green (primary) and SWIR (band 2)");
      return mndwi(demBytes, b1);
    case "nbr":
      if (!b1) throw new Error("NBR requires NIR (primary) and SWIR (band 2)");
      return nbr(demBytes, b1);
    case "ndre":
      if (!b1) throw new Error("NDRE requires NIR (primary) and Red Edge (band 2)");
      return ndre(demBytes, b1);
    case "gndvi":
      if (!b1) throw new Error("GNDVI requires NIR (primary) and Green (band 2)");
      return gndvi(demBytes, b1);
    case "ndbi":
      if (!b1) throw new Error("NDBI requires SWIR (primary) and NIR (band 2)");
      return ndbi(demBytes, b1);
    case "ndmi":
      if (!b1) throw new Error("NDMI requires NIR (primary) and SWIR (band 2)");
      return ndmi(demBytes, b1);
    case "msavi":
      if (!b1) throw new Error("MSAVI requires NIR (primary) and Red (band 2)");
      return msavi(demBytes, b1);
    case "evi2":
      if (!b1) throw new Error("EVI2 requires NIR (primary) and Red (band 2)");
      return evi2(demBytes, b1);
    case "normalized_diff":
      if (!b1) throw new Error("Normalized Difference requires bands A (primary) and B (band 2)");
      return normalized_diff(demBytes, b1);

    // ── Imagery (3- and 4-band) ──
    case "evi":
      if (!b1 || !b2) throw new Error("EVI requires NIR (primary), Red (band 2), Blue (band 3)");
      return evi(demBytes, b1, b2);
    case "bsi":
      if (!b1 || !b2 || !b3)
        throw new Error("BSI requires SWIR (primary), Red (band 2), NIR (band 3), Blue (band 4)");
      return bsi(demBytes, b1, b2, b3);

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
    case "focal_min":
      return focal_min(demBytes, params.radius ?? 3);
    case "focal_max":
      return focal_max(demBytes, params.radius ?? 3);
    case "focal_sum":
      return focal_sum(demBytes, params.radius ?? 3);
    case "focal_median":
      return focal_median(demBytes, params.radius ?? 3);
    case "focal_majority":
      return focal_majority(demBytes, params.radius ?? 3);
    case "focal_percentile":
      return focal_percentile(demBytes, params.radius ?? 3, params.percentile ?? 50);

    default:
      throw new Error(`Unknown algorithm: ${name}`);
  }
}
