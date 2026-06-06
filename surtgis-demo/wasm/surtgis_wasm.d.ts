/* tslint:disable */
/* eslint-disable */

/**
 * Compute one of Florinsky's 14 curvatures from a DEM. `ctype` accepts:
 * `mean_h`, `gaussian_k`, `unsphericity_m`, `difference_e`, `kmin`, `kmax`,
 * `kh` (horizontal), `kv` (vertical), `khe` (horizontal excess), `kve`
 * (vertical excess), `ka` (accumulation), `kr` (ring), `rotor`, `laplacian`.
 *
 * SurtGIS is the only library that exposes the complete 14-curvature
 * system from a single binary, and the only one that does it in the
 * browser via WebAssembly. See paper §2.3.2 for definitions and
 * references.
 */
export function advanced_curvature(tiff_bytes: Uint8Array, ctype: string): Uint8Array;

/**
 * Compute aspect in degrees (0-360, 0=North, clockwise).
 */
export function aspect_degrees(tiff_bytes: Uint8Array): Uint8Array;

/**
 * BSI (Bare Soil Index) from SWIR, Red, NIR, and Blue bands.
 */
export function bsi(swir_bytes: Uint8Array, red_bytes: Uint8Array, nir_bytes: Uint8Array, blue_bytes: Uint8Array): Uint8Array;

/**
 * Compute curvature. `ctype`: "general", "profile", "plan".
 */
export function curvature_compute(tiff_bytes: Uint8Array, ctype: string): Uint8Array;

/**
 * Compute curvedness.
 */
export function curvedness(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute DEV (Deviation from Mean Elevation). `radius`: window radius.
 */
export function dev_compute(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Compute eastness (sin of aspect).
 */
export function eastness_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * EVI (Enhanced Vegetation Index) from NIR, Red, and Blue bands.
 * Uses default EVI coefficients (G=2.5, L=1, C1=6, C2=7.5).
 */
export function evi(nir_bytes: Uint8Array, red_bytes: Uint8Array, blue_bytes: Uint8Array): Uint8Array;

/**
 * EVI2 (Enhanced Vegetation Index, 2-band variant) from NIR and Red bands.
 */
export function evi2(nir_bytes: Uint8Array, red_bytes: Uint8Array): Uint8Array;

/**
 * Fill sinks (depressions) in a DEM.
 */
export function fill_depressions(tiff_bytes: Uint8Array): Uint8Array;

export function flow_accumulation_d8(fdir_bytes: Uint8Array): Uint8Array;

/**
 * Compute D-infinity flow accumulation (contributing area in cell counts).
 * Internally computes the directions and the accumulation in one pass; if
 * you need the angles separately, use `flow_direction_dinf_compute` first.
 */
export function flow_accumulation_dinf_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute flow accumulation from a D8 flow direction raster.
 * Compute MFD (Multiple Flow Direction) accumulation directly from a DEM.
 * Distributes flow proportionally to all downslope neighbors based on slope,
 * better representing sheet flow on gentle terrain than D8.
 */
export function flow_accumulation_mfd_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute D8 flow direction from a filled DEM. Returns encoded direction raster.
 */
export function flow_direction_d8(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute D-infinity flow direction angles (radians, 0 = East, CCW).
 * NaN for nodata/pit cells. Reference implementation matches TauDEM.
 */
export function flow_direction_dinf_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Focal majority (mode). Useful for categorical rasters.
 */
export function focal_majority(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal maximum.
 */
export function focal_max(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal mean statistic. `radius`: window radius in cells.
 */
export function focal_mean(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal median.
 */
export function focal_median(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal minimum.
 */
export function focal_min(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal percentile. `percentile` in [0, 100]; e.g. 50 = median, 25 = Q1.
 */
export function focal_percentile(tiff_bytes: Uint8Array, radius: number, percentile: number): Uint8Array;

/**
 * Focal range (max - min). `radius`: window radius in cells.
 */
export function focal_range(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal standard deviation. `radius`: window radius in cells.
 */
export function focal_std(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Focal sum.
 */
export function focal_sum(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Compute geomorphons landform classification.
 * `flatness`: flatness threshold degrees. `radius`: search radius in cells.
 */
export function geomorphons_compute(tiff_bytes: Uint8Array, flatness: number, radius: number): Uint8Array;

/**
 * GNDVI (Green NDVI) from NIR and Green bands.
 */
export function gndvi(nir_bytes: Uint8Array, green_bytes: Uint8Array): Uint8Array;

/**
 * Compute HAND (Height Above Nearest Drainage) from a DEM.
 * Internally computes fill → flow_direction → flow_accumulation → HAND.
 * `stream_threshold`: accumulation threshold to define streams (default 1000).
 */
export function hand_compute(tiff_bytes: Uint8Array, stream_threshold: number): Uint8Array;

/**
 * Compute hillshade. `azimuth`: sun azimuth (default 315). `altitude`: sun altitude (default 45).
 */
export function hillshade_compute(tiff_bytes: Uint8Array, azimuth: number, altitude: number): Uint8Array;

/**
 * MNDWI (Modified NDWI) from Green and SWIR bands.
 */
export function mndwi(green_bytes: Uint8Array, swir_bytes: Uint8Array): Uint8Array;

/**
 * Morphological closing (dilation followed by erosion).
 */
export function morph_closing(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Morphological dilation with square kernel of given `radius`.
 */
export function morph_dilate(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Morphological erosion with square kernel of given `radius`.
 */
export function morph_erode(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Morphological opening (erosion followed by dilation).
 */
export function morph_opening(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * MRVBF — Multi-Resolution Valley Bottom Flatness (Gallant & Dowling 2003).
 * Returns only the MRVBF raster (companion MRRTF dropped); if you need both,
 * expose via the native Rust API.
 */
export function mrvbf(tiff_bytes: Uint8Array): Uint8Array;

/**
 * MSAVI (Modified SAVI) from NIR and Red bands. Auto-adjusts L factor.
 */
export function msavi(nir_bytes: Uint8Array, red_bytes: Uint8Array): Uint8Array;

/**
 * Compute multidirectional hillshade (6 azimuths combined).
 */
export function multidirectional_hillshade(tiff_bytes: Uint8Array): Uint8Array;

/**
 * NBR (Normalized Burn Ratio) from NIR and SWIR bands.
 */
export function nbr(nir_bytes: Uint8Array, swir_bytes: Uint8Array): Uint8Array;

/**
 * NDBI (Normalized Difference Built-up Index) from SWIR and NIR bands.
 */
export function ndbi(swir_bytes: Uint8Array, nir_bytes: Uint8Array): Uint8Array;

/**
 * NDMI (Normalized Difference Moisture Index) from NIR and SWIR bands.
 */
export function ndmi(nir_bytes: Uint8Array, swir_bytes: Uint8Array): Uint8Array;

/**
 * NDRE (Normalized Difference Red Edge) from NIR and Red Edge bands.
 */
export function ndre(nir_bytes: Uint8Array, red_edge_bytes: Uint8Array): Uint8Array;

/**
 * Compute NDVI from NIR and Red bands.
 */
export function ndvi(nir_bytes: Uint8Array, red_bytes: Uint8Array): Uint8Array;

/**
 * Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
 */
export function ndwi(green_bytes: Uint8Array, nir_bytes: Uint8Array): Uint8Array;

/**
 * Compute generic Normalized Difference Index: (A - B) / (A + B).
 */
export function normalized_diff(a_bytes: Uint8Array, b_bytes: Uint8Array): Uint8Array;

/**
 * Compute northness (cos of aspect).
 */
export function northness_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Negative openness — visibility of the ground below each cell. High values
 * = concave hollows / valleys.
 */
export function openness_negative(tiff_bytes: Uint8Array, radius: number, n_dirs: number): Uint8Array;

/**
 * Positive openness — visibility of the sky above each cell across `n_dirs`
 * directions out to `radius` cells. High values = exposed ridges/peaks.
 */
export function openness_positive(tiff_bytes: Uint8Array, radius: number, n_dirs: number): Uint8Array;

/**
 * Priority-flood depression filling (Barnes 2014).
 */
export function priority_flood_fill(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Full shaded-relief composite. Returns a raw RGBA `Vec<u8>` of length
 * `width * height * 4` in row-major order; the caller knows
 * `width × height` from the DEM and can either encode a PNG on the JS
 * side or upload it directly as a canvas/WebGL texture.
 *
 * `colormap`: scheme name (`"terrain"`, `"grayscale"`, `"divergent"`,
 * `"ndvi"`, `"bwr"`, `"geomorphons"`, `"water"`, `"accumulation"`).
 * Unknown names fall back to `"terrain"`.
 *
 * `shadows`: if true, runs `ray_shade` with 11 sun samples sharing the
 * azimuth (rayshader anglebreaks recipe — hits the amortised path).
 *
 * `ambient`: if true, runs `ambient_shade` (SVF) with radius=30.
 */
export function relief_compute(tiff_bytes: Uint8Array, colormap: string, sun_azimuth: number, sun_altitude: number, shadows: boolean, ambient: boolean): Uint8Array;

/**
 * Compute SAVI (Soil-Adjusted Vegetation Index). `l_factor`: soil adjustment (default 0.5).
 */
export function savi(nir_bytes: Uint8Array, red_bytes: Uint8Array, l_factor: number): Uint8Array;

/**
 * Compute shape index.
 */
export function shape_index(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute sky view factor. `n_dirs`: number of directions. `max_radius`: search radius in cells.
 */
export function sky_view_factor(tiff_bytes: Uint8Array, n_dirs: number, max_radius: number): Uint8Array;

/**
 * Compute slope from a DEM. `units`: "degrees" (default) or "percent".
 */
export function slope(tiff_bytes: Uint8Array, units: string): Uint8Array;

/**
 * Denoise DEM using 2D-SSA. `window`: window size, `components`: number of signal components.
 */
export function ssa_2d_denoise(tiff_bytes: Uint8Array, window: number, components: number): Uint8Array;

/**
 * Compute TPI (Topographic Position Index). `radius`: window radius in cells.
 */
export function tpi_compute(tiff_bytes: Uint8Array, radius: number): Uint8Array;

/**
 * Compute TRI (Terrain Ruggedness Index).
 */
export function tri_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute TWI (Topographic Wetness Index) from a DEM.
 * Internally computes fill → flow_direction → flow_accumulation → slope → TWI.
 */
export function twi_compute(tiff_bytes: Uint8Array): Uint8Array;

/**
 * Compute slope uncertainty (RMSE). `dem_rmse`: DEM vertical RMSE in meters.
 */
export function uncertainty_slope(tiff_bytes: Uint8Array, dem_rmse: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly advanced_curvature: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly aspect_degrees: (a: number, b: number) => [number, number, number, number];
    readonly bsi: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly curvature_compute: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly curvedness: (a: number, b: number) => [number, number, number, number];
    readonly dev_compute: (a: number, b: number, c: number) => [number, number, number, number];
    readonly eastness_compute: (a: number, b: number) => [number, number, number, number];
    readonly evi: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly evi2: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly fill_depressions: (a: number, b: number) => [number, number, number, number];
    readonly flow_accumulation_d8: (a: number, b: number) => [number, number, number, number];
    readonly flow_accumulation_dinf_compute: (a: number, b: number) => [number, number, number, number];
    readonly flow_accumulation_mfd_compute: (a: number, b: number) => [number, number, number, number];
    readonly flow_direction_d8: (a: number, b: number) => [number, number, number, number];
    readonly flow_direction_dinf_compute: (a: number, b: number) => [number, number, number, number];
    readonly focal_majority: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_max: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_mean: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_median: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_min: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_percentile: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly focal_range: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_std: (a: number, b: number, c: number) => [number, number, number, number];
    readonly focal_sum: (a: number, b: number, c: number) => [number, number, number, number];
    readonly geomorphons_compute: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly gndvi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly hand_compute: (a: number, b: number, c: number) => [number, number, number, number];
    readonly hillshade_compute: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly mndwi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly morph_closing: (a: number, b: number, c: number) => [number, number, number, number];
    readonly morph_dilate: (a: number, b: number, c: number) => [number, number, number, number];
    readonly morph_erode: (a: number, b: number, c: number) => [number, number, number, number];
    readonly morph_opening: (a: number, b: number, c: number) => [number, number, number, number];
    readonly mrvbf: (a: number, b: number) => [number, number, number, number];
    readonly msavi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly multidirectional_hillshade: (a: number, b: number) => [number, number, number, number];
    readonly nbr: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly ndbi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly ndmi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly ndre: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly ndvi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly ndwi: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly normalized_diff: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly northness_compute: (a: number, b: number) => [number, number, number, number];
    readonly openness_negative: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly openness_positive: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly priority_flood_fill: (a: number, b: number) => [number, number, number, number];
    readonly relief_compute: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly savi: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly shape_index: (a: number, b: number) => [number, number, number, number];
    readonly sky_view_factor: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly slope: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly ssa_2d_denoise: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly tpi_compute: (a: number, b: number, c: number) => [number, number, number, number];
    readonly tri_compute: (a: number, b: number) => [number, number, number, number];
    readonly twi_compute: (a: number, b: number) => [number, number, number, number];
    readonly uncertainty_slope: (a: number, b: number, c: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
