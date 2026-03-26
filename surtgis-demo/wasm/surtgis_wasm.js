/* @ts-self-types="./surtgis_wasm.d.ts" */

/**
 * Compute aspect in degrees (0-360, 0=North, clockwise).
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function aspect_degrees(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.aspect_degrees(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute curvature. `ctype`: "general", "profile", "plan".
 * @param {Uint8Array} tiff_bytes
 * @param {string} ctype
 * @returns {Uint8Array}
 */
export function curvature_compute(tiff_bytes, ctype) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ctype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.curvature_compute(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Compute curvedness.
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function curvedness(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.curvedness(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute DEV (Deviation from Mean Elevation). `radius`: window radius.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function dev_compute(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dev_compute(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute eastness (sin of aspect).
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function eastness_compute(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.eastness_compute(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Fill sinks (depressions) in a DEM.
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function fill_depressions(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fill_depressions(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute flow accumulation from a D8 flow direction raster.
 * @param {Uint8Array} fdir_bytes
 * @returns {Uint8Array}
 */
export function flow_accumulation_d8(fdir_bytes) {
    const ptr0 = passArray8ToWasm0(fdir_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.flow_accumulation_d8(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute D8 flow direction from a filled DEM. Returns encoded direction raster.
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function flow_direction_d8(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.flow_direction_d8(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Focal mean statistic. `radius`: window radius in cells.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function focal_mean(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.focal_mean(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Focal range (max - min). `radius`: window radius in cells.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function focal_range(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.focal_range(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Focal standard deviation. `radius`: window radius in cells.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function focal_std(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.focal_std(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute geomorphons landform classification.
 * `flatness`: flatness threshold degrees. `radius`: search radius in cells.
 * @param {Uint8Array} tiff_bytes
 * @param {number} flatness
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function geomorphons_compute(tiff_bytes, flatness, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.geomorphons_compute(ptr0, len0, flatness, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute HAND (Height Above Nearest Drainage) from a DEM.
 * Internally computes fill → flow_direction → flow_accumulation → HAND.
 * `stream_threshold`: accumulation threshold to define streams (default 1000).
 * @param {Uint8Array} tiff_bytes
 * @param {number} stream_threshold
 * @returns {Uint8Array}
 */
export function hand_compute(tiff_bytes, stream_threshold) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hand_compute(ptr0, len0, stream_threshold);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute hillshade. `azimuth`: sun azimuth (default 315). `altitude`: sun altitude (default 45).
 * @param {Uint8Array} tiff_bytes
 * @param {number} azimuth
 * @param {number} altitude
 * @returns {Uint8Array}
 */
export function hillshade_compute(tiff_bytes, azimuth, altitude) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hillshade_compute(ptr0, len0, azimuth, altitude);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Morphological closing (dilation followed by erosion).
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function morph_closing(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.morph_closing(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Morphological dilation with square kernel of given `radius`.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function morph_dilate(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.morph_dilate(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Morphological erosion with square kernel of given `radius`.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function morph_erode(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.morph_erode(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Morphological opening (erosion followed by dilation).
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function morph_opening(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.morph_opening(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute multidirectional hillshade (6 azimuths combined).
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function multidirectional_hillshade(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.multidirectional_hillshade(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute NDVI from NIR and Red bands.
 * @param {Uint8Array} nir_bytes
 * @param {Uint8Array} red_bytes
 * @returns {Uint8Array}
 */
export function ndvi(nir_bytes, red_bytes) {
    const ptr0 = passArray8ToWasm0(nir_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(red_bytes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ndvi(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
 * @param {Uint8Array} green_bytes
 * @param {Uint8Array} nir_bytes
 * @returns {Uint8Array}
 */
export function ndwi(green_bytes, nir_bytes) {
    const ptr0 = passArray8ToWasm0(green_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(nir_bytes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ndwi(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Compute generic Normalized Difference Index: (A - B) / (A + B).
 * @param {Uint8Array} a_bytes
 * @param {Uint8Array} b_bytes
 * @returns {Uint8Array}
 */
export function normalized_diff(a_bytes, b_bytes) {
    const ptr0 = passArray8ToWasm0(a_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(b_bytes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.normalized_diff(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Compute northness (cos of aspect).
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function northness_compute(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.northness_compute(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Priority-flood depression filling (Barnes 2014).
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function priority_flood_fill(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.priority_flood_fill(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute SAVI (Soil-Adjusted Vegetation Index). `l_factor`: soil adjustment (default 0.5).
 * @param {Uint8Array} nir_bytes
 * @param {Uint8Array} red_bytes
 * @param {number} l_factor
 * @returns {Uint8Array}
 */
export function savi(nir_bytes, red_bytes, l_factor) {
    const ptr0 = passArray8ToWasm0(nir_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(red_bytes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.savi(ptr0, len0, ptr1, len1, l_factor);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Compute shape index.
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function shape_index(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.shape_index(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute sky view factor. `n_dirs`: number of directions. `max_radius`: search radius in cells.
 * @param {Uint8Array} tiff_bytes
 * @param {number} n_dirs
 * @param {number} max_radius
 * @returns {Uint8Array}
 */
export function sky_view_factor(tiff_bytes, n_dirs, max_radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sky_view_factor(ptr0, len0, n_dirs, max_radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute slope from a DEM. `units`: "degrees" (default) or "percent".
 * @param {Uint8Array} tiff_bytes
 * @param {string} units
 * @returns {Uint8Array}
 */
export function slope(tiff_bytes, units) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(units, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.slope(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Denoise DEM using 2D-SSA. `window`: window size, `components`: number of signal components.
 * @param {Uint8Array} tiff_bytes
 * @param {number} window
 * @param {number} components
 * @returns {Uint8Array}
 */
export function ssa_2d_denoise(tiff_bytes, window, components) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ssa_2d_denoise(ptr0, len0, window, components);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute TPI (Topographic Position Index). `radius`: window radius in cells.
 * @param {Uint8Array} tiff_bytes
 * @param {number} radius
 * @returns {Uint8Array}
 */
export function tpi_compute(tiff_bytes, radius) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tpi_compute(ptr0, len0, radius);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute TRI (Terrain Ruggedness Index).
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function tri_compute(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tri_compute(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute TWI (Topographic Wetness Index) from a DEM.
 * Internally computes fill → flow_direction → flow_accumulation → slope → TWI.
 * @param {Uint8Array} tiff_bytes
 * @returns {Uint8Array}
 */
export function twi_compute(tiff_bytes) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.twi_compute(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

/**
 * Compute slope uncertainty (RMSE). `dem_rmse`: DEM vertical RMSE in meters.
 * @param {Uint8Array} tiff_bytes
 * @param {number} dem_rmse
 * @returns {Uint8Array}
 */
export function uncertainty_slope(tiff_bytes, dem_rmse) {
    const ptr0 = passArray8ToWasm0(tiff_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.uncertainty_slope(ptr0, len0, dem_rmse);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./surtgis_wasm_bg.js": import0,
    };
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('surtgis_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
