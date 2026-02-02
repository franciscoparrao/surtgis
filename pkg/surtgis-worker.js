/**
 * SurtGISWorker — main-thread API that delegates to a Web Worker.
 *
 * Usage:
 *   import { SurtGISWorker } from "surtgis/surtgis-worker.js";
 *   const gis = await SurtGISWorker.init();
 *   const result = await gis.slope(demBytes, { units: "degrees" });
 *   gis.terminate(); // when done
 */

let _nextId = 0;

export class SurtGISWorker {
  /** @type {Worker} */
  #worker;
  /** @type {Map<number, { resolve: Function, reject: Function }>} */
  #pending = new Map();

  /**
   * Create and initialise a SurtGISWorker.
   * @param {string|URL} [workerUrl] - URL to worker.js. Defaults to same-directory import.
   */
  static async init(workerUrl) {
    const instance = new SurtGISWorker();
    const url = workerUrl ?? new URL("./worker.js", import.meta.url);
    instance.#worker = new Worker(url, { type: "module" });
    instance.#worker.onmessage = (e) => instance.#onMessage(e);
    instance.#worker.onerror = (e) => {
      // Reject all pending calls
      for (const [, { reject }] of instance.#pending) {
        reject(new Error(`Worker error: ${e.message}`));
      }
      instance.#pending.clear();
    };
    // Warm up: trigger WASM init inside worker
    await instance.#call("slope", [new Uint8Array(0), "degrees"]).catch(() => {});
    return instance;
  }

  #onMessage(e) {
    const { id, result, error } = e.data;
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    if (error) {
      pending.reject(new Error(error));
    } else {
      pending.resolve(result);
    }
  }

  /**
   * @param {string} method - WASM function name
   * @param {any[]} args
   * @returns {Promise<Uint8Array>}
   */
  #call(method, args) {
    return new Promise((resolve, reject) => {
      const id = _nextId++;
      this.#pending.set(id, { resolve, reject });
      // Transfer Uint8Array buffers for zero-copy send
      const transfer = args
        .filter((a) => a instanceof Uint8Array)
        .map((a) => a.buffer);
      this.#worker.postMessage({ id, method, args }, transfer);
    });
  }

  /** Terminate the worker. */
  terminate() {
    this.#worker.terminate();
    for (const [, { reject }] of this.#pending) {
      reject(new Error("Worker terminated"));
    }
    this.#pending.clear();
  }

  // ── Terrain ──────────────────────────────────────────────────────────

  slope(dem, opts = {}) {
    return this.#call("slope", [dem, opts.units ?? "degrees"]);
  }
  aspect(dem) {
    return this.#call("aspect_degrees", [dem]);
  }
  hillshade(dem, opts = {}) {
    return this.#call("hillshade_compute", [dem, opts.azimuth ?? 315, opts.altitude ?? 45]);
  }
  multidirectionalHillshade(dem) {
    return this.#call("multidirectional_hillshade", [dem]);
  }
  curvature(dem, opts = {}) {
    return this.#call("curvature_compute", [dem, opts.type ?? "general"]);
  }
  tpi(dem, opts = {}) {
    return this.#call("tpi_compute", [dem, opts.radius ?? 3]);
  }
  tri(dem) {
    return this.#call("tri_compute", [dem]);
  }
  twi(dem) {
    return this.#call("twi_compute", [dem]);
  }
  geomorphons(dem, opts = {}) {
    return this.#call("geomorphons_compute", [dem, opts.flatness ?? 1.0, opts.radius ?? 10]);
  }
  northness(dem) {
    return this.#call("northness_compute", [dem]);
  }
  eastness(dem) {
    return this.#call("eastness_compute", [dem]);
  }
  dev(dem, opts = {}) {
    return this.#call("dev_compute", [dem, opts.radius ?? 10]);
  }
  shapeIndex(dem) {
    return this.#call("shape_index", [dem]);
  }
  curvedness(dem) {
    return this.#call("curvedness", [dem]);
  }
  skyViewFactor(dem, opts = {}) {
    return this.#call("sky_view_factor", [dem, opts.directions ?? 16, opts.radius ?? 10]);
  }
  uncertaintySlope(dem, opts = {}) {
    return this.#call("uncertainty_slope", [dem, opts.demRmse ?? 1.0]);
  }
  ssa2dDenoise(dem, opts = {}) {
    return this.#call("ssa_2d_denoise", [dem, opts.window ?? 10, opts.components ?? 3]);
  }

  // ── Hydrology ────────────────────────────────────────────────────────

  fillSinks(dem) {
    return this.#call("fill_depressions", [dem]);
  }
  priorityFlood(dem) {
    return this.#call("priority_flood_fill", [dem]);
  }
  flowDirectionD8(dem) {
    return this.#call("flow_direction_d8", [dem]);
  }
  flowAccumulationD8(fdir) {
    return this.#call("flow_accumulation_d8", [fdir]);
  }
  hand(dem, opts = {}) {
    return this.#call("hand_compute", [dem, opts.streamThreshold ?? 1000]);
  }

  // ── Imagery ──────────────────────────────────────────────────────────

  ndvi(nir, red) {
    return this.#call("ndvi", [nir, red]);
  }
  ndwi(green, nir) {
    return this.#call("ndwi", [green, nir]);
  }
  savi(nir, red, opts = {}) {
    return this.#call("savi", [nir, red, opts.lFactor ?? 0.5]);
  }
  normalizedDifference(a, b) {
    return this.#call("normalized_diff", [a, b]);
  }

  // ── Morphology ───────────────────────────────────────────────────────

  erode(dem, opts = {}) {
    return this.#call("morph_erode", [dem, opts.radius ?? 1]);
  }
  dilate(dem, opts = {}) {
    return this.#call("morph_dilate", [dem, opts.radius ?? 1]);
  }
  opening(dem, opts = {}) {
    return this.#call("morph_opening", [dem, opts.radius ?? 1]);
  }
  closing(dem, opts = {}) {
    return this.#call("morph_closing", [dem, opts.radius ?? 1]);
  }

  // ── Statistics ───────────────────────────────────────────────────────

  focalMean(dem, opts = {}) {
    return this.#call("focal_mean", [dem, opts.radius ?? 3]);
  }
  focalStd(dem, opts = {}) {
    return this.#call("focal_std", [dem, opts.radius ?? 3]);
  }
  focalRange(dem, opts = {}) {
    return this.#call("focal_range", [dem, opts.radius ?? 3]);
  }
}

export default SurtGISWorker;
