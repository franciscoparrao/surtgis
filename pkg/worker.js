/**
 * SurtGIS Web Worker — runs WASM algorithms off the main thread.
 *
 * Protocol (postMessage):
 *   Request:  { id, method, args: [...Uint8Array | primitive] }
 *   Response: { id, result: Uint8Array } | { id, error: string }
 *
 * The worker lazily initialises the WASM module on the first call.
 */

import * as wasm from "./surtgis_wasm.js";

// WASM auto-initialises on import for bundler target.

self.onmessage = async (e) => {
  const { id, method, args } = e.data;
  try {
    const fn = wasm[method];
    if (typeof fn !== "function") {
      throw new Error(`Unknown WASM method: ${method}`);
    }
    const result = fn(...args);
    // Transfer the underlying buffer for zero-copy
    self.postMessage({ id, result }, [result.buffer]);
  } catch (err) {
    self.postMessage({ id, error: err.message ?? String(err) });
  }
};
