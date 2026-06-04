/* tslint:disable */
/* eslint-disable */

/**
 * JS-facing handle. Wraps the shared lighting state — JS sliders mutate
 * it from outside the event loop without going through wgpu's queue.
 */
export class ReliefHandle {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    set_ambient(ambient: number): void;
    set_sun(azimuth_deg: number, altitude_deg: number): void;
    set_vertical_scale(zex: number): void;
}

/**
 * Compute the rayshader-style relief composite for the given DEM bytes,
 * build the 3D mesh, attach a wgpu surface to `canvas_id`, and start the
 * render loop.
 */
export function run_relief3d_canvas(canvas_id: string, tiff_bytes: Uint8Array, colormap: string, sun_azimuth: number, sun_altitude: number, shadows: boolean, ambient: boolean, vertical_exaggeration: number): ReliefHandle;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_reliefhandle_free: (a: number, b: number) => void;
    readonly reliefhandle_set_ambient: (a: number, b: number) => void;
    readonly reliefhandle_set_sun: (a: number, b: number, c: number) => void;
    readonly reliefhandle_set_vertical_scale: (a: number, b: number) => void;
    readonly run_relief3d_canvas: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly wasm_bindgen__closure__destroy__h0210e68aae7756cd: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h18a6ad49415e60d1: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hc7896da5159965b6: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h6b2cb778c5a7ebfe: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__hbdabf1596fc12a61: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_5: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_6: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_7: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_8: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h18c7e1d742d234c1_9: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h2724453e040b6af3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h2724453e040b6af3_12: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h6f24b05858f8b6eb: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
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
