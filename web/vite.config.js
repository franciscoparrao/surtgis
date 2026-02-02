import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  base: process.env.BASE_URL || "/",
  plugins: [svelte(), wasm(), topLevelAwait()],
  optimizeDeps: {
    exclude: ["surtgis-wasm"],
  },
  server: {
    fs: {
      allow: [".", "../pkg"],
    },
  },
});
