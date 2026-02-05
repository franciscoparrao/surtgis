<script>
  import { renderToCanvas, renderLegend } from "../lib/colormap.js";

  /**
   * @type {{
   *   raster: { width: number, height: number, data: Float32Array } | null,
   *   scheme: string,
   *   elapsed: number | null,
   *   resultBytes: Uint8Array | null
   * }}
   */
  let { raster, scheme = "terrain", elapsed = null, resultBytes = null } = $props();

  /** @type {HTMLCanvasElement} */
  let canvasEl = $state();
  /** @type {HTMLCanvasElement} */
  let legendEl = $state();

  let rangeMin = $state(0);
  let rangeMax = $state(1);

  $effect(() => {
    if (raster && canvasEl) {
      const { min, max } = renderToCanvas(canvasEl, raster, scheme);
      rangeMin = min;
      rangeMax = max;
    }
  });

  $effect(() => {
    if (legendEl && raster) {
      renderLegend(legendEl, scheme, rangeMin, rangeMax);
    }
  });

  function download() {
    if (!resultBytes) return;
    const blob = new Blob([resultBytes], { type: "image/tiff" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "surtgis_result.tif";
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="viewer">
  {#if raster}
    <div class="canvas-wrap">
      <canvas bind:this={canvasEl}></canvas>
    </div>

    <div class="info-bar">
      <canvas bind:this={legendEl} width="280" height="30" class="legend"></canvas>

      <div class="meta">
        <span>{raster.width} &times; {raster.height} px</span>
        {#if elapsed !== null}
          <span>{elapsed.toFixed(0)} ms</span>
        {/if}
      </div>

      <button class="dl-btn" onclick={download}>
        Download GeoTIFF
      </button>
    </div>
  {:else}
    <div class="placeholder">
      <span>Load a raster and run an algorithm to see results here.</span>
    </div>
  {/if}
</div>

<style>
  .viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }

  .canvas-wrap {
    flex: 1;
    overflow: auto;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    background: #000;
    border-radius: 8px;
    min-height: 0;
  }

  canvas {
    max-width: 100%;
    image-rendering: pixelated;
  }

  .info-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 0;
    flex-wrap: wrap;
  }

  .legend {
    border-radius: 4px;
    flex-shrink: 0;
  }

  .meta {
    display: flex;
    gap: 0.75rem;
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  .dl-btn {
    margin-left: auto;
    background: var(--accent);
    color: #fff;
    padding: 0.45rem 1rem;
    border-radius: 6px;
    font-weight: 500;
    transition: background 0.15s;
  }

  .dl-btn:hover {
    background: var(--accent-hover);
  }

  .placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 0.9rem;
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    min-height: 300px;
  }
</style>
