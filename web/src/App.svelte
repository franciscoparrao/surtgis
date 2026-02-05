<script>
  import { initWasm, runAlgorithm } from "./lib/wasm.js";
  import { parseTiff } from "./lib/tiff.js";
  import FileUpload from "./components/FileUpload.svelte";
  import AlgoPanel from "./components/AlgoPanel.svelte";
  import RasterView from "./components/RasterView.svelte";

  let demBytes = $state(null);
  let demName = $state("");
  let demParsed = $state(null);
  let secondBandBytes = $state(null);

  let resultBytes = $state(null);
  let resultParsed = $state(null);

  let selectedAlgo = $state("");
  let processing = $state(false);
  let elapsed = $state(null);
  let error = $state("");
  let wasmReady = $state(false);

  /** Algorithms that require a second band */
  const DUAL_BAND = new Set(["ndvi", "ndwi", "savi", "normalized_diff"]);

  const SCHEME_MAP = {
    // Terrain
    slope: "terrain",
    aspect: "terrain",
    hillshade: "grayscale",
    multidirectional_hillshade: "grayscale",
    curvature: "blue_white_red",
    tpi: "blue_white_red",
    tri: "terrain",
    twi: "water",
    geomorphons: "geomorphons",
    northness: "blue_white_red",
    eastness: "blue_white_red",
    dev: "blue_white_red",
    shape_index: "blue_white_red",
    curvedness: "terrain",
    sky_view_factor: "grayscale",
    uncertainty_slope: "terrain",
    ssa_2d_denoise: "terrain",
    // Hydrology
    fill_sinks: "terrain",
    priority_flood: "terrain",
    flow_direction_d8: "terrain",
    flow_accumulation_d8: "accumulation",
    hand: "water",
    // Imagery
    ndvi: "ndvi",
    ndwi: "water",
    savi: "ndvi",
    normalized_diff: "divergent",
    // Morphology
    morph_erode: "terrain",
    morph_dilate: "terrain",
    morph_opening: "terrain",
    morph_closing: "terrain",
    // Statistics
    focal_mean: "terrain",
    focal_std: "terrain",
    focal_range: "terrain",
  };

  /** Label shown above the second band uploader */
  const SECOND_BAND_LABEL = {
    ndvi: "Red Band",
    ndwi: "NIR Band",
    savi: "Red Band",
    normalized_diff: "Band B",
  };

  $effect(() => {
    initWasm()
      .then(() => { wasmReady = true; })
      .catch((e) => { error = `WASM init failed: ${e.message}`; });
  });

  async function onDemLoad({ name, bytes }) {
    demBytes = bytes;
    demName = name;
    error = "";
    resultBytes = null;
    resultParsed = null;

    try {
      demParsed = await parseTiff(bytes.buffer);
    } catch (e) {
      demParsed = null;
    }
  }

  function onSecondBandLoad({ bytes }) {
    secondBandBytes = bytes;
  }

  async function onRun(algo, params) {
    if (!demBytes) return;
    error = "";
    processing = true;
    elapsed = null;

    try {
      const t0 = performance.now();
      const band2 = DUAL_BAND.has(algo) ? secondBandBytes : undefined;
      const out = await runAlgorithm(algo, demBytes, params, band2);
      elapsed = performance.now() - t0;

      resultBytes = out;
      const parsed = await parseTiff(out.buffer);
      resultParsed = parsed;
    } catch (e) {
      error = e.message ?? String(e);
      resultBytes = null;
      resultParsed = null;
    } finally {
      processing = false;
    }
  }

  let needsSecondBand = $derived(DUAL_BAND.has(selectedAlgo));
</script>

<div class="layout">
  <aside class="sidebar">
    <h1 class="logo">SurtGIS <span class="badge">WASM</span></h1>

    <section>
      <h2>{needsSecondBand ? "NIR / Green Band" : "DEM / Raster"}</h2>
      <FileUpload onload={onDemLoad} />
    </section>

    {#if needsSecondBand}
      <section>
        <h2>{SECOND_BAND_LABEL[selectedAlgo] ?? "Second Band"}</h2>
        <FileUpload
          onload={onSecondBandLoad}
          label="Drop second band GeoTIFF"
        />
      </section>
    {/if}

    <section>
      <h2>Algorithms <span class="algo-count">33</span></h2>
      <AlgoPanel
        disabled={!demBytes || !wasmReady}
        {processing}
        bind:selectedAlgo
        onrun={onRun}
      />
    </section>

    {#if error}
      <div class="error">{error}</div>
    {/if}

    {#if !wasmReady && !error}
      <p class="status">Loading WASM module&hellip;</p>
    {/if}

    <footer class="footer">
      <span>surtgis &middot; Svelte 5 + WASM demo</span>
    </footer>
  </aside>

  <main class="main">
    <div class="views">
      <!-- DEM preview (left) -->
      <div class="view-pane">
        <h3 class="pane-title">Input {demName ? `(${demName})` : ""}</h3>
        <RasterView
          raster={demParsed}
          scheme="terrain"
          elapsed={null}
          resultBytes={demBytes}
        />
      </div>

      <!-- Result (right) -->
      <div class="view-pane">
        <h3 class="pane-title">Result {selectedAlgo ? `(${selectedAlgo})` : ""}</h3>
        <RasterView
          raster={resultParsed}
          scheme={SCHEME_MAP[selectedAlgo] ?? "terrain"}
          {elapsed}
          {resultBytes}
        />
      </div>
    </div>
  </main>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: var(--sidebar-width) 1fr;
    min-height: 100vh;
  }

  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    overflow-y: auto;
  }

  .logo {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .badge {
    font-size: 0.6rem;
    background: var(--accent);
    color: #fff;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    vertical-align: middle;
    font-weight: 600;
    letter-spacing: 0.06em;
  }

  .algo-count {
    font-size: 0.65rem;
    background: var(--surface-hover);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin-left: 0.3rem;
    vertical-align: middle;
  }

  section h2 {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.45rem;
  }

  .error {
    background: rgba(255, 107, 107, 0.12);
    border: 1px solid var(--danger);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    font-size: 0.82rem;
    color: var(--danger);
    word-break: break-word;
  }

  .status {
    font-size: 0.82rem;
    color: var(--text-muted);
  }

  .main {
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .views {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    flex: 1;
    min-height: 0;
  }

  .view-pane {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .pane-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
  }

  .footer {
    margin-top: auto;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    font-size: 0.7rem;
    color: var(--text-muted);
  }
</style>
