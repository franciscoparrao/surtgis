<script>
  import { initWasm, runAlgorithm } from "./lib/wasm.js";
  import { parseTiff } from "./lib/tiff.js";
  import FileUpload from "./components/FileUpload.svelte";
  import AlgoPanel from "./components/AlgoPanel.svelte";
  import RasterView from "./components/RasterView.svelte";

  let demBytes = $state(null);
  let demName = $state("");
  let demParsed = $state(null);
  /** Extra band buffers for multi-band indices. Indexed by position (band 2 = [0], 3 = [1], ...). */
  let extraBands = $state([null, null, null]);

  let resultBytes = $state(null);
  let resultParsed = $state(null);

  let selectedAlgo = $state("");
  let processing = $state(false);
  let elapsed = $state(null);
  let error = $state("");
  let wasmReady = $state(false);

  /**
   * Per-algorithm definition of the additional band slots after the primary
   * raster. Each entry is an array of label strings; UI renders one uploader
   * per slot. Primary raster label is derived from the first entry's role
   * (see PRIMARY_BAND_LABEL).
   */
  const MULTI_BAND_LABELS = {
    // 2-band
    ndvi: ["Red Band"],
    ndwi: ["NIR Band"],
    savi: ["Red Band"],
    normalized_diff: ["Band B"],
    mndwi: ["SWIR Band"],
    nbr: ["SWIR Band"],
    ndre: ["Red Edge Band"],
    gndvi: ["Green Band"],
    ndbi: ["NIR Band"],
    ndmi: ["SWIR Band"],
    msavi: ["Red Band"],
    evi2: ["Red Band"],
    // 3-band
    evi: ["Red Band", "Blue Band"],
    // 4-band
    bsi: ["Red Band", "NIR Band", "Blue Band"],
  };

  /** Label for the primary uploader when the selected algo needs multiple bands. */
  const PRIMARY_BAND_LABEL = {
    ndvi: "NIR Band",
    ndwi: "Green Band",
    savi: "NIR Band",
    normalized_diff: "Band A",
    mndwi: "Green Band",
    nbr: "NIR Band",
    ndre: "NIR Band",
    gndvi: "NIR Band",
    ndbi: "SWIR Band",
    ndmi: "NIR Band",
    msavi: "NIR Band",
    evi2: "NIR Band",
    evi: "NIR Band",
    bsi: "SWIR Band",
  };

  const SCHEME_MAP = {
    // Terrain
    slope: "terrain",
    aspect: "terrain",
    hillshade: "grayscale",
    multidirectional_hillshade: "grayscale",
    curvature: "blue_white_red",
    advanced_curvature: "blue_white_red",
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
    openness_positive: "grayscale",
    openness_negative: "grayscale",
    mrvbf: "water",
    uncertainty_slope: "terrain",
    ssa_2d_denoise: "terrain",
    // Hydrology
    fill_sinks: "terrain",
    priority_flood: "terrain",
    flow_direction_d8: "terrain",
    flow_accumulation_d8: "accumulation",
    flow_accumulation_mfd: "accumulation",
    flow_direction_dinf: "terrain",
    flow_accumulation_dinf: "accumulation",
    hand: "water",
    // Imagery
    ndvi: "ndvi",
    ndwi: "water",
    savi: "ndvi",
    normalized_diff: "divergent",
    mndwi: "water",
    nbr: "divergent",
    ndre: "ndvi",
    gndvi: "ndvi",
    ndbi: "divergent",
    ndmi: "water",
    msavi: "ndvi",
    evi2: "ndvi",
    evi: "ndvi",
    bsi: "divergent",
    // Morphology
    morph_erode: "terrain",
    morph_dilate: "terrain",
    morph_opening: "terrain",
    morph_closing: "terrain",
    // Statistics
    focal_mean: "terrain",
    focal_std: "terrain",
    focal_range: "terrain",
    focal_min: "terrain",
    focal_max: "terrain",
    focal_sum: "terrain",
    focal_median: "terrain",
    focal_majority: "terrain",
    focal_percentile: "terrain",
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

  function onExtraBandLoad(index, { bytes }) {
    extraBands[index] = bytes;
  }

  async function onRun(algo, params) {
    if (!demBytes) return;
    error = "";
    processing = true;
    elapsed = null;

    try {
      const t0 = performance.now();
      const slots = MULTI_BAND_LABELS[algo]?.length ?? 0;
      const bands = extraBands.slice(0, slots).filter((b) => b !== null);
      const out = await runAlgorithm(algo, demBytes, params, bands);
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

  let extraBandLabels = $derived(MULTI_BAND_LABELS[selectedAlgo] ?? []);
  let primaryLabel = $derived(
    PRIMARY_BAND_LABEL[selectedAlgo] ?? "DEM / Raster",
  );
</script>

<div class="layout">
  <aside class="sidebar">
    <h1 class="logo">SurtGIS <span class="badge">WASM</span></h1>

    <section>
      <h2>{primaryLabel}</h2>
      <FileUpload onload={onDemLoad} />
    </section>

    {#each extraBandLabels as label, i (label + i)}
      <section>
        <h2>{label}</h2>
        <FileUpload
          onload={(payload) => onExtraBandLoad(i, payload)}
          label={`Drop ${label} GeoTIFF`}
        />
      </section>
    {/each}

    <section>
      <h2>Algorithms <span class="algo-count">56</span></h2>
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
