<script>
  /**
   * @type {{
   *   disabled: boolean,
   *   processing: boolean,
   *   selectedAlgo: string,
   *   onrun: (algo: string, params: object) => void
   * }}
   */
  let { disabled, processing, selectedAlgo = $bindable(), onrun } = $props();

  // ── Parameter state ──
  let slopeUnits = $state("degrees");
  let azimuth = $state(315);
  let altitude = $state(45);
  let curvatureType = $state("general");
  let advCurvatureType = $state("mean_h");
  let tpiRadius = $state(3);
  let geoFlatness = $state(1.0);
  let geoRadius = $state(10);
  let devRadius = $state(10);
  let svfDirs = $state(16);
  let svfRadius = $state(10);
  let opennessRadius = $state(10);
  let opennessDirs = $state(8);
  let demRmse = $state(1.0);
  let ssaWindow = $state(10);
  let ssaComponents = $state(3);
  let handThreshold = $state(1000);
  let saviL = $state(0.5);
  let morphRadius = $state(1);
  let focalRadius = $state(3);
  let focalPercentile = $state(50);

  // ── Collapsed groups ──
  let collapsed = $state({});

  function toggle(title) {
    collapsed[title] = !collapsed[title];
  }

  const GROUPS = [
    {
      title: "Terrain",
      algos: [
        { id: "slope", label: "Slope" },
        { id: "aspect", label: "Aspect" },
        { id: "hillshade", label: "Hillshade" },
        { id: "multidirectional_hillshade", label: "Multi Hillshade" },
        { id: "curvature", label: "Curvature (classical)" },
        { id: "advanced_curvature", label: "Curvature (Florinsky 14)" },
        { id: "tpi", label: "TPI" },
        { id: "tri", label: "TRI" },
        { id: "twi", label: "TWI" },
        { id: "geomorphons", label: "Geomorphons" },
        { id: "northness", label: "Northness" },
        { id: "eastness", label: "Eastness" },
        { id: "dev", label: "DEV" },
        { id: "shape_index", label: "Shape Index" },
        { id: "curvedness", label: "Curvedness" },
        { id: "sky_view_factor", label: "Sky View Factor" },
        { id: "openness_positive", label: "Openness (positive)" },
        { id: "openness_negative", label: "Openness (negative)" },
        { id: "mrvbf", label: "MRVBF" },
        { id: "uncertainty_slope", label: "Slope Uncertainty" },
        { id: "ssa_2d_denoise", label: "SSA-2D Denoise" },
      ],
    },
    {
      title: "Hydrology",
      algos: [
        { id: "fill_sinks", label: "Fill Sinks" },
        { id: "priority_flood", label: "Priority Flood" },
        { id: "flow_direction_d8", label: "Flow Direction D8" },
        { id: "flow_accumulation_d8", label: "Flow Accumulation D8" },
        { id: "flow_accumulation_mfd", label: "Flow Accumulation MFD" },
        { id: "flow_direction_dinf", label: "Flow Direction D-∞" },
        { id: "flow_accumulation_dinf", label: "Flow Accumulation D-∞" },
        { id: "hand", label: "HAND" },
      ],
    },
    {
      title: "Imagery",
      algos: [
        { id: "ndvi", label: "NDVI" },
        { id: "ndwi", label: "NDWI" },
        { id: "mndwi", label: "MNDWI" },
        { id: "savi", label: "SAVI" },
        { id: "msavi", label: "MSAVI" },
        { id: "evi", label: "EVI (3-band)" },
        { id: "evi2", label: "EVI2" },
        { id: "nbr", label: "NBR" },
        { id: "ndre", label: "NDRE" },
        { id: "gndvi", label: "GNDVI" },
        { id: "ndbi", label: "NDBI" },
        { id: "ndmi", label: "NDMI" },
        { id: "bsi", label: "BSI (4-band)" },
        { id: "normalized_diff", label: "Norm. Difference" },
      ],
    },
    {
      title: "Morphology",
      algos: [
        { id: "morph_erode", label: "Erode" },
        { id: "morph_dilate", label: "Dilate" },
        { id: "morph_opening", label: "Opening" },
        { id: "morph_closing", label: "Closing" },
      ],
    },
    {
      title: "Statistics",
      algos: [
        { id: "focal_mean", label: "Focal Mean" },
        { id: "focal_std", label: "Focal Std Dev" },
        { id: "focal_range", label: "Focal Range" },
        { id: "focal_min", label: "Focal Min" },
        { id: "focal_max", label: "Focal Max" },
        { id: "focal_sum", label: "Focal Sum" },
        { id: "focal_median", label: "Focal Median" },
        { id: "focal_majority", label: "Focal Majority" },
        { id: "focal_percentile", label: "Focal Percentile" },
      ],
    },
  ];

  function run(id) {
    selectedAlgo = id;
    const params = {};
    if (id === "slope") params.units = slopeUnits;
    if (id === "hillshade") {
      params.azimuth = Number(azimuth);
      params.altitude = Number(altitude);
    }
    if (id === "curvature") params.type = curvatureType;
    if (id === "advanced_curvature") params.ctype = advCurvatureType;
    if (id === "tpi") params.radius = Number(tpiRadius);
    if (id === "geomorphons") {
      params.flatness = Number(geoFlatness);
      params.radius = Number(geoRadius);
    }
    if (id === "dev") params.radius = Number(devRadius);
    if (id === "sky_view_factor") {
      params.directions = Number(svfDirs);
      params.radius = Number(svfRadius);
    }
    if (id === "openness_positive" || id === "openness_negative") {
      params.radius = Number(opennessRadius);
      params.directions = Number(opennessDirs);
    }
    if (id === "uncertainty_slope") params.demRmse = Number(demRmse);
    if (id === "ssa_2d_denoise") {
      params.window = Number(ssaWindow);
      params.components = Number(ssaComponents);
    }
    if (id === "hand") params.streamThreshold = Number(handThreshold);
    if (id === "savi") params.lFactor = Number(saviL);
    if (id.startsWith("morph_")) params.radius = Number(morphRadius);
    if (id.startsWith("focal_")) params.radius = Number(focalRadius);
    if (id === "focal_percentile") params.percentile = Number(focalPercentile);
    onrun(id, params);
  }
</script>

<div class="panel">
  {#each GROUPS as group}
    <fieldset>
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <legend onclick={() => toggle(group.title)} class="clickable">
        <span class="arrow" class:open={!collapsed[group.title]}>&#9654;</span>
        {group.title}
        <span class="count">{group.algos.length}</span>
      </legend>

      {#if !collapsed[group.title]}
        {#each group.algos as algo}
          <div class="algo-row">
            <button
              class="algo-btn"
              class:active={selectedAlgo === algo.id}
              disabled={disabled || processing}
              onclick={() => run(algo.id)}
            >
              {#if processing && selectedAlgo === algo.id}
                <span class="spinner"></span>
              {/if}
              {algo.label}
            </button>

            <!-- Slope params -->
            {#if algo.id === "slope"}
              <select bind:value={slopeUnits} disabled={disabled || processing}>
                <option value="degrees">Degrees</option>
                <option value="percent">Percent</option>
              </select>
            {/if}

            <!-- Hillshade params -->
            {#if algo.id === "hillshade"}
              <label class="param">
                Az
                <input type="number" bind:value={azimuth} min="0" max="360" step="1" disabled={disabled || processing} />
              </label>
              <label class="param">
                Alt
                <input type="number" bind:value={altitude} min="0" max="90" step="1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- Curvature params -->
            {#if algo.id === "curvature"}
              <select bind:value={curvatureType} disabled={disabled || processing}>
                <option value="general">General</option>
                <option value="profile">Profile</option>
                <option value="plan">Plan</option>
              </select>
            {/if}

            <!-- Advanced curvature (Florinsky 14) -->
            {#if algo.id === "advanced_curvature"}
              <select bind:value={advCurvatureType} disabled={disabled || processing}>
                <option value="mean_h">Mean H</option>
                <option value="gaussian_k">Gaussian K</option>
                <option value="unsphericity_m">Unsphericity M</option>
                <option value="difference_e">Difference E</option>
                <option value="kmin">k_min</option>
                <option value="kmax">k_max</option>
                <option value="kh">Horizontal k_h</option>
                <option value="kv">Vertical k_v</option>
                <option value="khe">Horizontal excess k_he</option>
                <option value="kve">Vertical excess k_ve</option>
                <option value="ka">Accumulation K_a</option>
                <option value="kr">Ring K_r</option>
                <option value="rotor">Rotor</option>
                <option value="laplacian">Laplacian</option>
              </select>
            {/if}

            <!-- Openness params -->
            {#if algo.id === "openness_positive" || algo.id === "openness_negative"}
              <label class="param">
                R
                <input type="number" bind:value={opennessRadius} min="1" max="50" step="1" disabled={disabled || processing} />
              </label>
              <label class="param">
                Dirs
                <input type="number" bind:value={opennessDirs} min="4" max="32" step="4" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- Focal percentile param -->
            {#if algo.id === "focal_percentile"}
              <label class="param">
                Q
                <input type="number" bind:value={focalPercentile} min="0" max="100" step="5" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- TPI params -->
            {#if algo.id === "tpi"}
              <label class="param">
                R
                <input type="number" bind:value={tpiRadius} min="1" max="50" step="1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- Geomorphons params -->
            {#if algo.id === "geomorphons"}
              <label class="param">
                Flat
                <input type="number" bind:value={geoFlatness} min="0" max="10" step="0.1" disabled={disabled || processing} />
              </label>
              <label class="param">
                R
                <input type="number" bind:value={geoRadius} min="1" max="50" step="1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- DEV params -->
            {#if algo.id === "dev"}
              <label class="param">
                R
                <input type="number" bind:value={devRadius} min="1" max="50" step="1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- Sky View Factor params -->
            {#if algo.id === "sky_view_factor"}
              <label class="param">
                Dirs
                <input type="number" bind:value={svfDirs} min="4" max="64" step="4" disabled={disabled || processing} />
              </label>
              <label class="param">
                R
                <input type="number" bind:value={svfRadius} min="1" max="50" step="1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- Uncertainty params -->
            {#if algo.id === "uncertainty_slope"}
              <label class="param">
                RMSE
                <input type="number" bind:value={demRmse} min="0" max="100" step="0.1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- SSA-2D params -->
            {#if algo.id === "ssa_2d_denoise"}
              <label class="param">
                Win
                <input type="number" bind:value={ssaWindow} min="3" max="50" step="1" disabled={disabled || processing} />
              </label>
              <label class="param">
                Comp
                <input type="number" bind:value={ssaComponents} min="1" max="20" step="1" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- HAND params -->
            {#if algo.id === "hand"}
              <label class="param">
                Thr
                <input type="number" bind:value={handThreshold} min="10" max="100000" step="100" disabled={disabled || processing} />
              </label>
            {/if}

            <!-- SAVI params -->
            {#if algo.id === "savi"}
              <label class="param">
                L
                <input type="number" bind:value={saviL} min="0" max="1" step="0.1" disabled={disabled || processing} />
              </label>
            {/if}
          </div>
        {/each}

        <!-- Shared morphology radius -->
        {#if group.title === "Morphology"}
          <div class="shared-param">
            <label class="param">
              Kernel radius
              <input type="number" bind:value={morphRadius} min="1" max="20" step="1" disabled={disabled || processing} />
            </label>
          </div>
        {/if}

        <!-- Shared focal radius -->
        {#if group.title === "Statistics"}
          <div class="shared-param">
            <label class="param">
              Window radius
              <input type="number" bind:value={focalRadius} min="1" max="50" step="1" disabled={disabled || processing} />
            </label>
          </div>
        {/if}
      {/if}
    </fieldset>
  {/each}
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  fieldset {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
  }

  legend {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    padding: 0 0.4rem;
  }

  .clickable {
    cursor: pointer;
    user-select: none;
  }

  .clickable:hover {
    color: var(--accent);
  }

  .arrow {
    display: inline-block;
    font-size: 0.55rem;
    transition: transform 0.15s;
    margin-right: 0.2rem;
  }

  .arrow.open {
    transform: rotate(90deg);
  }

  .count {
    font-size: 0.6rem;
    background: var(--surface-hover);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    margin-left: 0.3rem;
  }

  .algo-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.4rem;
    flex-wrap: wrap;
  }

  .algo-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface-hover);
    color: var(--text);
    padding: 0.35rem 0.7rem;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.8rem;
    transition: background 0.15s;
  }

  .algo-btn:hover:not(:disabled) {
    background: var(--accent);
    color: #fff;
  }

  .algo-btn.active {
    background: var(--accent);
    color: #fff;
  }

  .algo-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .param {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .param input {
    width: 58px;
  }

  .shared-param {
    margin-top: 0.5rem;
    padding-top: 0.4rem;
    border-top: 1px solid var(--border);
  }

  select {
    max-width: 110px;
  }

  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
