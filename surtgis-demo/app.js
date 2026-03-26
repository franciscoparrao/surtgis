import init, {
    slope,
    aspect_degrees,
    hillshade_compute,
    tpi_compute,
    tri_compute,
    fill_depressions,
    flow_direction_d8,
    flow_accumulation_d8,
    morph_erode,
    morph_dilate,
    focal_mean
} from './wasm/surtgis_wasm.js';

// Initialize Mapbox (use a generic style, no API key needed for basic map)
mapboxgl.accessToken = '';

// State
let wasmReady = false;
let currentDem = null;
let currentRaster = null;
let currentResult = null;
let currentExtent = null;
let map = null;

// DOM elements
const demFile = document.getElementById('demFile');
const algorithm = document.getElementById('algorithm');
const computeBtn = document.getElementById('computeBtn');
const downloadBtn = document.getElementById('downloadBtn');
const status = document.getElementById('status');
const fileStatus = document.getElementById('fileStatus');
const radius = document.getElementById('radius');
const azimuth = document.getElementById('azimuth');
const progress = document.getElementById('progress');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');

// Initialize WASM
(async () => {
    try {
        await init();
        wasmReady = true;
        updateStatus('✓ WASM loaded. Ready to process.');
    } catch (err) {
        updateStatus(`✗ WASM init failed: ${err.message}`);
    }
})();

// Initialize Map
function initMap() {
    if (!map) {
        map = new mapboxgl.Map({
            container: 'map',
            style: 'mapbox://styles/mapbox/satellite-v9',
            center: [0, 0],
            zoom: 2,
            attributionControl: false
        });
        map.addControl(new mapboxgl.NavigationControl());
    }
}

initMap();

// File upload handler
demFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    updateStatus(`Loading ${file.name}...`);
    fileStatus.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;

    try {
        const buffer = await file.arrayBuffer();
        currentDem = new Uint8Array(buffer);

        // Try to parse GeoTIFF metadata
        try {
            const tiff = await GeoTIFF.fromArrayBuffer(buffer);
            const image = await tiff.getImage();
            const bbox = image.getBoundingBox();
            const [width, height] = image.getSize();

            // Store extent for later visualization
            currentExtent = {
                bbox: bbox,
                width: width,
                height: height,
                data: null
            };

            updateStatus(`✓ Loaded: ${width}x${height}px`);
            fileStatus.textContent = `${width}x${height}px, ${(file.size / 1024 / 1024).toFixed(1)}MB`;

            // Center map on data
            const [minX, minY, maxX, maxY] = bbox;
            map.fitBounds([[minX, minY], [maxX, maxY]], { padding: 20 });
        } catch (geoErr) {
            updateStatus('✓ Loaded (GeoTIFF metadata unavailable, visualization may be limited)');
        }
    } catch (err) {
        updateStatus(`✗ Error loading file: ${err.message}`);
    }
});

// Compute handler
computeBtn.addEventListener('click', async () => {
    if (!wasmReady) {
        updateStatus('⏳ WASM not ready yet, please wait...');
        return;
    }

    if (!currentDem) {
        updateStatus('⚠ Please upload a GeoTIFF first');
        return;
    }

    const algo = algorithm.value;
    const param = parseInt(radius.value) || 3;
    const az = parseFloat(azimuth.value) || 315;

    showProgress(true);

    try {
        updateStatus(`Computing ${algo}...`);

        let result;
        const startTime = performance.now();

        switch (algo) {
            case 'slope':
                result = slope(currentDem, 'degrees');
                break;
            case 'aspect':
                result = aspect_degrees(currentDem);
                break;
            case 'hillshade':
                result = hillshade_compute(currentDem, az, 45);
                break;
            case 'tpi':
                result = tpi_compute(currentDem, param);
                break;
            case 'tri':
                result = tri_compute(currentDem);
                break;
            case 'fill':
                result = fill_depressions(currentDem);
                break;
            case 'flow_direction':
                result = flow_direction_d8(currentDem);
                break;
            case 'flow_accumulation':
                // Requires flow direction first
                const fdir = flow_direction_d8(currentDem);
                result = flow_accumulation_d8(fdir);
                break;
            case 'erode':
                result = morph_erode(currentDem, param);
                break;
            case 'dilate':
                result = morph_dilate(currentDem, param);
                break;
            case 'focal_mean':
                result = focal_mean(currentDem, param);
                break;
            default:
                throw new Error(`Unknown algorithm: ${algo}`);
        }

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        currentResult = result;

        // Parse and visualize
        await visualizeResult(result, algo);

        updateStatus(`✓ ${algo} complete in ${elapsed}s`);
        downloadBtn.disabled = false;
    } catch (err) {
        updateStatus(`✗ Error: ${err.message}`);
        console.error(err);
    } finally {
        showProgress(false);
    }
});

// Visualize result
async function visualizeResult(resultBytes, algoName) {
    try {
        const tiff = await GeoTIFF.fromArrayBuffer(resultBytes);
        const image = await tiff.getImage();
        const [width, height] = image.getSize();
        const data = await image.readRasters();

        // Get first raster band as float64 array
        const rasterData = data[0];

        // Create canvas from raster data
        const canvas = createRasterCanvas(rasterData, width, height, algoName);

        // Add or update layer
        if (map.getSource('result')) {
            map.removeLayer('result');
            map.removeSource('result');
        }

        map.addSource('result', {
            type: 'canvas',
            canvas: canvas,
            animate: false
        });

        map.addLayer(
            {
                id: 'result',
                type: 'raster',
                source: 'result',
                paint: {
                    'raster-opacity': 0.85
                }
            },
            'waterway' // Insert below water layer if exists
        );

        // Add legend
        addLegend(algoName, rasterData);
    } catch (err) {
        console.error('Visualization error:', err);
        // Fallback: just show a message
        updateStatus(`✓ Computed (visualization unavailable)`);
    }
}

// Create canvas from raster data
function createRasterCanvas(data, width, height, algoName) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;

    // Normalize data to [0, 1]
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
        if (!isNaN(data[i]) && isFinite(data[i])) {
            min = Math.min(min, data[i]);
            max = Math.max(max, data[i]);
        }
    }

    if (max === min) max = min + 1; // Avoid division by zero

    // Select colormap
    const colormap = getColormap(algoName);

    // Map each value to RGBA
    for (let i = 0; i < data.length; i++) {
        const val = data[i];
        let normalized = 0;

        if (!isNaN(val) && isFinite(val)) {
            normalized = (val - min) / (max - min);
            normalized = Math.max(0, Math.min(1, normalized));
        }

        const [r, g, b] = getColorForValue(normalized, colormap);
        pixels[i * 4] = r;
        pixels[i * 4 + 1] = g;
        pixels[i * 4 + 2] = b;
        pixels[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

// Get color for normalized value [0, 1]
function getColorForValue(value, colormap) {
    const colors = colormaps[colormap];
    const idx = Math.floor(value * (colors.length - 1));
    return colors[Math.min(idx, colors.length - 1)];
}

// Select colormap based on algorithm
function getColormap(algoName) {
    const colormapMap = {
        'slope': 'viridis',
        'aspect': 'hsv',
        'hillshade': 'gray',
        'tpi': 'seismic',
        'tri': 'viridis',
        'fill': 'viridis',
        'flow_direction': 'hsv',
        'flow_accumulation': 'plasma',
        'erode': 'gray',
        'dilate': 'gray',
        'focal_mean': 'viridis'
    };
    return colormapMap[algoName] || 'viridis';
}

// Download handler
document.getElementById('downloadBtn').addEventListener('click', () => {
    if (!currentResult) {
        updateStatus('⚠ No result to download');
        return;
    }

    const blob = new Blob([currentResult], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `surtgis_${algorithm.value}_${Date.now()}.tif`;
    a.click();
    URL.revokeObjectURL(url);

    updateStatus('✓ Downloaded');
});

// Helper functions
function updateStatus(msg) {
    status.textContent = msg;
    console.log(msg);
}

function showProgress(show) {
    if (show) {
        progress.classList.remove('hidden');
        computeBtn.disabled = true;
    } else {
        progress.classList.add('hidden');
        computeBtn.disabled = !wasmReady || !currentDem;
    }
}

function addLegend(algoName, data) {
    // Remove old legend if exists
    const oldLegend = document.getElementById('legend');
    if (oldLegend) oldLegend.remove();

    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
        if (!isNaN(data[i]) && isFinite(data[i])) {
            min = Math.min(min, data[i]);
            max = Math.max(max, data[i]);
        }
    }

    if (max === min) max = min + 1;

    const legend = document.createElement('div');
    legend.id = 'legend';
    legend.style.cssText = `
        position: absolute;
        bottom: 20px;
        right: 20px;
        background: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 12px;
        z-index: 100;
    `;

    legend.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 6px;">${algoName}</div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span>Min: ${min.toFixed(2)}</span>
            <div style="width: 100px; height: 10px; background: linear-gradient(to right, #440154, #31688e, #35b779, #fde724); border-radius: 2px;"></div>
            <span>Max: ${max.toFixed(2)}</span>
        </div>
    `;

    document.getElementById('map').appendChild(legend);
}

// Colormaps - simple 256-color lookups
const colormaps = {
    'viridis': generateViridis(),
    'plasma': generatePlasma(),
    'hsv': generateHSV(),
    'gray': generateGray(),
    'seismic': generateSeismic()
};

function generateViridis() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        const r = Math.round(68 * (1 - t) + 253 * t);
        const g = Math.round(1 * (1 - t) + 231 * t);
        const b = Math.round(84 * (1 - t) + 36 * t);
        colors.push([r, g, b]);
    }
    return colors;
}

function generatePlasma() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        const r = Math.round(13 + 242 * Math.pow(t, 0.5));
        const g = Math.round(8 + 229 * t);
        const b = Math.round(135 + 100 * (1 - t));
        colors.push([r, Math.min(255, g), Math.min(255, b)]);
    }
    return colors;
}

function generateHSV() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
        const h = (i / 255) * 360;
        const [r, g, b] = hslToRgb(h / 360, 1, 0.5);
        colors.push([r, g, b]);
    }
    return colors;
}

function generateGray() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
        colors.push([i, i, i]);
    }
    return colors;
}

function generateSeismic() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        const r = t > 0.5 ? Math.round(2 * (t - 0.5) * 255) : 0;
        const b = t < 0.5 ? Math.round(2 * (0.5 - t) * 255) : 0;
        const g = Math.sin(t * Math.PI) * 200;
        colors.push([r, Math.round(g), b]);
    }
    return colors;
}

function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// Update button states
document.getElementById('computeBtn').disabled = true;
document.getElementById('downloadBtn').disabled = true;

// Listen for WASM readiness and file upload
const updateButtonStates = () => {
    computeBtn.disabled = !wasmReady || !currentDem;
};

const originalDemFileListener = demFile.onchange;
demFile.addEventListener('change', updateButtonStates);

// Periodically check WASM status
setInterval(() => {
    if (wasmReady && !currentDem) {
        computeBtn.disabled = true;
    }
}, 500);
