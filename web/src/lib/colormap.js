/**
 * Color schemes for raster visualisation.
 * Each scheme maps a normalised value [0,1] to [r, g, b] (0–255).
 */

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(c1, c2, t) {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
  ];
}

function multiStop(stops, t) {
  if (t <= 0) return stops[0][1];
  if (t >= 1) return stops[stops.length - 1][1];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const ratio =
        (t - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
      return lerpColor(stops[i - 1][1], stops[i][1], ratio);
    }
  }
  return stops[stops.length - 1][1];
}

const SCHEMES = {
  /** Green → Yellow → Brown → White (terrain / elevation-like) */
  terrain(t) {
    return multiStop(
      [
        [0.0, [34, 139, 34]],
        [0.25, [144, 190, 60]],
        [0.5, [220, 200, 80]],
        [0.75, [180, 120, 60]],
        [1.0, [255, 255, 255]],
      ],
      t,
    );
  },

  /** Blue → White → Red (divergent, good for NDVI centered at 0) */
  divergent(t) {
    return multiStop(
      [
        [0.0, [44, 62, 180]],
        [0.25, [120, 160, 220]],
        [0.5, [240, 240, 240]],
        [0.75, [220, 120, 80]],
        [1.0, [180, 30, 30]],
      ],
      t,
    );
  },

  /** Black → White */
  grayscale(t) {
    const v = Math.round(t * 255);
    return [v, v, v];
  },

  /** NDVI-specific: brown → yellow → green */
  ndvi(t) {
    return multiStop(
      [
        [0.0, [120, 70, 20]],
        [0.3, [200, 170, 60]],
        [0.5, [240, 230, 100]],
        [0.7, [100, 180, 50]],
        [1.0, [10, 100, 20]],
      ],
      t,
    );
  },

  /** Blue → White → Red (centred on 0 for signed data: curvature, TPI, DEV) */
  blue_white_red(t) {
    return multiStop(
      [
        [0.0, [33, 102, 172]],
        [0.25, [103, 169, 207]],
        [0.5, [247, 247, 247]],
        [0.75, [239, 138, 98]],
        [1.0, [178, 24, 43]],
      ],
      t,
    );
  },

  /** Geomorphons: 10 distinct landform classes */
  geomorphons(t) {
    const idx = Math.min(Math.floor(t * 10), 9);
    const palette = [
      [255, 255, 255], // 0: flat
      [56, 168, 0],    // 1: summit
      [198, 219, 124], // 2: ridge
      [255, 255, 115], // 3: shoulder
      [255, 200, 65],  // 4: spur
      [233, 150, 0],   // 5: slope
      [255, 85, 0],    // 6: hollow
      [196, 0, 0],     // 7: footslope
      [132, 0, 168],   // 8: valley
      [0, 92, 230],    // 9: depression
    ];
    return palette[idx];
  },

  /** Water-themed: white → cyan → blue */
  water(t) {
    return multiStop(
      [
        [0.0, [240, 249, 255]],
        [0.25, [186, 228, 250]],
        [0.5, [80, 180, 230]],
        [0.75, [30, 120, 200]],
        [1.0, [8, 48, 107]],
      ],
      t,
    );
  },

  /** Flow accumulation: log-scale friendly yellow → orange → red → purple */
  accumulation(t) {
    return multiStop(
      [
        [0.0, [255, 255, 212]],
        [0.25, [254, 217, 142]],
        [0.5, [254, 153, 41]],
        [0.75, [204, 76, 2]],
        [1.0, [102, 37, 6]],
      ],
      t,
    );
  },
};

/**
 * Compute min / max ignoring NaN.
 * @param {Float32Array} data
 * @returns {{ min: number, max: number }}
 */
function dataRange(data) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return { min, max };
}

/**
 * Render a Float32Array raster to an HTMLCanvasElement.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ width: number, height: number, data: Float32Array }} raster
 * @param {string} scheme - one of 'terrain', 'divergent', 'grayscale', 'ndvi'
 * @returns {{ min: number, max: number }}
 */
export function renderToCanvas(canvas, raster, scheme = "terrain") {
  const { width, height, data } = raster;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(width, height);
  const pixels = imgData.data;

  const { min, max } = dataRange(data);
  const range = max - min || 1;
  const colorFn = SCHEMES[scheme] ?? SCHEMES.terrain;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    const off = i * 4;
    if (!Number.isFinite(v)) {
      pixels[off] = 0;
      pixels[off + 1] = 0;
      pixels[off + 2] = 0;
      pixels[off + 3] = 0;
    } else {
      const t = (v - min) / range;
      const [r, g, b] = colorFn(t);
      pixels[off] = r;
      pixels[off + 1] = g;
      pixels[off + 2] = b;
      pixels[off + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return { min, max };
}

/**
 * Draw a horizontal colour-bar legend on a canvas.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {string} scheme
 * @param {number} min
 * @param {number} max
 */
export function renderLegend(canvas, scheme, min, max) {
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d");
  const colorFn = SCHEMES[scheme] ?? SCHEMES.terrain;

  for (let x = 0; x < w; x++) {
    const t = x / (w - 1);
    const [r, g, b] = colorFn(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, 0, 1, h - 18);
  }

  ctx.fillStyle = "#e2e4ed";
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(min.toFixed(2), 2, h - 16);
  ctx.textAlign = "right";
  ctx.fillText(max.toFixed(2), w - 2, h - 16);
}
