/**
 * WASM Benchmark for SurtGIS Paper (EMS)
 *
 * Measures execution time of key algorithms in the WASM (V8/Node.js) runtime.
 * Uses the same DEM files as the native benchmarks for comparability.
 *
 * Usage:
 *   node benchmarks/bench_wasm.mjs [--reps N] [--warmup N] [--sizes 1000,5000]
 *
 * Output:
 *   benchmarks/results/experiment_wasm.csv
 */

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { performance } from "perf_hooks";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

// ── Parse CLI args ──
const args = process.argv.slice(2);
function getArg(name, def) {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
}

const REPS = parseInt(getArg("reps", "10"));
const WARMUP = parseInt(getArg("warmup", "3"));
const SIZES = getArg("sizes", "1000,5000").split(",").map(Number);

// ── Import WASM module (nodejs target) ──
const wasm = await import(join(ROOT, "pkg-node", "surtgis_wasm.js"));

// ── Algorithms to benchmark ──
const ALGORITHMS = [
  {
    name: "slope",
    run: (bytes) => wasm.slope(bytes, "degrees"),
  },
  {
    name: "aspect",
    run: (bytes) => wasm.aspect_degrees(bytes),
  },
  {
    name: "hillshade",
    run: (bytes) => wasm.hillshade_compute(bytes, 315.0, 45.0),
  },
  {
    name: "tpi",
    run: (bytes) => wasm.tpi_compute(bytes, 10),
  },
  {
    name: "fill",
    run: (bytes) => wasm.priority_flood_fill(bytes),
  },
  {
    name: "flow_acc",
    run: (bytes) => {
      const fdir = wasm.flow_direction_d8(bytes);
      return wasm.flow_accumulation_d8(fdir);
    },
  },
];

// ── Helpers ──
function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function iqr(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  return q3 - q1;
}

function peakMemoryMB() {
  const mem = process.memoryUsage();
  return (mem.rss / 1024 / 1024).toFixed(1);
}

// ── Main ──
console.log("╔══════════════════════════════════════════════════╗");
console.log("║   SurtGIS WASM Benchmark (Node.js / V8)         ║");
console.log("╚══════════════════════════════════════════════════╝");
console.log(`Reps: ${REPS}, Warmup: ${WARMUP}, Sizes: ${SIZES.join(", ")}`);
console.log(`Node.js: ${process.version}, V8: ${process.versions.v8}`);
console.log(`WASM binary: ${(readFileSync(join(ROOT, "pkg-node", "surtgis_wasm_bg.wasm")).length / 1024).toFixed(0)} KB`);
console.log();

const results = [];

for (const size of SIZES) {
  const demPath = join(ROOT, "benchmarks", "results", "dems", `fbm_${size}_raw.tif`);
  let demBytes;
  try {
    demBytes = new Uint8Array(readFileSync(demPath));
  } catch (e) {
    console.log(`⚠ DEM ${size}x${size} not found at ${demPath}, skipping`);
    continue;
  }

  console.log(`── DEM ${size}×${size} (${(demBytes.length / 1024 / 1024).toFixed(1)} MB) ──`);

  for (const algo of ALGORITHMS) {
    // Skip heavy algorithms for large sizes
    if (size >= 5000 && (algo.name === "fill" || algo.name === "flow_acc" || algo.name === "tpi")) {
      console.log(`  ${algo.name}: skipped (too slow for ${size}² in WASM)`);
      results.push({
        algorithm: algo.name,
        size,
        target: "wasm_nodejs",
        median_s: "NA",
        iqr_s: "NA",
        peak_mem_mb: "NA",
        node_version: process.version,
        v8_version: process.versions.v8,
      });
      continue;
    }

    // Warmup
    process.stdout.write(`  ${algo.name}: warming up...`);
    for (let i = 0; i < WARMUP; i++) {
      try {
        algo.run(demBytes);
      } catch (e) {
        console.log(` ERROR: ${e.message}`);
        break;
      }
    }

    // Benchmark
    process.stdout.write(` running ${REPS} reps...`);
    const times = [];
    for (let i = 0; i < REPS; i++) {
      const t0 = performance.now();
      algo.run(demBytes);
      const t1 = performance.now();
      times.push((t1 - t0) / 1000); // seconds
    }

    const med = median(times);
    const iqrVal = iqr(times);
    const mem = peakMemoryMB();

    console.log(` median=${med.toFixed(3)}s, IQR=${iqrVal.toFixed(3)}s, mem=${mem}MB`);

    results.push({
      algorithm: algo.name,
      size,
      target: "wasm_nodejs",
      median_s: med.toFixed(4),
      iqr_s: iqrVal.toFixed(4),
      peak_mem_mb: mem,
      node_version: process.version,
      v8_version: process.versions.v8,
    });

    // Store individual runs for CSV
    for (let i = 0; i < times.length; i++) {
      results.push({
        algorithm: algo.name,
        size,
        target: "wasm_nodejs",
        run: i + 1,
        time_s: times[i].toFixed(4),
      });
    }
  }
  console.log();
}

// ── Write CSV ──
const outDir = join(ROOT, "benchmarks", "results");
mkdirSync(outDir, { recursive: true });

// Summary CSV
const summaryLines = ["algorithm,size,target,median_s,iqr_s,peak_mem_mb,node_version,v8_version"];
for (const r of results) {
  if (r.median_s !== undefined && r.run === undefined) {
    summaryLines.push(`${r.algorithm},${r.size},${r.target},${r.median_s},${r.iqr_s},${r.peak_mem_mb},${r.node_version},${r.v8_version}`);
  }
}
const summaryPath = join(outDir, "experiment_wasm.csv");
writeFileSync(summaryPath, summaryLines.join("\n") + "\n");
console.log(`Results written to ${summaryPath}`);

// Raw runs CSV
const rawLines = ["algorithm,size,target,run,time_s"];
for (const r of results) {
  if (r.run !== undefined) {
    rawLines.push(`${r.algorithm},${r.size},${r.target},${r.run},${r.time_s}`);
  }
}
const rawPath = join(outDir, "experiment_wasm_raw.csv");
writeFileSync(rawPath, rawLines.join("\n") + "\n");
console.log(`Raw runs written to ${rawPath}`);
