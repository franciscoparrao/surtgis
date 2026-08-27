#!/usr/bin/env python3
"""Parameter sweep + acceptance metrics for the Quebrada de Macul (3 May 1993)
calibration of `surtgis flow run` (spec surtgis-flow §7, milestone N4).

Runs the solver over a grid of Voellmy/entrainment parameters and scores each
run against the observable predictions:

  * **runout**: distance from the fan apex to the farthest deposited cell.
    The 1993 event's testable target is a front reaching ~947 m from the apex.
  * **deposited volume**: total and, with an observed footprint, inside it.
  * **IoU**: intersection-over-union of simulated and observed footprints —
    the headline agreement number. Skipped when no footprint is supplied.

Every run is independent, so an interrupted sweep resumes: existing rows in
the output CSV are kept and their parameter combinations are not re-run.

Usage
-----
    python3 benchmarks/calibrate_macul.py \
        --dem      macul_dem_filled.tif \
        --release  macul_release_v2.tif \
        --erodible macul_emax.tif \
        --observed huella_1993.tif \
        --apex 348120 6293450 \
        --duration 900 --outdir sweep_macul

Parameter grids are given as comma-separated lists (`--mu 0.10,0.12,0.15`).
The defaults bracket the values usually reported for Andean debris flows;
narrow them once a first pass localises the optimum.

Notes
-----
The solver uses the horizontal-coordinate (g·tanθ) formulation, so the μ/ξ
found here are effective parameters of THIS solver and are not
interchangeable with RAMMS/r.avaflow values (see `surtgis-flow` crate docs).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

try:
    import numpy as np
    import rasterio
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("needs numpy + rasterio: pip install numpy rasterio")


# ── metrics ──────────────────────────────────────────────────────────────

def _read(path: Path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float64"), src.transform, src.crs


def footprint(h, threshold: float):
    """Boolean deposit mask: cells whose final thickness exceeds `threshold`."""
    return np.isfinite(h) & (h > threshold)


def runout_distance(mask, transform, apex_xy) -> float:
    """Distance (m) from the apex to the farthest cell of `mask`, 0 if empty.

    Uses cell centres; the apex is given in the raster's CRS.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0.0
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    dx = np.asarray(xs) - apex_xy[0]
    dy = np.asarray(ys) - apex_xy[1]
    return float(np.sqrt(dx * dx + dy * dy).max())


def iou(sim_mask, obs_mask) -> float | None:
    """Intersection over union; None when the observed mask is empty."""
    union = np.logical_or(sim_mask, obs_mask).sum()
    if union == 0:
        return None
    return float(np.logical_and(sim_mask, obs_mask).sum() / union)


def cell_area(transform) -> float:
    return abs(transform.a * transform.e)


# ── one run ──────────────────────────────────────────────────────────────

def run_once(args, mu: float, xi: float, k: float, tag: str) -> dict:
    """Invoke the CLI for one parameter triple and score the last frame."""
    outdir = Path(args.outdir) / tag
    cmd = [
        args.binary, "flow", "run",
        str(args.dem), str(args.release), str(outdir),
        "--mu", f"{mu}", "--xi", f"{xi}",
        "--duration", f"{args.duration}",
        "--output-interval", f"{args.output_interval}",
    ]
    if args.erodible:
        cmd += ["--erodible", str(args.erodible), "--entrainment-k", f"{k}"]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        # A failed combination is data too — record why instead of aborting
        # the sweep (a diverged or budget-violating run is informative).
        return {
            "tag": tag, "mu": mu, "xi": xi, "k": k, "status": "failed",
            "wall_s": round(wall, 1),
            "error": (proc.stderr.strip().splitlines() or ["unknown"])[-1][:200],
        }

    frames = sorted(outdir.glob("h_t*.tif"))
    if not frames:
        return {"tag": tag, "mu": mu, "xi": xi, "k": k, "status": "no-output",
                "wall_s": round(wall, 1)}

    h, transform, _ = _read(frames[-1])
    area = cell_area(transform)
    sim = footprint(h, args.h_deposit)

    row = {
        "tag": tag, "mu": mu, "xi": xi, "k": k, "status": "ok",
        "wall_s": round(wall, 1),
        "runout_m": round(runout_distance(sim, transform, args.apex), 1),
        "footprint_ha": round(sim.sum() * area / 1e4, 2),
        "volume_m3": round(float(np.nansum(h[sim])) * area, 0),
    }

    # Lowest elevation the flow reached. For a channelised event this is a
    # sharper acceptance test than planimetric distance: Macul 1993 has to
    # reach the fan apex at 947 m, and pre-entrainment runs stalled at
    # 959–978 m (GEODEO docs/validacion-surtgis-flow.md).
    if args.dem:
        dem, _, _ = _read(args.dem)
        if dem.shape == h.shape and sim.any():
            row["min_elev_m"] = round(float(np.nanmin(dem[sim])), 1)
            if args.target_elevation:
                row["reaches_apex"] = row["min_elev_m"] <= args.target_elevation
    # The manifest carries the solver's own mass ledger (entrainment, borders).
    manifest = outdir / "manifest.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text())
        ent = meta.get("entrainment") or {}
        if "total_eroded_m3" in ent:
            row["eroded_m3"] = ent["total_eroded_m3"]

    if args.observed:
        obs, obs_tf, _ = _read(args.observed)
        if obs.shape != h.shape:
            row["iou"] = None
            row["iou_note"] = "observed footprint not on the simulation grid"
        else:
            obs_mask = np.isfinite(obs) & (obs > 0)
            row["iou"] = round(iou(sim, obs_mask) or 0.0, 4)

    if args.target_runout:
        row["runout_err_m"] = round(row["runout_m"] - args.target_runout, 1)
    return row


# ── sweep ────────────────────────────────────────────────────────────────

def parse_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--erodible", type=Path, help="e_max raster; enables entrainment")
    p.add_argument("--observed", type=Path, help="observed footprint (>0 = affected)")
    p.add_argument("--apex", nargs=2, type=float, metavar=("X", "Y"), required=True,
                   help="fan apex in the DEM's CRS — the runout origin")
    p.add_argument("--target-runout", type=float, default=947.0,
                   help="observed front distance from the apex, m (default: 947)")
    p.add_argument("--target-elevation", type=float,
                   help="elevation the front must reach, m (Macul 1993: 947); "
                        "scored from --dem")
    p.add_argument("--mu", default="0.08,0.10,0.12,0.15,0.20")
    p.add_argument("--xi", default="300,500,700,1000")
    p.add_argument("--k", default="1e-4,5e-4,1e-3,5e-3")
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--output-interval", type=float, default=900.0,
                   help="only the last frame is scored; keep it large to save I/O")
    p.add_argument("--h-deposit", type=float, default=0.1,
                   help="thickness (m) above which a cell counts as deposited")
    p.add_argument("--outdir", type=Path, default=Path("sweep_macul"))
    p.add_argument("--binary", default="surtgis")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "sweep.csv"

    done: set[str] = set()
    rows: list[dict] = []
    if csv_path.is_file():
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
                done.add(row["tag"])
        print(f"resuming: {len(done)} runs already in {csv_path}")

    mus, xis = parse_list(args.mu), parse_list(args.xi)
    ks = parse_list(args.k) if args.erodible else [0.0]
    combos = list(product(mus, xis, ks))
    print(f"{len(combos)} combinations "
          f"({len(mus)} mu x {len(xis)} xi x {len(ks)} k), "
          f"{len(combos) - len(done)} to run")

    for i, (mu, xi, k) in enumerate(combos, 1):
        tag = f"mu{mu}_xi{xi}_k{k}"
        if tag in done:
            continue
        row = run_once(args, mu, xi, k, tag)
        rows.append(row)
        flag = ""
        if row.get("status") == "ok":
            flag = f"runout={row['runout_m']}m"
            if row.get("iou") is not None:
                flag += f" IoU={row['iou']}"
        else:
            flag = row.get("status", "?")
        print(f"[{i}/{len(combos)}] {tag}: {flag} ({row.get('wall_s')}s)")

        # Persist after every run so an interrupted sweep loses nothing.
        fields: list[str] = []
        for r in rows:
            for key in r:
                if key not in fields:
                    fields.append(key)
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # ── ranking ──────────────────────────────────────────────────────────
    ok = [r for r in rows if str(r.get("status")) == "ok"]
    if not ok:
        print("\nno successful run to rank")
        return 1

    def score(r) -> float:
        """Rank by IoU when an observed footprint exists; otherwise by how
        far the front got — lowest elevation reached when a DEM is given
        (the sharper test for a channelised event), else runout error."""
        if r.get("iou") not in (None, "", "None"):
            return -float(r["iou"])
        if r.get("min_elev_m") not in (None, "", "None"):
            return float(r["min_elev_m"])
        return abs(float(r.get("runout_m", 0)) - args.target_runout)

    ok.sort(key=score)
    print(f"\nbest 10 of {len(ok)} successful runs "
          f"({'IoU' if args.observed else f'|runout − {args.target_runout} m|'}):")
    for r in ok[:10]:
        line = (f"  mu={r['mu']:<6} xi={r['xi']:<6} k={r['k']:<8} "
                f"area={r['footprint_ha']:>7} ha")
        if r.get("min_elev_m") not in (None, "", "None"):
            mark = " REACHES APEX" if str(r.get("reaches_apex")) == "True" else ""
            line += f"  min_elev={r['min_elev_m']:>7} m{mark}"
        else:
            line += f"  runout={r['runout_m']:>7} m"
        if r.get("iou") not in (None, "", "None"):
            line += f"  IoU={r['iou']}"
        print(line)
    print(f"\nfull results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
