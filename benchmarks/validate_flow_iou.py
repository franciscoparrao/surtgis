#!/usr/bin/env python3
"""Validación cuantitativa de una corrida de `surtgis flow run` contra una
huella observada (spec surtgis-flow v1.0 §7, validación científica).

Métricas (las acordadas con GEODEO para el caso Macul 1993):
  - IoU entre la huella simulada (envolvente máxima de h sobre TODOS los
    frames, no el último — el flujo adelgaza al final y subestima el
    alcance; misma metodología que geodeo_bridge) y la huella observada.
  - Runout: elevación mínima y distancia máxima alcanzadas por la simulación.
  - Tiempo de arribo a un punto de control (p. ej. el ápice del cono),
    desde el raster de arrival de `--arrival`.

Uso:
  validate_flow_iou.py OUTDIR HUELLA [--dem DEM.tif] [--h-wet 0.02]
                       [--control-point X Y] [--report out.json]

  OUTDIR   directorio de `surtgis flow run` (h_t####.tif + manifest.json,
           idealmente corrido con --arrival OUTDIR/arrival.tif)
  HUELLA   huella observada: raster (mismo grid o reproyectable) o vector
           (cualquier formato fiona; se rasteriza al grid de la simulación)

La huella simulada usa el umbral h > H_WET (default 0.02 m, el mismo del
bridge de GEODEO). Sale con código != 0 si los grids no son conciliables.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject


def load_envelope(outdir: str, h_wet: float):
    """Envolvente máxima de h por celda sobre todos los frames."""
    frames = sorted(glob.glob(os.path.join(outdir, "h_t*.tif")))
    if not frames:
        sys.exit(f"error: no hay frames h_t*.tif en {outdir}")
    with rasterio.open(frames[0]) as first:
        profile = first.profile.copy()
        env = np.zeros((first.height, first.width), dtype="float64")
    for f in frames:
        with rasterio.open(f) as src:
            h = src.read(1)
        env = np.fmax(env, np.nan_to_num(h, nan=0.0))
    return env, (env > h_wet), profile, len(frames)


def load_observed(path: str, profile) -> np.ndarray:
    """Huella observada como máscara booleana en el grid de la simulación."""
    shape = (profile["height"], profile["width"])
    try:
        with rasterio.open(path) as src:
            if (
                src.crs == profile["crs"]
                and src.transform == profile["transform"]
                and (src.height, src.width) == shape
            ):
                data = src.read(1)
            else:
                data = np.zeros(shape, dtype="float32")
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.nearest,
                )
            return np.nan_to_num(data, nan=0.0) > 0
    except rasterio.errors.RasterioIOError:
        pass  # no es raster: probar vector
    import fiona
    from fiona.transform import transform_geom

    with fiona.open(path) as src:
        geoms = [
            transform_geom(src.crs, profile["crs"].to_dict(), f["geometry"])
            if src.crs and profile["crs"]
            else f["geometry"]
            for f in src
        ]
    if not geoms:
        sys.exit(f"error: {path} no contiene geometrías")
    return rasterize(
        [(g, 1) for g in geoms],
        out_shape=shape,
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
    ).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("huella")
    ap.add_argument("--dem", help="DEM para métricas de elevación de runout")
    ap.add_argument("--h-wet", type=float, default=0.02,
                    help="umbral mojado [m] (default 0.02, igual que geodeo_bridge)")
    ap.add_argument("--control-point", nargs=2, type=float, metavar=("X", "Y"),
                    help="punto de control (CRS de la sim) para tiempo de arribo, p. ej. el ápice")
    ap.add_argument("--report", help="escribir métricas como JSON aquí")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "manifest.json")) as f:
        manifest = json.load(f)

    env, sim_mask, profile, n_frames = load_envelope(args.outdir, args.h_wet)
    obs_mask = load_observed(args.huella, profile)

    inter = np.logical_and(sim_mask, obs_mask).sum()
    union = np.logical_or(sim_mask, obs_mask).sum()
    cell_area = manifest["cellsize"] ** 2
    metrics = {
        "n_frames": n_frames,
        "h_wet_threshold_m": args.h_wet,
        "iou": float(inter / union) if union else 0.0,
        "sim_area_km2": float(sim_mask.sum() * cell_area / 1e6),
        "obs_area_km2": float(obs_mask.sum() * cell_area / 1e6),
        "intersection_km2": float(inter * cell_area / 1e6),
        "h_max_m": float(env.max()),
        # Descomposición del error: qué fracción de lo observado se cubre y
        # cuánta área simulada cae fuera de lo observado.
        "recall_of_observed": float(inter / obs_mask.sum()) if obs_mask.sum() else 0.0,
        "precision_of_simulated": float(inter / sim_mask.sum()) if sim_mask.sum() else 0.0,
    }

    if args.dem:
        with rasterio.open(args.dem) as src:
            dem = src.read(1)
        if dem.shape != sim_mask.shape:
            sys.exit("error: el DEM no está en el grid de la simulación")
        metrics["sim_min_elevation_m"] = float(np.nanmin(np.where(sim_mask, dem, np.nan)))
        metrics["obs_min_elevation_m"] = float(np.nanmin(np.where(obs_mask, dem, np.nan)))

    if args.control_point:
        arrival_path = os.path.join(args.outdir, "arrival.tif")
        if os.path.exists(arrival_path):
            with rasterio.open(arrival_path) as src:
                arr = src.read(1)
                row, col = src.index(*args.control_point)
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                t = float(arr[row, col])
                metrics["control_point_arrival_s"] = None if np.isnan(t) else t
            else:
                metrics["control_point_arrival_s"] = "fuera del dominio"
        else:
            metrics["control_point_arrival_s"] = "sin arrival.tif (correr con --arrival)"

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
