#!/usr/bin/env python3
"""Generate publication-quality figures for the SurtGIS EMS paper.

Data from native Rust benchmarks with uncompressed Float32 DEMs.
Benchmark: crates/algorithms/examples/bench_comparison.rs
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {"SurtGIS": "#2196F3", "GDAL": "#4CAF50", "GRASS": "#FF9800", "WBT": "#F44336"}
MARKERS = {"SurtGIS": "o", "GDAL": "s", "GRASS": "^", "WBT": "D"}

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

# ============================================================
# Data from native Rust benchmarks (median ms, 10 reps, 3 warmup)
# ============================================================
# Full pipeline: read GeoTIFF -> compute -> write GeoTIFF
FULL = {
    "slope": {
        "SurtGIS": {1000: 19.8, 5000: 689.0, 10000: 2028.0, 20000: 9605.6},
        "GDAL":    {1000: 101.3, 5000: 1167.5, 10000: 3568.3, 20000: 17656.2},
        "GRASS":   {1000: 618.6, 5000: 3368.4, 10000: 9696.9, 20000: 43382.1},
        "WBT":     {1000: 393.0, 5000: 6113.5, 10000: 19701.8, 20000: 107900.7},
    },
    "aspect": {
        "SurtGIS": {1000: 56.6, 5000: 1058.6, 10000: 2527.2, 20000: 14498.3},
        "GDAL":    {1000: 138.5, 5000: 2110.1, 10000: 4943.1, 20000: 22258.9},
        "GRASS":   {1000: 663.3, 5000: 3724.1, 10000: 12113.5, 20000: 55871.2},
        "WBT":     {1000: 336.4, 5000: 4406.7, 10000: 19893.0},
    },
    "hillshade": {
        "SurtGIS": {1000: 32.0, 5000: 768.3, 10000: 3616.9},
        "GDAL":    {1000: 119.0, 5000: 665.3, 10000: 2196.2},
        "GRASS":   {1000: 674.4, 5000: 4099.0, 10000: 14930.6},
        "WBT":     {1000: 332.5, 5000: 3527.3, 10000: 11460.4},
    },
}

# Compute-only (excluding I/O)
COMPUTE = {
    "slope":     {1000: 12, 5000: 241, 10000: 859, 20000: 3890},
    "aspect":    {1000: 38, 5000: 377, 10000: 1271, 20000: 6363},
    "hillshade": {1000: 22, 5000: 415, 10000: 2410},
}


# ===== FIGURE 1: Scalability (log-log) =====
def fig1_scalability():
    algs = ["slope", "aspect", "hillshade"]
    tools = ["SurtGIS", "GDAL", "GRASS", "WBT"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, alg in zip(axes, algs):
        for tool in tools:
            sizes = sorted(FULL[alg][tool].keys())
            times = [FULL[alg][tool][s] / 1000.0 for s in sizes]
            ax.loglog(sizes, times, marker=MARKERS[tool], color=COLORS[tool],
                      label=tool, linewidth=2, markersize=6)

        # Compute-only for SurtGIS
        comp_sizes = sorted(COMPUTE[alg].keys())
        comp_times = [COMPUTE[alg][s] / 1000.0 for s in comp_sizes]
        ax.loglog(comp_sizes, comp_times, marker="o", color=COLORS["SurtGIS"],
                  linestyle="--", alpha=0.5, linewidth=1.5, label="SurtGIS (compute)")

        ax.set_xlabel("DEM size (cells per side)")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(alg.capitalize(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xticks([1000, 5000, 10000, 20000])
        ax.set_xticklabels(["1K", "5K", "10K", "20K"])

    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_scalability.pdf")
    fig.savefig(FIGURES / "fig1_scalability.png", dpi=300)
    plt.close(fig)
    print("Fig 1: scalability done")


# ===== FIGURE 2: Speedup heatmap =====
def fig2_speedup():
    algs = ["slope", "aspect", "hillshade"]
    tools = ["GDAL", "GRASS", "WBT"]
    all_sizes = [1000, 5000, 10000, 20000]

    # Exclude slope vs WBT: WBT uses a different derivative method
    # (Florinsky 2016 5x5 Taylor, RMSE=6.53° vs Horn 3x3), not comparable.
    skip = {("slope", "WBT")}

    matrix = []
    ylabels = []
    for alg in algs:
        for tool in tools:
            if (alg, tool) in skip:
                continue
            row = []
            for s in all_sizes:
                sg_val = FULL[alg].get("SurtGIS", {}).get(s)
                other_val = FULL[alg].get(tool, {}).get(s)
                if sg_val and other_val:
                    row.append(other_val / sg_val)
                else:
                    row.append(np.nan)
            matrix.append(row)
            ylabels.append(f"{alg} vs {tool}")

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=20)

    xlabels = ["1K", "5K", "10K", "20K"]
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("DEM size (cells per side)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                txt = f"{v:.1f}x"
                color = "white" if v > 12 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold")
            else:
                ax.text(j, i, "---", ha="center", va="center",
                        fontsize=8, color="gray")

    # Separator lines between algorithm groups (slope has 2 rows, aspect 3, hillshade 3)
    for i in [2, 5]:
        ax.axhline(i - 0.5, color="white", linewidth=2)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Speedup (tool / SurtGIS)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_speedup.pdf")
    fig.savefig(FIGURES / "fig2_speedup.png", dpi=300)
    plt.close(fig)
    print("Fig 2: speedup heatmap done (slope vs WBT excluded)")


# ===== FIGURE 3: Cross-platform =====
def fig3_crossplatform():
    """Uses data from experiment3_crossplatform.csv if available."""
    csv_path = RESULTS / "experiment3_crossplatform.csv"
    if not csv_path.exists():
        print("Fig 3: SKIPPED (no experiment3_crossplatform.csv)")
        return

    data = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t = row["time_seconds"]
            if t in ("TIMEOUT", "ERROR", "PENDING"):
                continue
            key = (row["algorithm"], row["target"])
            data[key].append(float(t))

    algs = ["slope", "aspect", "hillshade"]
    targets = ["rust_multithread", "rust_singlethread", "python_bindings"]
    target_labels = ["Rust (MT)", "Rust (ST)", "Python"]
    target_colors = ["#2196F3", "#64B5F6", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(algs))
    width = 0.25

    for i, (target, label, color) in enumerate(zip(targets, target_labels, target_colors)):
        meds = []
        for alg in algs:
            vals = data.get((alg, target), [])
            meds.append(np.median(vals) if vals else 0)
        ax.bar(x + i * width, meds, width, label=label, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([a.capitalize() for a in algs])
    ax.set_ylabel("Time (s)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_crossplatform.pdf")
    fig.savefig(FIGURES / "fig3_crossplatform.png", dpi=300)
    plt.close(fig)
    print("Fig 3: cross-platform done")


# ===== FIGURE 4: Accuracy =====
def fig4_accuracy():
    csv_path = RESULTS / "experiment2_accuracy.csv"
    if not csv_path.exists():
        print("Fig 4: SKIPPED (no experiment2_accuracy.csv)")
        return

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel A: Analytical accuracy
    analytical = {}
    for row in rows:
        if row["comparison"] == "surtgis_vs_analytical":
            key = (row["algorithm"], row["metric"])
            analytical[key] = float(row["value"])

    entries = [
        ("slope", "rmse", "Slope RMSE"),
        ("slope", "r2", "Slope R\u00b2"),
        ("aspect", "rmse", "Aspect RMSE"),
        ("aspect", "max_angular_error", "Aspect max error"),
        ("curvature_mean", "r2", "Mean curv. R\u00b2"),
        ("curvature_gaussian", "r2", "Gaussian curv. R\u00b2"),
    ]

    table_data = []
    row_labels = []
    for alg, metric, label in entries:
        val = analytical.get((alg, metric))
        if val is not None:
            if metric == "r2":
                table_data.append(f"{val:.6f}")
            else:
                table_data.append(f"{val:.6f}\u00b0")
            row_labels.append(label)

    ax1.axis("off")
    cell_text = [[v] for v in table_data]
    colors = [["#E3F2FD"] if i % 2 == 0 else ["white"] for i in range(len(cell_text))]

    table = ax1.table(cellText=cell_text, rowLabels=row_labels, colLabels=["Value"],
                      cellColours=colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax1.set_title("(a) Analytical validation", fontsize=10, fontweight="bold", pad=20)

    # Panel B: Cross-tool agreement
    cross_tool = {}
    for row in rows:
        if row["metric"] == "rmse" and row["algorithm"] == "slope" and "analytical" not in row["comparison"]:
            cross_tool[row["comparison"]] = float(row["value"])

    comparisons = ["surtgis_vs_gdal", "surtgis_vs_grass", "surtgis_vs_wbt"]
    comp_labels = ["SurtGIS\nvs GDAL", "SurtGIS\nvs GRASS", "SurtGIS\nvs WBT"]
    comp_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    rmse_vals = [cross_tool.get(c, 0) for c in comparisons]

    bars = ax2.bar(comp_labels, rmse_vals, color=comp_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"{val:.4f}\u00b0", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylabel("Slope RMSE (\u00b0)")
    ax2.set_title("(b) Cross-tool agreement (slope)", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_accuracy.pdf")
    fig.savefig(FIGURES / "fig4_accuracy.png", dpi=300)
    plt.close(fig)
    print("Fig 4: accuracy done")


if __name__ == "__main__":
    fig1_scalability()
    fig2_speedup()
    fig3_crossplatform()
    fig4_accuracy()
    print(f"\nAll figures saved to: {FIGURES}/")
