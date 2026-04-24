# Extract patches: points vs polygons

`surtgis extract-patches` writes CNN-ready NPY tensors from feature
rasters. The two modes answer different questions.

## Points mode

Use when you have **labels at specific point locations** and want spatial
context (a patch) around each point for the model to learn from.

Example: 60 water wells with measured water-table depth. Each well gets
a 256 × 256 patch of the surrounding terrain/imagery features; the
model learns to predict water depth from that spatial context.

```bash
surtgis extract-patches \
    --features-dir factors/ \
    --points wells.gpkg \
    --label-col water_depth \
    --size 256 \
    patches/
```

- One patch per point.
- `--label-col` can be numeric (i64 for classification, f32 for regression,
  auto-detected) or a string that parses to a number.
- Points whose patch would extend outside the raster are silently skipped.

## Polygons mode

Use when you have **labelled regions** and want many patches per region
for training a CNN to classify or regress pixels.

Example: a shapefile with land-cover polygons (crop, pasture, bare soil,
forest). Each polygon produces many patches; the model learns
per-polygon class from spatial context.

```bash
surtgis extract-patches \
    --features-dir factors/ \
    --polygons land_use.gpkg \
    --label-col class \
    --size 256 \
    --stride 128 \
    patches/
```

- Grid sampling: patches are placed every `--stride` pixels (default =
  `--size`, giving non-overlapping tiles). Setting `--stride` to half of
  `--size` gives 50 % overlap, which common for data augmentation.
- Each candidate centre is tested for point-in-polygon; patches whose
  centre falls inside the polygon inherit its `--label-col` value.
- MultiPolygons are flattened. Interior rings (holes) are **not** honoured
  in v1 — patch centres in holes still get the outer polygon's label.
  This is a documented limitation.

## Which to pick

- **Sparse labels on a grid you care about** → points.
- **Dense labels over connected regions** → polygons.
- **Both** → run twice to two different output directories and merge.

## Output format

Both modes write the same four files:

```text
patches_dir/
├── patches.npy    # [N, bands, H, W] f32
├── labels.npy     # [N] i64 or f32
├── manifest.csv   # idx, label, center_row/col, center_x/y, source_idx
└── meta.json      # bands order, CRS, pixel size, skipped counts, seed
```

Load in Python:

```python
import numpy as np
X = np.load('patches_dir/patches.npy')  # ready for torch.from_numpy(X)
y = np.load('patches_dir/labels.npy')
```

## Memory ceiling

The entire `patches.npy` tensor is held in RAM before writing in v0.7.0.
For `N = 10,000` patches of `256 × 256 × 10 bands × 4 bytes`, that's
26 GB. Two mitigations:

- `--max-patches N` deterministically subsamples down to `N` (seeded by
  `--seed`, default 42). Useful for balancing very dense polygon grids.
- Shrink `--size` or `--stride` to reduce patch count.

Streaming writes (no full-tensor-in-RAM) are on the roadmap; they require
knowing `N` upfront without extracting data, which is doable but not yet
done.

## NaN filtering

`--skip-nan-threshold 0.1` (default) skips any patch where more than 10 %
of pixels are NaN across all bands. Raise it to `0.5` if you're tolerant
of partial coverage; lower it to `0.0` to require fully valid patches.
The per-patch NaN count lands in `manifest.csv` for downstream filtering.
