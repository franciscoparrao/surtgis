# Prepare training data for a Geospatial Foundation Model

This guide walks through the full pipeline from a STAC bbox to a tensor
ready to fine-tune **Prithvi-EO-2.0** (NASA / IBM). The same flow works
for Clay v1.5 by swapping one flag.

The motivation: in October 2025 the [InstaGeo paper](https://arxiv.org/html/2510.05617v1)
identified that no published Geospatial Foundation Model ships its
preprocessing pipeline — users get only model checkpoints. SurtGIS
fills that gap with a single CLI command.

## Step 1 — fetch a multi-temporal HLS composite

Prithvi was pre-trained on Harmonized Landsat-Sentinel-2 (HLS). For
each timestamp you want in the temporal stack, run one `stac composite`
call into its own subdirectory. The subdirectory name becomes the
timestamp label in the output tensor.

```bash
# 2024 Maule, Chile — three months of HLS coverage
for month in 01 02 03; do
    surtgis stac composite \
        --catalog pc \
        --collection hls2-s30 \
        --asset "B02,B03,B04,B05,B06,B07" \
        --bbox=-72.0,-35.5,-71.5,-35.0 \
        --datetime "2024-${month}-01/2024-${month}-31" \
        --max-scenes 10 \
        features/t_2024_${month}/composite.tif
done
```

Each composite uses the HLS Fmask asset for cloud masking
automatically — the asset name "Fmask" triggers SurtGIS's
[`HlsFmask`](../reference/cli/stac.md) strategy (cloud, adjacent-to-cloud,
and cloud-shadow bits dropped; cirrus and snow kept). The output is
one `.tif` per band per timestamp.

After the loop your tree should look like:

```text
features/
├── t_2024_01/
│   ├── composite_B02.tif
│   ├── composite_B03.tif
│   └── ... (one per asset)
├── t_2024_02/
│   └── ...
└── t_2024_03/
    └── ...
```

## Step 2 — extract Prithvi-ready chips

One command turns the per-timestamp directories into a tensor matched
to Prithvi-EO-2.0's input convention:

```bash
surtgis extract-patches \
    --features-dir features/ \
    --points labels.geojson \
    --label-col landslide_class \
    --profile prithvi-v2 \
    --size 224 \
    --output-format zarr \
    --emit-stac \
    chips/
```

What the flags do, in order:

- `--features-dir features/`: SurtGIS auto-detects multi-timestamp mode
  because the directory contains subdirectories (each holding the same
  band set). Subdirs are sorted lexicographically — that's why we used
  `t_2024_01`, `t_2024_02`, `t_2024_03`.
- `--profile prithvi-v2`: validates that 6 bands are present in the
  order Prithvi expects (B02, B03, B04, B05, B06, B07), then applies
  the per-band z-score normalization from the official
  [Prithvi-EO-2.0-300M config](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M).
  Records full provenance in `meta.json`.
- `--size 224`: matches Prithvi's pre-training tile size. The profile
  warns if you pass a different size, since the model would then need
  to resize at training time.
- `--output-format zarr`: emit chunked Zarr v2 instead of a single
  `.npy`. One chunk per chip, parallel I/O. Optional but useful when
  N is large.
- `--emit-stac`: write a STAC ML-AOI Collection and one Item per chip,
  embedding the [STAC MLM extension](https://github.com/stac-extensions/mlm)
  with `mlm:model_target = ibm-nasa-geospatial/Prithvi-EO-2.0-300M` and
  the full `mlm:input` descriptor. This is what makes the dataset
  publishable on its own.

The output tree:

```text
chips/
├── patches.zarr/            # [N, 6, 3, 224, 224] f32, chunk = 1 chip
│   ├── .zarray
│   ├── .zattrs
│   └── 0.0.0.0.0, 1.0.0.0.0, ...
├── labels.npy               # [N] i64 or f32
├── manifest.csv
├── meta.json                # bands, timestamps, gfm_profile{...}
└── stac/
    ├── collection.json      # MLM + ML-AOI
    └── items/
        ├── chip_000000.json
        └── chip_000001.json
```

The tensor shape `[N, 6, 3, 224, 224]` is the standard
`[batch, channels, time, height, width]` that Prithvi accepts directly.

## Step 3 — load and fine-tune

With TerraTorch (the IBM fine-tuning toolkit):

```python
import zarr
import numpy as np
import torch

X = zarr.open('chips/patches.zarr', mode='r')   # [N, 6, 3, 224, 224]
y = np.load('chips/labels.npy')

# Materialise into a Torch dataset — the Zarr open is lazy, [:] forces it.
X_tensor = torch.from_numpy(X[:])
y_tensor = torch.from_numpy(y)

# Prithvi-EO-2.0 expects (B, C, T, H, W) which is exactly our layout.
# No transpose, no extra normalization — SurtGIS already z-scored using
# the official Prithvi statistics.
```

The profile metadata in `meta.json` lets a TerraTorch config file pick
up the right band order, normalization params, and tile size without
hand-editing:

```python
import json
with open('chips/meta.json') as f: m = json.load(f)
print(m['gfm_profile']['model_target'])
# → 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'
print(m['tensor_layout'])
# → '[N, C, T, H, W]'
```

## Variation — Clay v1.5 instead

Clay expects 10 Sentinel-2 bands at tile 256 with reflectance values in
`[0, 1]`. Swap the profile and band list:

```bash
surtgis stac composite \
    --catalog pc --collection sentinel-2-l2a \
    --asset "B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12" \
    --bbox=-72.0,-35.5,-71.5,-35.0 --datetime 2024-02-01/2024-02-28 \
    features/t0/composite.tif

surtgis extract-patches \
    --features-dir features/ --points labels.geojson \
    --label-col landslide_class --profile clay-v1.5 --size 256 \
    --output-format zarr --emit-stac chips/
```

Cloud masking switches to Sentinel-2 SCL automatically because the
collection is L2A, not HLS. No flag needed.

## Variation — NPY instead of Zarr

If your downstream loader is `np.load()`-based, drop `--output-format zarr`:

```bash
surtgis extract-patches \
    --features-dir features/ --points labels.geojson \
    --label-col landslide_class --profile prithvi-v2 --size 224 \
    --emit-stac chips/
```

You get `chips/patches.npy` with the same shape and dtype. Pick Zarr
when N is large enough that `np.load` would page-fault your machine.

## Validation

After the run, verify the tensor matches what Prithvi expects:

```bash
python3 -c "
import zarr, json
X = zarr.open('chips/patches.zarr', mode='r')
print('shape:', X.shape, 'dtype:', X.dtype)
m = json.load(open('chips/meta.json'))
spec = m['gfm_profile']
print('bands:', spec['bands_order'])
print('mean:', spec['band_norm_mean'])
print('std:',  spec['band_norm_std'])
print('source:', spec['source_url'])
"
```

You should see `dtype: float32`, the canonical Prithvi band order, and
the means around `[1087, 1342, 1433, 2734, 1958, 1363]`.

## When things break

- **"Profile expects N bands, but M feature rasters were loaded"**:
  curate `features/<timestamp>/` to contain exactly the bands the
  profile expects, in the expected order. The error message lists them.
- **"Band-name mismatch at timestamp 't_2024_02'"**: all timestamp
  subdirs must declare the *same* bands. Re-run the composite step for
  the offending timestamp.
- **"Mixed mode: top-level .tif AND subdirs with .tifs"**: move the
  top-level files into a subdirectory. SurtGIS refuses to guess.
- **STAC items have bbox in source CRS instead of WGS84**: compile with
  `--features projections` (default in precompiled binaries). The
  warning message tells you when this fallback triggered.
