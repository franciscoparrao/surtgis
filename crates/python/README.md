# surtgis

High-performance geospatial analysis library for Python, powered by Rust.

SurtGIS provides 40+ terrain, hydrology, and imagery algorithms with native performance through PyO3 bindings.

## Installation

```bash
pip install surtgis
```

## Quick Start

```python
import numpy as np
import surtgis

# Create a sample DEM (100x100 grid)
dem = np.random.rand(100, 100) * 100  # Elevation in meters
cell_size = 30.0  # 30m resolution

# Compute terrain derivatives
slope = surtgis.slope(dem, cell_size)
aspect = surtgis.aspect_degrees(dem, cell_size)
hillshade = surtgis.hillshade_compute(dem, cell_size, azimuth=315.0, altitude=45.0)

# Curvature analysis
profile_curv = surtgis.curvature_compute(dem, cell_size, ctype="profile")
plan_curv = surtgis.curvature_compute(dem, cell_size, ctype="plan")

# Advanced Florinsky curvatures (14 types available)
mean_h = surtgis.advanced_curvature(dem, cell_size, ctype="mean_h")
gaussian_k = surtgis.advanced_curvature(dem, cell_size, ctype="gaussian_k")

# Hydrology
twi = surtgis.twi_compute(dem, cell_size)  # Topographic Wetness Index
hand = surtgis.hand_compute(dem, cell_size)  # Height Above Nearest Drainage

print(f"Slope range: {slope.min():.1f} - {slope.max():.1f}")
print(f"TWI range: {twi.min():.2f} - {twi.max():.2f}")
```

## Available Functions

### Terrain Analysis (21 functions)

| Function | Description |
|----------|-------------|
| `slope(dem, cell_size)` | Slope in degrees or percent |
| `aspect_degrees(dem, cell_size)` | Aspect in degrees (0-360) |
| `hillshade_compute(dem, cell_size, azimuth, altitude)` | Analytical hillshade |
| `multidirectional_hillshade(dem, cell_size)` | Multi-azimuth hillshade |
| `curvature_compute(dem, cell_size, ctype)` | Profile, plan, or general curvature |
| `advanced_curvature(dem, cell_size, ctype)` | 14 Florinsky curvatures |
| `tpi_compute(dem, cell_size, radius)` | Topographic Position Index |
| `tri_compute(dem, cell_size)` | Terrain Ruggedness Index |
| `dev_compute(dem, cell_size, radius)` | Deviation from Mean Elevation |
| `vrm_compute(dem, cell_size, radius)` | Vector Ruggedness Measure |
| `geomorphons_compute(dem, cell_size, flatness, radius)` | Landform classification |
| `mrvbf_compute(dem, cell_size)` | Multi-resolution Valley Bottom Flatness |
| `northness_compute(dem, cell_size)` | Cosine of aspect |
| `eastness_compute(dem, cell_size)` | Sine of aspect |
| `shape_index_compute(dem, cell_size)` | Shape index |
| `curvedness_compute(dem, cell_size)` | Curvedness |
| `sky_view_factor_compute(dem, cell_size)` | Sky View Factor |
| `viewshed_compute(dem, cell_size, observer_row, observer_col)` | Line-of-sight visibility |
| `openness_positive(dem, cell_size)` | Positive openness |
| `openness_negative(dem, cell_size)` | Negative openness |
| `uncertainty_slope(dem, cell_size, dem_rmse)` | Slope uncertainty from DEM error |

### Hydrology (9 functions)

| Function | Description |
|----------|-------------|
| `fill_depressions(dem, cell_size)` | Planchon-Darboux sink filling |
| `priority_flood_fill(dem, cell_size)` | Priority-Flood (Barnes 2014) |
| `breach_fill(dem, cell_size)` | Breach depressions (Lindsay 2016) |
| `flow_direction_d8(dem, cell_size)` | D8 flow direction |
| `flow_accumulation_d8(fdir, cell_size)` | D8 flow accumulation |
| `flow_accumulation_mfd_compute(dem, cell_size)` | Multi-flow direction accumulation |
| `twi_compute(dem, cell_size)` | Topographic Wetness Index |
| `hand_compute(dem, cell_size)` | Height Above Nearest Drainage |
| `stream_network_compute(dem, cell_size, threshold)` | Stream network extraction |

### Imagery (4 functions)

| Function | Description |
|----------|-------------|
| `ndvi_compute(nir, red)` | Normalized Difference Vegetation Index |
| `ndwi_compute(green, nir)` | Normalized Difference Water Index |
| `savi_compute(nir, red, l_factor)` | Soil-Adjusted Vegetation Index |
| `normalized_diff(a, b)` | Generic normalized difference |

### Morphology (4 functions)

| Function | Description |
|----------|-------------|
| `morph_erode(data, radius)` | Morphological erosion |
| `morph_dilate(data, radius)` | Morphological dilation |
| `morph_opening(data, radius)` | Morphological opening |
| `morph_closing(data, radius)` | Morphological closing |

### Statistics (3 functions)

| Function | Description |
|----------|-------------|
| `focal_mean(data, radius)` | Focal mean |
| `focal_std(data, radius)` | Focal standard deviation |
| `focal_range(data, radius)` | Focal range (max - min) |

## Advanced Curvature Types

The `advanced_curvature` function supports 14 curvature types from Florinsky (2025):

- `mean_h` - Mean curvature H
- `gaussian_k` - Gaussian curvature K
- `unsphericity_m` - Unsphericity M
- `difference_e` - Difference curvature E
- `minimal_kmin` - Minimal principal curvature
- `maximal_kmax` - Maximal principal curvature
- `horizontal_kh` - Horizontal (plan) curvature
- `vertical_kv` - Vertical (profile) curvature
- `horizontal_excess_khe` - Horizontal excess curvature
- `vertical_excess_kve` - Vertical excess curvature
- `accumulation_ka` - Accumulation curvature
- `ring_kr` - Ring curvature
- `rotor` - Rotor (flow-line torsion)
- `laplacian` - Laplacian

## Performance

SurtGIS is 5-10x faster than pure Python implementations thanks to Rust's zero-cost abstractions and automatic parallelization via Rayon.

## License

MIT OR Apache-2.0
