# SurtGIS Generic STAC Guide

## Overview

As of Phase 4, SurtGIS supports **any STAC v1.0+ API** without code changes or special configuration. The system automatically detects:

- Available spectral bands
- Cloud masking strategy (categorical, bitmask, or none)
- Spatial reference system (CRS)
- Pixel resolution

This guide shows how to use SurtGIS with different STAC catalogs around the world.

## Quick Start

### Planetary Computer (Sentinel-2, Landsat, Sentinel-1)

```bash
surtgis stac composite \
  --catalog pc \
  --collection sentinel-2-l2a \
  --asset red \
  --bbox -70.5,-33.6,-70.2,-33.3 \
  --datetime 2024-06-01/2024-08-31 \
  --outdir red_composite.tif
```

### Earth Search (AWS Open Data)

```bash
surtgis stac composite \
  --catalog es \
  --collection landsat-c2-l2 \
  --asset nir \
  --bbox -70.5,-33.6,-70.2,-33.3 \
  --datetime 2024-06-01/2024-08-31 \
  --outdir nir_composite.tif
```

### Custom STAC API (any provider)

```bash
surtgis stac composite \
  --catalog https://your-stac-api.com/v1 \
  --collection your-collection-name \
  --asset your-band-name \
  --bbox -70.5,-33.6,-70.2,-33.3 \
  --datetime 2024-06-01/2024-08-31 \
  --outdir output.tif
```

## Band Names (Fuzzy Matching)

The system recognizes bands by multiple names. These all refer to the same band:

### Red Band
- Exact: `B04`, `SR_B4`
- Generic: `red`, `banda_roja`, `rouge`

### NIR Band
- Exact: `B08`, `SR_B5`
- Generic: `nir`, `infrared`, `proche_infrarouge`

### Green Band
- Exact: `B03`, `SR_B3`
- Generic: `green`, `banda_verde`

### Blue Band
- Exact: `B02`, `SR_B2`
- Generic: `blue`, `banda_azul`, `coastal`

### SWIR1 Band
- Exact: `B11`, `SR_B6`
- Generic: `swir1`, `swir16`, `mid_infrared`

### SWIR2 Band
- Exact: `B12`, `SR_B7`
- Generic: `swir2`, `swir22`

### Thermal
- Exact: `ST_B10`, `ST_B11`
- Generic: `thermal`, `tirs1`, `tirs2`, `b10`, `lwir`

### SAR (Radar)
- `VV`, `VH`, `HH`, `HV` (polarizations)
- `_amplitude` variants also supported

### Panchromatic
- `pan`, `panchromatic`

## Language Support

Band names work in Spanish, French, and English:

```bash
# Spanish
--asset banda_roja      # red band
--asset banda_verde     # green band
--asset banda_azul      # blue band

# French
--asset rouge           # red band
--asset proche_infrarouge  # NIR band

# English
--asset red
--asset nir
--asset thermal
```

## Cloud Masking (Automatic)

The system detects and applies the appropriate cloud masking:

### Sentinel-2 (SCL - Scene Classification Layer)
- **Valid classes:** Vegetation, not vegetated, water, clouds
- **Masked classes:** Cloud, shadow, snow, etc.
- **Auto-applied:** No configuration needed

### Landsat (QA_PIXEL - Bitmask)
- **Masked bits:** Cloud, cloud shadow, snow
- **Auto-applied:** No configuration needed

### Sentinel-1 SAR (No Cloud Masking)
- **Radar penetrates clouds:** No mask applied
- **All pixels valid:** Unless nodata is explicitly defined

## Examples by Collection

### Sentinel-2 (Planetary Computer)

```bash
# Red band (B04 at 10m resolution)
surtgis stac composite pc --collection sentinel-2-l2a --asset red ...

# NDVI (need B04 and B08)
surtgis stac composite pc --collection sentinel-2-l2a --asset nir ...

# SWIR (true color with B11)
surtgis stac composite pc --collection sentinel-2-l2a --asset swir1 ...
```

### Landsat C2 L2 (Planetary Computer or Earth Search)

```bash
# Red band (SR_B4 at 30m resolution)
surtgis stac composite pc --collection landsat-c2-l2 --asset sr_b4 ...

# Or use fuzzy matching
surtgis stac composite pc --collection landsat-c2-l2 --asset red ...

# Thermal (ST_B10)
surtgis stac composite pc --collection landsat-c2-l2 --asset thermal ...
```

### Sentinel-1 RTC (Planetary Computer)

```bash
# VV polarization (radar)
surtgis stac composite pc --collection sentinel-1-rtc --asset vv ...

# VH polarization (cross-pol)
surtgis stac composite pc --collection sentinel-1-rtc --asset vh ...
```

## Parameters Explained

```bash
surtgis stac composite \
  --catalog pc                                    # STAC catalog (pc, es, or full URL)
  --collection sentinel-2-l2a                     # Collection name in catalog
  --asset red                                     # Band to fetch (fuzzy matched)
  --bbox -70.5,-33.6,-70.2,-33.3                 # West,South,East,North in WGS84
  --datetime 2024-06-01/2024-08-31               # Start/End date range
  --max-scenes 12                                 # Max images to composite (default 12)
  --align-to dem.tif                              # Optional: align output to reference grid
  --outdir red_composite.tif                      # Output file path
```

## Inspection: What Bands Are Available?

The system automatically reports available bands:

```bash
surtgis stac composite pc --collection sentinel-2-l2a --asset b04 ...
```

Output will show:
```
📊 Collection: sentinel-2-l2a
   Available bands: B02 (Blue), B03 (Green), B04 (Red), B08 (Nir), ...
   Cloud masking: Categorical SCL (12 classes)
   Band matched: red → B04
```

## Troubleshooting

### "Band 'X' not found"

The band name wasn't recognized. Try:
1. Use exact asset key: `B04` instead of `red`
2. Check collection's available bands
3. Use a generic band type: `nir` instead of `b8`

### "No items found in bbox"

The collection has no data in that region/date. Try:
- Expand bbox or dates
- Check collection's coverage
- Use different collection (e.g., Landsat if S2 unavailable)

### "Asset not found in first item"

The asset exists in some items but not the first one. This is a STAC issue with that collection.

## Architecture (Technical)

### How Band Detection Works

1. **Search STAC:** Get first item in collection
2. **Introspect item:** Extract assets and metadata
3. **Pattern match:** Detect band types from asset keys
4. **Match user request:** Find best matching band
5. **Auto-detect cloud mask:** Identify strategy from metadata
6. **Create strategy:** Instantiate CloudMaskStrategy trait implementation

### Supported Band Detection Patterns

```
Sentinel-2: B02, B03, B04, B08, B11, B12, ...
Landsat: SR_B1-SR_B7, ST_B10, ST_B11, ...
SAR: VV, VH, HH, HV, _amplitude variants
Thermal: thermal, TIRS1, B10, LWIR, ...
Pan: pan, panchromatic
Multilingual: banda_roja, rouge, proche_infrarouge, ...
```

Total: **30+ patterns** covering major data providers.

### Supported Cloud Masking Strategies

```
CloudMaskType::Categorical { asset: "SCL", num_classes: 12 }     → S2SclMask
CloudMaskType::Bitmask { asset: "QA_PIXEL", bits: [...] }       → LandsatQaMask
CloudMaskType::None                                               → NoCloudMask (SAR)
```

## Future Extensions

The architecture supports adding new:
- **Collections:** No code changes needed (auto-detected)
- **Catalogs:** Add shorthand name → URL mapping
- **Band patterns:** Add strings to detection heuristics
- **Cloud masks:** Extend CloudMaskStrategy trait

## References

- [STAC Specification](https://stacspec.org/)
- [Planetary Computer STAC API](https://planetarycomputer.microsoft.com/docs/overview/about/)
- [Earth Search by Element 84](https://www.element84.com/earth-search/)
- [SurtGIS Repository](https://github.com/franciscoparrao/surtgis)
