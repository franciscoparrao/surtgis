# SurtGIS Web Demo

Interactive browser-based geospatial analysis powered by **136+ Rust algorithms compiled to WebAssembly**.

🚀 **[Launch Demo](https://franciscoparrao.github.io/surtgis/demo/)** (Soon)

## Features

- ✅ **No installation** - Run in any modern browser
- ✅ **Real geospatial algorithms** - Slope, aspect, hillshade, flow accumulation, morphology, etc.
- ✅ **Fast computation** - 512×512 DEM in <2 seconds
- ✅ **Instant visualization** - Results on Mapbox GL map with colormaps
- ✅ **Download results** - Export computed rasters as GeoTIFF

## Quick Start

### Option A: Use Demo DEM (Fastest)

1. **Open** `index.html` in a web browser
2. **Click** "📂 Load Demo DEM (Andes 512×512)" button
3. **Select** an algorithm (e.g., "Slope")
4. **Click** "Compute" → result appears on map in <1 second
5. **Download** the result as GeoTIFF

### Option B: Upload Your Own Data

1. **Open** `index.html` in a web browser
2. **Upload** your GeoTIFF DEM (Copernicus 30m, USGS 3DEP, SRTM, etc.)
   - Recommended: 512×512 pixels for fast results
   - Max: 100 MB (browser limit)
3. **Select** an algorithm from the dropdown
4. **Click** "Compute"
5. **Download** the result as GeoTIFF

### Demo DEM Details

- **Size**: 512×512 pixels (~445 KB)
- **Location**: Andes Mountains, Chile (-70.5°E, -33.5°S)
- **Elevation**: 357m - 1643m
- **CRS**: WGS84 (EPSG:4326)
- **Ready to use**: No download needed, included in repo

## Algorithms Included

### Terrain Analysis
- **Slope** (degrees or percent gradient)
- **Aspect** (0-360° compass direction)
- **Hillshade** (3D shaded relief)
- **TPI** (Topographic Position Index)
- **TRI** (Terrain Ruggedness Index)

### Hydrology
- **Fill Depressions** (sink filling)
- **Flow Direction** (D8 steepest descent)
- **Flow Accumulation** (upstream area)

### Morphological Operations
- **Erosion** (morphological)
- **Dilation** (morphological)

### Statistics
- **Focal Mean** (neighborhood averaging)

## Example Workflows

### Analyze Watershed Hydrology
1. Upload 30m DEM (Copernicus, USGS 3DEP, etc.)
2. Run "Fill Depressions"
3. Run "Flow Accumulation"
4. Download stream network

### Generate Hillshade Map
1. Upload DEM
2. Select "Hillshade" → adjust azimuth if desired
3. Compute
4. Download for use in GIS

### Terrain Complexity Analysis
1. Upload DEM
2. Run "TRI" (Terrain Ruggedness Index)
3. Visualize rugged vs smooth areas
4. Download classified raster

## Technical Details

- **Frontend**: HTML5 + Vanilla JavaScript + Mapbox GL JS
- **Compute**: SurtGIS algorithms (Rust) compiled to WebAssembly
- **I/O**: GeoTIFF reading/writing in browser
- **Visualization**: Mapbox GL JS with automatic colormapping

## Browser Support

✅ Chrome 57+ | Firefox 79+ | Safari 14.1+ | Edge 79+

**Requirements**: WebAssembly (WASM) support

## Getting Sample DEMs

### Free Global Sources
- **Copernicus DEM 30m**: [ESA Copernicus](https://dem.copernicus.eu/)
- **USGS 3DEP**: [USGS 3DEP Database](https://www.usgs.gov/3dep)
- **SRTM 90m**: [CGIAR-CSI SRTM](https://srtm.csi.cgiar.org/)
- **OpenTopography**: [OpenTopography](https://www.opentopodata.org/)

### From SurtGIS CLI
```bash
# Fetch DEM via STAC
surtgis stac composite pc \
  --collection copernicus-dem-30m \
  --asset elevation \
  --bbox -70.5,-33.6,-70,-33.1 \
  --datetime 2023-01-01/2023-12-31 \
  --outdir dem.tif
```

## Limitations & Future Work

### Current Limitations
- Max 100 MB input (browser memory limit)
- No streaming for very large datasets
- Single-threaded (WASM limitation)

### Future Enhancements (v0.4.0+)
- [ ] Batch processing (multiple files)
- [ ] Real-time parameter adjustment
- [ ] Advanced visualization (3D, animation)
- [ ] NDVI/NDWI for satellite imagery
- [ ] Cloud masking integration
- [ ] Export to GeoJSON, ENVI format

## Performance Benchmarks

| Algorithm | 512×512 | 1024×1024 | 2048×2048 |
|-----------|---------|-----------|-----------|
| Slope | 0.3s | 1.1s | 4.2s |
| Aspect | 0.2s | 0.8s | 3.5s |
| TPI (r=3) | 0.8s | 3.2s | 12s |
| Fill | 1.5s | 6s | 24s |
| Flow Accum | 2s | 8s | 30s |

*On Chrome 120+, M1 MacBook Pro*

## Troubleshooting

### "WASM not ready"
- Wait a few seconds for WASM to load
- Check browser console (F12) for errors
- Ensure JavaScript is enabled

### "Algorithm failed"
- Check file is valid GeoTIFF
- Ensure GeoTIFF has numeric data (float or integer)
- Try with smaller DEM (512×512)
- Check browser console for error details

### Slow computation
- Use smaller DEM (512×512 or 1024×1024)
- Complex algorithms (flow accumulation, TPI) are slower
- Single-threaded WASM means no parallelism

### Download not working
- Check browser's download folder
- Try in incognito/private mode
- Verify JavaScript is enabled

## License

MIT OR Apache-2.0

See [SurtGIS main repo](https://github.com/franciscoparrao/surtgis) for details.

## Links

- 📖 [SurtGIS Documentation](https://github.com/franciscoparrao/surtgis/wiki)
- 💬 [GitHub Discussions](https://github.com/franciscoparrao/surtgis/discussions)
- 🐛 [Report Issues](https://github.com/franciscoparrao/surtgis/issues)
- 📝 [SurtGIS Paper](https://github.com/franciscoparrao/surtgis/blob/main/paper/)

---

**SurtGIS v0.3.0** | Built with ❤️ for geospatial researchers, developers, and teams
