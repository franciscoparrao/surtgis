# EMS Submission Metadata

## Full Title

SurtGIS: A High-Performance Geospatial Analysis Library in Rust with WebAssembly and Python Support

## Abstract

We present SurtGIS, an open-source geospatial analysis library in Rust providing 127 terrain, hydrological, and remote sensing algorithms. The library compiles to native code with automatic parallelization via Rayon, to WebAssembly for browser-based applications, and exposes Python bindings through PyO3. Key contributions include Florinsky's complete 14-curvature morphometric system, three depression handling algorithms, and four flow direction methods. Benchmarks on synthetic DEMs up to 20,000² cells demonstrate speedups on full GeoTIFF pipelines: 1.7–1.8x over GDAL and 4.5–4.9x over GRASS GIS for slope; 2.0x over GDAL for aspect; and 7.5–23.1x for D8 flow accumulation. WebAssembly adds 10–32x overhead versus native Rust but enables browser-based terrain analysis unavailable in existing tools. Numerical accuracy is within 5.7 × 10⁻⁵ degrees RMSE against analytical solutions.

## Keywords

terrain analysis; digital elevation model; hydrology; WebAssembly; Rust; geomorphometry
