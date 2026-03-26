---
title: 'SurtGIS: A High-Performance Geospatial Analysis Library in Rust with WebAssembly and Python Support'
tags:
  - Rust
  - Python
  - geospatial
  - terrain analysis
  - hydrology
  - WebAssembly
authors:
  - name: Francisco Parra
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Universidad de Santiago de Chile
    index: 1
date: 10 February 2026
bibliography: paper.bib
---

# Summary

SurtGIS is a high-performance geospatial analysis library implemented in Rust, providing over 100 terrain analysis, hydrological modeling, and remote sensing algorithms. The library compiles to native code with automatic parallelization via Rayon, to WebAssembly for browser-based applications, and exposes Python bindings through PyO3. Key contributions include an open-source implementation of Florinsky's complete 14-curvature morphometric system [@florinsky2025], state-of-the-art depression handling algorithms, and four flow direction methods (D8, MFD, D-infinity, TFGA).

Systematic benchmarks on synthetic DEMs up to 20,000 × 20,000 cells (400 million cells) demonstrate that SurtGIS achieves 8.2× speedup over GDAL and 21.7× over GRASS GIS for slope computation, while maintaining numerical accuracy within 5.7 × 10⁻⁵ degrees RMSE against analytical solutions. SurtGIS enables reproducible geospatial workflows that run identically on desktop, server, and browser environments from a single codebase.

# Statement of Need

Geospatial analysis in scientific computing faces a fundamental tension: Python-based tools like rasterio provide accessibility but sacrifice performance, while C/C++ libraries offer speed but require complex compilation and lack browser deployment options. SurtGIS addresses this gap through Rust's combination of memory safety, zero-cost abstractions, and WebAssembly compilation support.

The need for browser-based geospatial processing has grown with the expansion of cloud-native geospatial formats (Cloud-Optimized GeoTIFF, STAC catalogs) and web GIS applications. However, no existing library provides comprehensive terrain and hydrological algorithms that run natively in the browser. SurtGIS fills this gap by compiling the same codebase to both native code (via Rayon for parallelism) and WebAssembly (single-threaded fallback).

Compared to existing tools:

- **GDAL** [@gdal2024]: The de-facto standard lacks many terrain analysis algorithms and cannot run in browsers
- **WhiteboxTools** [@lindsay2016]: Comprehensive algorithms but no WebAssembly support; advanced features require paid extension
- **GRASS GIS** [@neteler2012]: Powerful but heavyweight, designed for desktop/server use only
- **SAGA GIS**: Rich functionality but Windows-centric, no Python/browser deployment

# Key Features

## Terrain Analysis

SurtGIS implements the complete morphometric variable system from Florinsky [@florinsky2025], including:

- **Basic derivatives**: slope (Horn's method [@horn1981]), aspect, northness/eastness
- **14 curvatures**: mean (H), Gaussian (K), unsphericity (M), difference (E), minimal/maximal principal curvatures, horizontal/vertical curvatures, excess curvatures, accumulation (Ka), and ring (Kr) curvatures
- **Landform indices**: TPI, DEV, TRI, VRM, geomorphons [@jasiewicz2013], MRVBF
- **Visibility**: viewshed, sky view factor, positive/negative openness
- **Multiresolution**: Gaussian scale-space, Chebyshev spectral derivatives, 2D-SSA

## Hydrology

- **Depression handling**: Priority-Flood [@barnes2014], breach depressions [@lindsay2016]
- **Flow direction**: D8, Multiple Flow Direction [@quinn1991], D-infinity [@tarboton1997], TFGA
- **Derived indices**: TWI [@kopecky2021], SPI, STI, HAND [@nobre2011]
- **Network extraction**: stream network, watershed delineation

## Architecture

| Crate | Purpose |
|-------|---------|
| `surtgis-core` | Raster data structures, I/O |
| `surtgis-parallel` | Rayon-based parallelization |
| `surtgis-algorithms` | All geospatial algorithms |
| `surtgis-cloud` | STAC client, COG reader |
| `surtgis-wasm` | WebAssembly bindings |
| `surtgis-python` | Python bindings via PyO3 |
| `surtgis-cli` | Command-line interface |

# Performance

Benchmarks comparing SurtGIS against GDAL 3.11, GRASS GIS 8.3, and WhiteboxTools 2.4 on synthetic DEMs up to 20K × 20K cells (400M cells):

| Algorithm | SurtGIS | GDAL | GRASS | WBT |
|-----------|---------|------|-------|-----|
| Slope | **7.51s** | 61.90s (8.2×) | 163.25s (21.7×) | 172.81s (23.0×) |
| Aspect | **7.67s** | 32.12s (4.2×) | 81.77s (10.7×) | 176.58s (23.0×) |
| Hillshade | **12.36s** | 21.04s (1.7×) | 79.09s (6.4×) | 203.25s (16.4×) |
| Fill (PF) | **171.58s** | --- | ERR | T/O |
| D8 flow acc. | **22.89s** | --- | ERR | T/O |

Numerical accuracy: slope RMSE = 5.7 × 10⁻⁵° against analytical solutions (R² = 1.000000).

Hardware: Intel i7-1270P (16 threads), 38.8 GB RAM, Linux 6.14.

# Availability

SurtGIS is available under MIT/Apache-2.0 dual licensing:

- **Source code**: https://github.com/franciscoparrao/surtgis
- **Rust crates**: https://crates.io/crates/surtgis-core
- **Python package**: https://pypi.org/project/surtgis/
- **npm package**: https://www.npmjs.com/package/surtgis

# Acknowledgements

The author thanks the Rust geospatial community for foundational crates including `ndarray`, `geo`, and `proj`.

# References
