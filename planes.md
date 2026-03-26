 Here is Claude's plan:                                                                                                                        
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Script de Benchmarking para Paper EMS                                                                                                   
                                                                                                   
 Contexto

 El paper para Environmental Modelling & Software (paper/paper.tex) define 3 experimentos de rendimiento con tablas vacías pendientes de datos
  reales. Necesitamos un script Python que ejecute los 3 experimentos, genere CSVs con resultados, y reporte info del hardware.

 Herramientas disponibles en el sistema:
 - GDAL 3.11.4 (gdaldem)
 - GRASS GIS 8.3.2 (grass --tmp-project)
 - WhiteboxTools 2.3.6 (Python wrapper whitebox, binario auto-descargable)
 - surtgis: NO instalado como pip — requiere maturin develop previo
 - Hardware: Intel i7-1270P (12 cores/16 threads), 38 GB RAM

 Infraestructura existente:
 - crates/algorithms/examples/bench_comparison.rs — comparación Rust vs GDAL/SAGA/WBT (slope/aspect/hillshade)
 - crates/algorithms/benches/terrain.rs — criterion benchmarks internos
 - tests/fixtures/andes_chile_30m.tif — DEM real 720×720 (Andes, Copernicus 30m)

 Archivos a crear

 benchmarks/
 ├── run_benchmarks.py        # Script principal (~600 LOC)
 └── results/                 # Creado automáticamente
     ├── experiment1_scalability.csv
     ├── experiment2_accuracy.csv
     ├── experiment3_crossplatform.csv
     ├── system_info.json
     └── dems/                # DEMs sintéticos generados

 Diseño: benchmarks/run_benchmarks.py

 Paso 0: Prerrequisitos

 Antes de correr el script:
 cd crates/python && maturin develop --release
 pip install whitebox  # ya instalado

 El script verifica automáticamente qué herramientas están disponibles y salta las que faltan.

 Generación de DEMs sintéticos

 Fractal Brownian Surface (espectral):
 def generate_fbm_dem(size, hurst=0.7, seed=42, cell_size=10.0):
     """FFT-based fractal Brownian surface."""
     rng = np.random.default_rng(seed)
     noise = rng.standard_normal((size, size))
     # Power spectrum: P(f) ~ f^(-(2H+2))
     freqs = np.fft.fftfreq(size)
     fx, fy = np.meshgrid(freqs, freqs)
     power = (fx**2 + fy**2 + 1e-10) ** (-(hurst + 1))
     power[0, 0] = 0
     spectrum = np.fft.fft2(noise) * np.sqrt(power)
     surface = np.real(np.fft.ifft2(spectrum))
     # Normalizar a rango realista (200-2000m)
     surface = 200 + (surface - surface.min()) / (surface.max() - surface.min()) * 1800
     return surface

 Gaussian Hill (analítica, para Exp. 2):
 def generate_gaussian_hill(size, A=500, sigma=None, cell_size=10.0):
     """z = A * exp(-(x²+y²)/(2σ²)), con derivadas cerradas."""
     if sigma is None:
         sigma = size * cell_size * 0.2
     ...
     return dem, analytical_slope, analytical_aspect, analytical_H, analytical_K

 Tamaños: 1000, 5000, 10000, 20000 (el de 20K² = 400M celdas, ~3GB float64 — verificar RAM).

 Ajuste: Para 20K² puede ser demasiado para GRASS en RAM razonable. Evaluar y reportar si algún tool falla.

 Experimento 1: Escalabilidad

 Algoritmos × Herramientas:
 ┌──────────────────┬────────────────────────────────┬───────────────────┬────────────────────────┬─────────────────────────────────┐
 │    Algoritmo     │            SurtGIS             │       GDAL        │         GRASS          │               WBT               │
 ├──────────────────┼────────────────────────────────┼───────────────────┼────────────────────────┼─────────────────────────────────┤
 │ slope            │ surtgis.slope()                │ gdaldem slope     │ r.slope.aspect slope=  │ whitebox.slope()                │
 ├──────────────────┼────────────────────────────────┼───────────────────┼────────────────────────┼─────────────────────────────────┤
 │ aspect           │ surtgis.aspect_degrees()       │ gdaldem aspect    │ r.slope.aspect aspect= │ whitebox.aspect()               │
 ├──────────────────┼────────────────────────────────┼───────────────────┼────────────────────────┼─────────────────────────────────┤
 │ hillshade        │ surtgis.hillshade_compute()    │ gdaldem hillshade │ r.relief.shading       │ whitebox.hillshade()            │
 ├──────────────────┼────────────────────────────────┼───────────────────┼────────────────────────┼─────────────────────────────────┤
 │ TPI r=10         │ surtgis.tpi_compute()          │ gdaldem TPI       │ —                      │ —                               │
 ├──────────────────┼────────────────────────────────┼───────────────────┼────────────────────────┼─────────────────────────────────┤
 │ fill depressions │ surtgis.priority_flood_fill()  │ —                 │ r.fill.dir             │ whitebox.fill_depressions()     │
 ├──────────────────┼────────────────────────────────┼───────────────────┼────────────────────────┼─────────────────────────────────┤
 │ D8 flow acc      │ surtgis.flow_accumulation_d8() │ —                 │ r.watershed            │ whitebox.d8_flow_accumulation() │
 └──────────────────┴────────────────────────────────┴───────────────────┴────────────────────────┴─────────────────────────────────┘
 Metodología:
 - 3 warm-up + 10 repeticiones por combinación
 - Mediana + IQR
 - Cada herramienta se ejecuta sobre GeoTIFF en disco (para GDAL/GRASS/WBT)
 - SurtGIS se ejecuta sobre array numpy en memoria
 - Timeout: 300s por ejecución individual (para evitar que 20K² cuelgue)

 Ejecución GRASS (sin proyecto permanente):
 def run_grass(algorithm, dem_path, output_path):
     cmd = [
         "grass", "--tmp-project", "EPSG:32719", "--exec",
         "bash", "-c",
         f"r.in.gdal input={dem_path} output=dem && "
         f"r.slope.aspect elevation=dem slope=slope && "
         f"r.out.gdal input=slope output={output_path}"
     ]

 Ejecución WBT:
 import whitebox
 wbt = whitebox.WhiteboxTools()
 wbt.slope(dem_path, output_path)

 Output: experiment1_scalability.csv
 algorithm,size,tool,run,time_seconds
 slope,1000,surtgis,1,0.008
 slope,1000,surtgis,2,0.007
 ...
 slope,1000,gdal,1,0.031

 Experimento 2: Precisión numérica

 2a. Validación analítica (Gaussian Hill):
 - DEM: z = A * exp(-(x²+y²) / (2σ²)) con A=500, σ=sizecell0.2
 - Derivadas cerradas:
   - dz/dx = -A * x/σ² * exp(...)
   - dz/dy = -A * y/σ² * exp(...)
   - slope_analytical = arctan(sqrt((dz/dx)² + (dz/dy)²))
   - H = ... (curvatura media cerrada)
   - K = ... (curvatura Gaussiana cerrada)
 - Tamaño: 1000×1000 (suficiente para validación)
 - Métricas: RMSE, MAE, R² (excluyendo borde de 1 celda)
 - Comparar: SurtGIS slope/aspect/H/K vs analítico

 2b. Cross-validación herramientas (DEM real):
 - DEM: tests/fixtures/andes_chile_30m.tif (720×720)
 - Calcular slope con SurtGIS, GDAL, GRASS, WBT
 - Métricas: RMSE entre pares, correlación, % celdas con diferencia < 0.1°
 - Nota: las diferencias se deben a distintos métodos de derivación (Horn vs Evans-Young vs Zevenbergen-Thorne)

 Output: experiment2_accuracy.csv
 metric,algorithm,comparison,value
 rmse,slope,surtgis_vs_analytical,0.023
 mae,slope,surtgis_vs_analytical,0.015
 ...
 rmse,slope,surtgis_vs_gdal,0.45

 Experimento 3: Cross-platform

 Targets:
 1. Rust nativo multi-thread: cargo run --example bench_comparison --release -- --size 5000
 2. Rust nativo single-thread: RAYON_NUM_THREADS=1 cargo run --example bench_comparison --release -- --size 5000
 3. Python bindings: surtgis.slope(dem_5k, 10.0) directo
 4. WebAssembly: placeholder — requiere automatización de browser, se reporta separado

 DEM: 5000×5000 fractal, 10 repeticiones
 Output: experiment3_crossplatform.csv

 Info del sistema

 Capturar automáticamente:
 {
   "cpu": "12th Gen Intel Core i7-1270P",
   "cores": 12,
   "threads": 16,
   "ram_gb": 38,
   "os": "Ubuntu 24.04",
   "gdal_version": "3.11.4",
   "grass_version": "8.3.2",
   "wbt_version": "2.3.6",
   "surtgis_version": "0.1.1",
   "python_version": "3.12",
   "numpy_version": "2.4.2",
   "timestamp": "2026-02-07T..."
 }

 CLI del script

 # Ejecutar todo
 python benchmarks/run_benchmarks.py

 # Solo un experimento
 python benchmarks/run_benchmarks.py --experiment 1
 python benchmarks/run_benchmarks.py --experiment 2
 python benchmarks/run_benchmarks.py --experiment 3

 # Tamaños reducidos (debug rápido)
 python benchmarks/run_benchmarks.py --quick

 # Solo ciertos tools
 python benchmarks/run_benchmarks.py --tools surtgis,gdal

 --quick: usa tamaños 500,1000,2000 con 3 repeticiones (sin warm-up), para verificar que todo funciona antes de la ejecución completa.

 Modificaciones adicionales

 Actualizar bench_comparison.rs

 Agregar flag --single-thread para Experimento 3:
 if args.contains(&"--single-thread".into()) {
     rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
 }

 Agregar los 3 algoritmos faltantes (TPI, fill, flow_acc) al ejemplo existente.

 Instalar surtgis localmente

 Antes de correr benchmarks:
 cd crates/python && maturin develop --release

 Orden de implementación
 ┌─────┬─────────────────────────────────────────────────────────────────────────────────────┬─────────────────────┐
 │  #  │                                        Tarea                                        │      Archivos       │
 ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
 │ 1   │ Crear benchmarks/run_benchmarks.py con estructura, CLI, DEM generation, system info │ run_benchmarks.py   │
 ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
 │ 2   │ Implementar Experimento 1 (scalability)                                             │ run_benchmarks.py   │
 ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
 │ 3   │ Implementar Experimento 2 (accuracy)                                                │ run_benchmarks.py   │
 ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
 │ 4   │ Implementar Experimento 3 (crossplatform)                                           │ run_benchmarks.py   │
 ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
 │ 5   │ Actualizar bench_comparison.rs (--single-thread, más algoritmos)                    │ bench_comparison.rs │
 ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
 │ 6   │ Instalar surtgis + correr --quick para verificar                                    │ —                   │
 └─────┴─────────────────────────────────────────────────────────────────────────────────────┴─────────────────────┘
 Verificación

 # 2. Test rápido
 python benchmarks/run_benchmarks.py --quick

 # 3. Verificar outputs
 ls benchmarks/results/
 cat benchmarks/results/system_info.json
 head benchmarks/results/experiment1_scalability.csv

 # 4. Ejecución completa (puede tomar 30-60 min)Ga
 python benchmarks/run_benchmarks.py

  Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: SurtGIS MCP Server para Gateway

 Contexto

 El mcp-gateway unifica ~1,500 herramientas GIS (OTB, QGIS, GRASS, SAGA, GDAL, etc.) en 8 meta-tools vía MCP. SurtGIS tiene 51 funciones Python (PyO3) que
 cubren terreno, hidrología, imagery, morfología y estadísticas focales. Integrar SurtGIS como backend del gateway le agrega 51 herramientas de alto
 rendimiento (1.8x más rápido que GDAL en slope, 23x en flow accumulation).

 Archivos a crear

 1. /home/franciscoparrao/proyectos/surtgis-mcp-server/surtgis_mcp_server.py

 Servidor MCP principal (~400 líneas). Patrón: mcp.server.Server + stdio.

 Arquitectura:
 - Función build_tools() genera las 51 definiciones Tool con inputSchema JSON
 - Cada tool acepta rutas de archivo (input GeoTIFF → output GeoTIFF)
 - Bridge I/O: rasterio lee → numpy array + metadata → surtgis función → rasterio escribe
 - Helper _read_raster(path) → (data, profile, cell_size)
 - Helper _write_raster(data, profile, path) → escribe GeoTIFF preservando CRS/transform/nodata

 Categorías de tools (51 total):
 - Terrain (20): slope, aspect, hillshade, multidirectional_hillshade, curvature (3 tipos), tpi, tri, geomorphons, northness, eastness, dev, shape_index,
 curvedness, svf, uncertainty_slope, viewshed, openness_pos/neg, mrvbf, vrm, advanced_curvature (14 subtipos)
 - Hydrology (10): fill_depressions, priority_flood, breach, flow_dir_d8, flow_dir_dinf, flow_acc_d8, flow_acc_mfd, twi, hand, stream_network
 - Imagery (4): ndvi, ndwi, savi, normalized_diff
 - Morphology (4): erode, dilate, opening, closing
 - Focal (3): focal_mean, focal_std, focal_range

 2. /home/franciscoparrao/proyectos/surtgis-mcp-server/requirements.txt

 mcp>=1.0.0
 numpy
 rasterio
 surtgis

 3. Setup del venv

 cd /home/franciscoparrao/proyectos/surtgis-mcp-server
 python3 -m venv venv
 source venv/bin/activate
 pip install mcp numpy rasterio
 cd /home/franciscoparrao/proyectos/surtgis
 maturin develop --release  # instala surtgis en el venv del server

 Archivo a modificar

 4. /home/franciscoparrao/proyectos/mcp-gateway/tools_registry.py (~línea 167)

 Agregar entrada en _load_backend_configs() después de "chile":
 "surtgis": {
     "name": "SurtGIS",
     "full_name": "SurtGIS High-Performance Geospatial",
     "server_path": MCP_SERVERS_BASE / "surtgis-mcp-server",
     "run_command": ["venv/bin/python3", "surtgis_mcp_server.py"],
     "description": "Análisis geoespacial de alto rendimiento: terreno, hidrología, índices espectrales, morfología. 51 algoritmos en Rust.",
     "categories": ["terrain", "hydrology", "imagery", "feature_extraction"]
 }

 Diseño del servidor MCP

 Patrón I/O (bridge rasterio ↔ surtgis)

 def _read_raster(path):
     with rasterio.open(path) as src:
         data = src.read(1).astype(np.float64)
         cell_size = abs(src.transform.a)
         return data, src.profile.copy(), cell_size

 def _write_raster(data, profile, path, dtype='float64'):
     profile.update(dtype=rasterio.float64, count=1)
     with rasterio.open(path, 'w', **profile) as dst:
         dst.write(data.astype(np.float64), 1)

 Patrón de tool definition (ejemplo)

 Tool(name="slope",
      description="Compute slope from DEM (Horn method). SurtGIS: 1.8x faster than GDAL.",
      inputSchema={
          "type": "object",
          "properties": {
              "input": {"type": "string", "description": "Input DEM GeoTIFF"},
              "output": {"type": "string", "description": "Output GeoTIFF"},
              "units": {"type": "string", "enum": ["degrees", "percent"], "default": "degrees"}
          },
          "required": ["input", "output"]
      })

 Patrón de ejecución

 if name == "slope":
     data, profile, cs = _read_raster(args["input"])
     result = surtgis.slope(data, cell_size=cs, units=args.get("units", "degrees"))
     _write_raster(result, profile, args["output"])
     return {"success": True, "output": args["output"]}

 Verificación

 1. python3 surtgis_mcp_server.py — debe arrancar sin error
 2. Test JSON-RPC: echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | venv/bin/python3 surtgis_mcp_server.py
 3. Desde gateway: reload_registry → verificar 51 herramientas surtgis.*
 4. Test con DEM real: execute_tool("surtgis.slope", {"input": "dem.tif", "output": "slope.tif"})

  Plan: Preparar paquete de submisión EMS                                                                                                                
                                                                                                                                                        
 Contexto                                                                                                                                               
                                                        
 Paper listo (paper.tex, v4 revision complete, 49 páginas). EMS requiere: cover letter, highlights, manuscript sin autores (double-blind), declaration
 of competing interests. El manuscrito se puede enviar en .tex pero debe incluir todo lo necesario para compilar.

 Archivos que necesita el .tex para compilar

 ┌──────────────────────────┬────────┬──────────────────────────────────────────┐
 │         Archivo          │ Existe │                   Nota                   │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ paper.tex                │ ✓      │ Manuscrito principal                     │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ paper.bib                │ ✓      │ 34 referencias                           │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ elsarticle-harv.bst      │ ✓      │ Estilo Harvard (ya en paper/)            │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ graphical_abstract.pdf   │ ✓      │ En benchmarks/results/figures/           │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ fig1_scalability.pdf     │ ✓      │ "                                        │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ fig2_speedup.pdf         │ ✓      │ "                                        │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ fig3_crossplatform.pdf   │ ✓      │ "                                        │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ fig5_casestudy.pdf       │ ✓      │ "                                        │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ fig_flood_zones.pdf      │ ✓      │ "                                        │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ fig_hydro_validation.pdf │ ✓      │ "                                        │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ elsarticle.cls           │ ✓      │ Sistema (texlive), no hace falta incluir │
 ├──────────────────────────┼────────┼──────────────────────────────────────────┤
 │ competing_interests      │ ❌     │ HAY QUE CREAR                            │
 └──────────────────────────┴────────┴──────────────────────────────────────────┘

 Pasos

 1. Crear paper/submission/ con paquete compilable

 Directorio autocontenido:
 paper/submission/
 ├── paper.tex              # Versión blind (sin author/affiliation)
 ├── paper.bib              # Bibliografía
 ├── elsarticle-harv.bst    # Estilo biblio
 ├── figures/               # 7 figuras referenciadas
 │   ├── graphical_abstract.pdf
 │   ├── fig1_scalability.pdf
 │   ├── fig2_speedup.pdf
 │   ├── fig3_crossplatform.pdf
 │   ├── fig5_casestudy.pdf
 │   ├── fig_flood_zones.pdf
 │   └── fig_hydro_validation.pdf
 ├── cover_letter.pdf       # Ya compilada
 ├── highlights.txt         # 5 bullets texto plano
 ├── competing_interests.tex → .pdf
 └── supplementary_algorithms.pdf

 2. Crear versión double-blind de paper.tex

 En la copia dentro de submission/:
 - Reemplazar bloque \author, \ead, \cortext, \affiliation → \author{[Removed for review]}
 - Cambiar paths ../benchmarks/results/figures/ → figures/
 - Mantener todo lo demás intacto

 3. Crear competing_interests.tex

 Texto estándar Elsevier: "The authors declare that they have no known competing financial interests..."

 4. Crear highlights.txt

 5 bullets extraídos del .tex, sin LaTeX math.

 5. Compilar todo en submission/

 cd paper/submission
 pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
 pdflatex competing_interests

 6. Verificar

 - PDF compila sin errores (49 páginas)
 - PDF NO contiene nombre del autor (grep)
 - Bibliografía correcta (34 refs, Lindsay 2016, Nobre 2011 con DOIs corregidos)
 - Todos los archivos presentes

  Plan: Expandir CLI Imagery + Median Composite + Cloud Masking                                                                                                                       

 Contexto

 Feedback de otra sesión de Claude Code que usó SurtGIS en workflow real de susceptibilidad. El módulo imagery del CLI expone solo 8 de 16+ índices existentes en Rust. Falta lo más
  impactante: el comando calc con expresiones arbitrarias (el parser ya existe como index_builder()), median composite temporal, y cloud masking.

 Descubrimiento clave

 SurtGIS ya tiene en crates/algorithms/src/imagery/:
 - index_builder(formula, bands) — parser de expresiones con +, -, *, /, paréntesis
 - 16 índices espectrales (ndvi, ndwi, mndwi, nbr, savi, evi, evi2, gndvi, ngrdi, reci, ndre, ndsi, ndmi, ndbi, msavi, bsi)
 - reclassify(), raster_difference(), change_vector_analysis()

 Solo 8 están expuestos en CLI. El parser de expresiones no está expuesto.

 Archivos a modificar

 ┌─────────────────────────────────────────────┬───────────┬────────────────────────────────────────────────────┐
 │                   Archivo                   │  Acción   │                     Qué cambia                     │
 ├─────────────────────────────────────────────┼───────────┼────────────────────────────────────────────────────┤
 │ crates/cli/src/main.rs                      │ MODIFICAR │ +14 variantes ImageryCommands, match arms, helpers │
 ├─────────────────────────────────────────────┼───────────┼────────────────────────────────────────────────────┤
 │ crates/algorithms/src/imagery/composite.rs  │ CREAR     │ median_composite() + tests                         │
 ├─────────────────────────────────────────────┼───────────┼────────────────────────────────────────────────────┤
 │ crates/algorithms/src/imagery/cloud_mask.rs │ CREAR     │ cloud_mask_scl() + tests                           │
 ├─────────────────────────────────────────────┼───────────┼────────────────────────────────────────────────────┤
 │ crates/algorithms/src/imagery/mod.rs        │ MODIFICAR │ mod composite; mod cloud_mask; + exports           │
 └─────────────────────────────────────────────┴───────────┴────────────────────────────────────────────────────┘

 Implementación por prioridad

 P1: Comando calc con expresiones (MAYOR IMPACTO)

 surtgis imagery calc \
   --expression "(NIR - Red) / (NIR + Red)" \
   --band NIR=nir.tif --band Red=red.tif \
   output.tif

 - Reusa index_builder() que ya existe en index_builder.rs
 - Helper parse_band_assignments() para parsear NAME=path
 - ~60 líneas de código
 - Con esto se pueden calcular TODOS los índices sin subcomando dedicado

 P2: 9 índices faltantes como subcomandos CLI

 Indices que existen en Rust pero no en CLI: EVI2, GNDVI, NGRDI, RECI, NDRE, NDSI, NDMI, NDBI, MSAVI

 - ~180 líneas (boilerplate, mismo patrón que los existentes)
 - Importar desde surtgis_algorithms::imagery

 P3: Reclassify + Change Detection

 - surtgis imagery reclassify --class "min,max,value" ... output.tif
 - surtgis imagery change-diff --before t1.tif --after t2.tif output.tif
 - surtgis imagery cva --band1-before ... --band2-before ... output.tif
 - ~130 líneas, funciones ya existen

 P4: Median Composite (algoritmo nuevo)

 surtgis imagery median-composite \
   --input scene1.tif --input scene2.tif --input scene3.tif \
   output.tif

 Nuevo archivo composite.rs:
 - Per-pixel median across N rasters (temporal composite)
 - Ignora NaN/nodata (nube ya enmascarada)
 - Patrón: validar dimensiones → parallel por fila → sort + mediana
 - ~100 líneas Rust + tests

 P5: Cloud Masking SCL

 surtgis imagery cloud-mask \
   --input scene.tif --scl scl.tif \
   --keep 4,5,6,11 \
   output.tif

 Nuevo archivo cloud_mask.rs:
 - SCL classes: 4=vegetación, 5=suelo, 6=agua, 11=nieve (defaults válidos)
 - Pixels fuera de --keep → NaN
 - ~80 líneas Rust + tests

 Diseño: índices exóticos via calc

 Los índices solicitados que NO merecen subcomando dedicado (se resuelven con calc):
 - EXG: "2 * Green - Red - Blue"
 - VARI: "(Green - Red) / (Green + Red - Blue)"
 - RGRI: "Red / Green"
 - Clay: "SWIR1 / SWIR2"
 - Iron oxide: "Red / Blue"
 - EBBI/NDISI/UI: expresiones similares

 Documentar en --help del comando calc como ejemplos.

 Verificación

 # P1: calc con expresión
 surtgis imagery calc --expression "(NIR-Red)/(NIR+Red)" \
   --band NIR=nir.tif --band Red=red.tif ndvi_calc.tif

 # P2: índice individual
 surtgis imagery msavi --nir nir.tif --red red.tif msavi.tif

 # P4: composite
 surtgis imagery median-composite --input s1.tif --input s2.tif --input s3.tif composite.tif

 # P5: cloud mask
 surtgis imagery cloud-mask --input band.tif --scl scl.tif --keep 4,5,6 masked.tif

 # Compilación
 cargo build --release -p surtgis
 cargo test -p surtgis-algorithms -- imagery

 Estimación total

 ~600 líneas de código, 2 archivos nuevos, 2 modificados.
 CLI imagery: 8 → 22 subcomandos.

  Plan: Multi-tile STAC fetch con mosaic automático      

 Contexto

 surtgis stac fetch baja un solo item STAC. Para cuencas grandes (ej: Copiapó, 18,000 km²) que abarcan múltiples tiles satelitales, se necesita descargar todos los tiles y
 fusionarlos. search_all() ya maneja paginación, pero no existe función de mosaic de rasters.

 Qué existe

 - StacClientBlocking::search_all() — búsqueda paginada, hasta 100 items
 - CogReaderBlocking::read_bbox() — lee bbox de un solo COG
 - BBox::intersects() — test de intersección
 - Reproyección WGS84↔UTM pure-Rust
 - stac_search_and_read() — busca + lee 1 item

 Qué falta

 - Función mosaic() que combine N rasters en uno cubriendo la extensión unión
 - Comando CLI stac fetch-mosaic
 - Comando CLI mosaic (para archivos locales)

 Archivos a modificar/crear

 ┌───────────────────────────┬───────────┬──────────────────────────────────────────────┐
 │          Archivo          │  Acción   │                  Qué cambia                  │
 ├───────────────────────────┼───────────┼──────────────────────────────────────────────┤
 │ crates/core/src/mosaic.rs │ CREAR     │ mosaic() + MosaicOptions + tests             │
 ├───────────────────────────┼───────────┼──────────────────────────────────────────────┤
 │ crates/core/src/lib.rs    │ MODIFICAR │ pub mod mosaic; + re-exports                 │
 ├───────────────────────────┼───────────┼──────────────────────────────────────────────┤
 │ crates/cli/src/main.rs    │ MODIFICAR │ Commands::Mosaic + StacCommands::FetchMosaic │
 └───────────────────────────┴───────────┴──────────────────────────────────────────────┘

 Implementación

 1. mosaic() en crates/core/src/mosaic.rs

 pub struct MosaicOptions {
     pub cell_size_tolerance: f64,   // default 0.01 (1%)
     pub require_same_crs: bool,     // default true
 }

 pub fn mosaic<T: RasterElement>(
     tiles: &[&Raster<T>],
     options: Option<MosaicOptions>,
 ) -> Result<Raster<T>>

 Algoritmo:
 1. Validar: no vacío, todos north-up, CRS compatibles, cell sizes compatibles
 2. Computar union bbox: min/max de todos los tile.bounds()
 3. Crear output raster de dimensiones (union_rows, union_cols) lleno de nodata
 4. Para cada tile: calcular offset en pixeles y copiar datos
 5. Overlap: NaN-aware last-write-wins (NaN/nodata no sobreescribe datos válidos)

 Offset de cada tile en el output:
 col_offset = round((tile_origin_x - output_origin_x) / pixel_width)
 row_offset = round((output_origin_y - tile_origin_y) / |pixel_height|)

 2. CLI: Commands::Mosaic (archivos locales)

 surtgis mosaic -i tile1.tif -i tile2.tif -i tile3.tif output.tif

 - No depende del feature cloud
 - Lee cada archivo, llama mosaic(), escribe resultado

 3. CLI: StacCommands::FetchMosaic

 surtgis stac fetch-mosaic \
   --catalog es \
   --bbox -71.5,-28.0,-69.0,-26.0 \
   --collection cop-dem-glo-30 \
   --asset data \
   --datetime 2024-01-01/2024-12-31 \
   --max-items 20 \
   output.tif

 Flujo:
 1. search_all() con max_items
 2. Para cada item: sign href → open COG → read bbox (auto-reproject)
 3. Reportar progreso: "Fetching tile N of M..."
 4. Skip tiles que no intersectan (log warning)
 5. mosaic() de todos los rasters
 6. Write output

 Se implementa inline en el handler CLI (no como función de librería) para tener UX de progreso por tile.

 4. Exportar en crates/core/src/lib.rs

 pub mod mosaic;
 pub use mosaic::{mosaic, MosaicOptions};

 Tests para mosaic.rs

 1. 2 tiles adyacentes — 100x100 + 100x100 → 100x200
 2. 2 tiles superpuestos — overlap de 50 cols, last-write-wins
 3. NaN no sobreescribe válido — tile 2 tiene NaN en zona overlap, tile 1 se preserva
 4. 1 solo tile — retorna copia
 5. 0 tiles — error
 6. Cell size incompatible — error
 7. 4 tiles en grid 2x2 — verificar assembly correcto

 Verificación

 # Mosaic local
 surtgis mosaic -i tile1.tif -i tile2.tif output.tif

 # STAC fetch-mosaic (Copernicus DEM, Earth Search)
 surtgis stac fetch-mosaic \
   --catalog es \
   --bbox -70.5,-27.5,-70.0,-27.0 \
   --collection cop-dem-glo-30 \
   --asset data \
   --max-items 10 \
   dem_mosaic.tif

 # Verificar con info
 surtgis info dem_mosaic.tif

 # Build + tests
 cargo build --release -p surtgis
 cargo test -p surtgis-core -- mosaic

 Limitaciones (documentadas, no abordadas)

 - Requiere mismo CRS en todos los tiles (no reproyecta entre zonas UTM)
 - Todos los tiles en memoria simultáneamente
 - Fetching secuencial de COGs (cada uno usa concurrencia interna)

  Plan: stac composite — Pipeline S2 end-to-end nativo   

 Contexto

 fetch-mosaic mezcla fechas — descarga N items y los une espacialmente sin distinguir temporalidad. Para Sentinel-2, el flujo correcto es:
 1. Agrupar items por fecha de adquisición
 2. Mosaic espacial por fecha (unir tiles de la misma pasada)
 3. Cloud-mask cada escena con SCL
 4. Median composite temporal entre escenas limpias

 Ninguna herramienta CLI ofrece esto end-to-end sin Python.

 Comando propuesto

 surtgis stac composite \
   --catalog es \
   --bbox -71.5,-28.0,-69.0,-26.0 \
   --collection sentinel-2-l2a \
   --asset red \
   --datetime 2023-01-01/2023-12-31 \
   --max-scenes 12 \
   --scl-asset scl \
   --scl-keep 4,5,6,11 \
   output.tif

 Algoritmo interno

 search_all(bbox, collection, datetime, limit=max_scenes*4)
   │
   ├── Agrupar items por fecha (truncar datetime a YYYY-MM-DD)
   │
   ├── Por cada fecha:
   │   ├── Para cada item del grupo:
   │   │   ├── Fetch asset principal (e.g. "red") → raster
   │   │   └── Fetch SCL asset → scl_raster
   │   ├── Mosaic espacial de los rasters de datos
   │   ├── Mosaic espacial de los rasters SCL
   │   └── cloud_mask_scl(mosaic_data, mosaic_scl, keep_classes)
   │       → escena_limpia
   │
   └── median_composite(escenas_limpias)
       → output final

 Archivos a modificar

 ┌────────────────────────┬───────────┬─────────────────────────────────────────────────┐
 │        Archivo         │  Acción   │                   Qué cambia                    │
 ├────────────────────────┼───────────┼─────────────────────────────────────────────────┤
 │ crates/cli/src/main.rs │ MODIFICAR │ StacCommands::Composite + handler (~180 líneas) │
 └────────────────────────┴───────────┴─────────────────────────────────────────────────┘

 No se necesitan nuevos archivos Rust — todo reutiliza funciones existentes:
 - surtgis_core::mosaic() — mosaic espacial
 - surtgis_algorithms::imagery::cloud_mask_scl() — enmascarado SCL
 - surtgis_algorithms::imagery::median_composite() — composite temporal
 - StacClientBlocking::search_all() — búsqueda paginada
 - CogReaderBlocking::read_bbox() — lectura COG
 - parse_bbox(), parse_scl_classes() — helpers existentes

 Diseño del handler

 1. Parsear argumentos y buscar

 StacCommands::Composite {
     catalog, bbox, collection, asset, datetime,
     max_scenes, scl_asset, scl_keep, output,
 } => {
     // search_all con limit alto (max_scenes * 4 tiles por fecha)
     let items = client.search_all(&params)?;
 }

 2. Agrupar por fecha

 let mut by_date: BTreeMap<String, Vec<&StacItem>> = BTreeMap::new();
 for item in &items {
     let date = item.properties.datetime
         .as_deref()
         .unwrap_or("")
         .get(..10)  // "2023-06-15T10:56:21Z" → "2023-06-15"
         .unwrap_or("")
         .to_string();
     by_date.entry(date).or_default().push(item);
 }
 // Limitar a max_scenes fechas
 let dates: Vec<_> = by_date.keys().take(max_scenes).cloned().collect();

 3. Por cada fecha: fetch + mosaic + cloud-mask

 let mut clean_scenes: Vec<Raster<f64>> = Vec::new();

 for date in &dates {
     let group = &by_date[date];
     let mut data_tiles = Vec::new();
     let mut scl_tiles = Vec::new();

     for item in group {
         // Fetch data asset
         let data_raster = fetch_item_asset(item, &asset_key, &bb, &client)?;
         data_tiles.push(data_raster);

         // Fetch SCL asset
         let scl_raster = fetch_item_asset(item, &scl_asset, &bb, &client)?;
         scl_tiles.push(scl_raster);
     }

     // Mosaic spatial tiles for this date
     let data_mosaic = mosaic(&data_refs, None)?;
     let scl_mosaic = mosaic(&scl_refs, None)?;

     // Cloud mask
     let clean = cloud_mask_scl(&data_mosaic, &scl_mosaic, &keep_classes)?;
     clean_scenes.push(clean);
 }

 4. Median composite temporal

 let refs: Vec<&Raster<f64>> = clean_scenes.iter().collect();
 let result = median_composite(&refs)?;
 write_result(&result, &output, compress)?;

 5. Helper fetch_item_asset

 Extraer la lógica repetida de fetch-mosaic en una función helper para reusar:

 fn fetch_item_asset(
     item: &StacItem,
     asset_key: &str,
     bbox: &BBox,
     client: &StacClientBlocking,
 ) -> Result<Raster<f64>>

 Contiene: asset lookup → sign href → open COG → reproject bbox → read_bbox.

 Argumentos CLI

 /// End-to-end satellite composite: search, mosaic per date, cloud-mask, median composite
 Composite {
     #[arg(long, default_value = "es")]
     catalog: String,
     #[arg(long)]
     bbox: String,
     #[arg(long)]
     collection: String,
     /// Data asset to composite (e.g. "red", "nir", "B04")
     #[arg(long)]
     asset: String,
     #[arg(long)]
     datetime: String,
     /// Maximum number of temporal scenes to composite
     #[arg(long, default_value = "12")]
     max_scenes: usize,
     /// SCL asset key for cloud masking (default: "scl")
     #[arg(long, default_value = "scl")]
     scl_asset: String,
     /// SCL classes to keep (default: vegetation, bare soil, water, snow)
     #[arg(long, default_value = "4,5,6,11")]
     scl_keep: String,
     /// Output GeoTIFF file
     output: PathBuf,
 },

 UX: Progreso

 Searching STAC catalog...
 Found 48 items across 12 dates
 Processing date 1/12: 2023-01-15 (4 tiles)...
   Fetching red tiles... done (4 tiles)
   Fetching scl tiles... done (4 tiles)
   Mosaic + cloud mask... done (2450 x 3200, 78% clear)
 Processing date 2/12: 2023-02-14 (4 tiles)...
   ...
 Computing median composite of 12 scenes...
 Composite saved to: output.tif
   12 scenes, 2450 x 3200 (7.8M cells), processing time: 45.2s

 Verificación

 # Composite de banda red de S2 sobre zona pequeña
 surtgis stac composite \
   --catalog es \
   --bbox -70.3,-27.2,-70.2,-27.1 \
   --collection sentinel-2-l2a \
   --asset red \
   --datetime 2024-06-01/2024-08-31 \
   --max-scenes 5 \
   test_composite.tif

 surtgis info test_composite.tif

 # Build
 cargo build --release -p surtgis

 Limitaciones

 - Todos los tiles y escenas en memoria simultáneamente
 - SCL debe tener mismas dimensiones que el asset de datos (S2 L2A: SCL es 20m, red es 10m → necesitan resolución compatible). Para S2 L2A en Earth Search, SCL está a 20m y las
 bandas 10m. Solución: usar overview level en SCL para coincidir, o documentar que ambos assets deben ser del mismo resolution group.
 - Requiere mismo CRS en todos los tiles (normal para S2 dentro de una zona UTM)

  Plan: Landscape metrics globales para SurtGIS                                                                                                                                       
  
 Contexto                                                                                                                                                                            
                                                        
 El módulo landscape de SurtGIS tiene shannon_diversity, simpson_diversity y patch_density, pero todos operan como moving window (output = raster). Para análisis de humedales y
 ecología del paisaje se necesitan métricas globales (output = un escalar por parche o por paisaje). Faltan: connected components, PARA, FRAC, COHESION, AI, y versiones globales de
  SHDI/SIDI.

 Qué existe y se reutiliza

 - patch_density() en diversity.rs tiene flood-fill 4-connected por ventana — la lógica de BFS se puede adaptar
 - zonal_statistics() en statistics/zonal.rs usa Raster<i32> como zonas — mismo patrón para métricas por zona
 - Raster<f64> con .round() as i64 para clases — patrón ya usado en diversity.rs

 Archivos a crear/modificar

 ┌─────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
 │                         Archivo                         │                  Acción                   │
 ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
 │ crates/algorithms/src/landscape/connected_components.rs │ CREAR — label_patches()                   │
 ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
 │ crates/algorithms/src/landscape/patch_metrics.rs        │ CREAR — PARA, FRAC, perimeter, area       │
 ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
 │ crates/algorithms/src/landscape/class_metrics.rs        │ CREAR — AI, COHESION, SHDI/SIDI global    │
 ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
 │ crates/algorithms/src/landscape/mod.rs                  │ MODIFICAR — agregar módulos + exports     │
 ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
 │ crates/cli/src/main.rs                                  │ MODIFICAR — agregar subcomandos landscape │
 └─────────────────────────────────────────────────────────┴───────────────────────────────────────────┘

 Implementación

 1. Connected Components (connected_components.rs)

 Building block para todo lo demás.

 /// Label connected patches in a classification raster.
 /// Returns (labeled raster, number of patches).
 pub fn label_patches(
     classification: &Raster<f64>,
     connectivity: Connectivity,
 ) -> Result<(Raster<i32>, usize)>

 pub enum Connectivity {
     Four,   // cardinal neighbors
     Eight,  // cardinal + diagonal
 }

 Algoritmo: Union-Find (disjoint set) en dos pasadas:
 1. Primera pasada: asignar labels provisionales, registrar equivalencias
 2. Segunda pasada: resolver equivalencias, relabeling

 Union-Find es O(n·α(n)) ≈ O(n), más eficiente que BFS por parche.

 Output: Raster<i32> donde cada pixel tiene el ID de su parche (1..N), 0 = nodata.

 2. Patch Metrics (patch_metrics.rs)

 Métricas por parche individual, computadas desde el label raster.

 /// Per-patch statistics
 pub struct PatchStats {
     pub label: i32,
     pub class: i64,        // class value from original classification
     pub area_cells: usize, // number of cells
     pub area_m2: f64,      // area in map units
     pub perimeter_cells: usize, // edge cells (adjacent to different class or border)
     pub perimeter_m: f64,  // perimeter in map units
     pub para: f64,         // Perimeter-Area Ratio = perimeter / area
     pub frac: f64,         // Fractal Dimension = 2 * ln(0.25 * perimeter) / ln(area)
 }

 /// Compute per-patch metrics for all patches
 pub fn patch_metrics(
     classification: &Raster<f64>,
     labels: &Raster<i32>,
     num_patches: usize,
 ) -> Result<Vec<PatchStats>>

 Algoritmo en una sola pasada sobre el raster:
 - Para cada pixel: incrementar area del patch, checkear 4 vecinos para perimeter
 - Perimeter: un pixel contribuye al perímetro si algún vecino tiene clase diferente o es borde

 PARA = perimeter / area
 FRAC = 2 * ln(0.25 * perimeter) / ln(area)  (para area > 1)

 3. Class/Landscape Metrics (class_metrics.rs)

 Métricas agregadas por clase o para todo el paisaje.

 /// Landscape-level metrics
 pub struct LandscapeMetrics {
     pub shdi: f64,        // Shannon Diversity Index (global)
     pub sidi: f64,        // Simpson Diversity Index (global)
     pub num_patches: usize,
     pub num_classes: usize,
     pub total_area_m2: f64,
 }

 /// Per-class metrics
 pub struct ClassMetrics {
     pub class: i64,
     pub area_m2: f64,
     pub proportion: f64,     // proportion of landscape
     pub num_patches: usize,
     pub mean_patch_area: f64,
     pub ai: f64,             // Aggregation Index
     pub cohesion: f64,       // Patch Cohesion Index
 }

 /// Compute global landscape metrics
 pub fn landscape_metrics(
     classification: &Raster<f64>,
 ) -> Result<LandscapeMetrics>

 /// Compute per-class metrics
 pub fn class_metrics(
     classification: &Raster<f64>,
     labels: &Raster<i32>,
     patches: &[PatchStats],
 ) -> Result<Vec<ClassMetrics>>

 SHDI = -Σ(p_i * ln(p_i)) donde p_i = proporción de clase i
 SIDI = 1 - Σ(p_i²)
 AI por clase = (g_ii / max_g_ii) * 100, donde g_ii = adyacencias same-class
 COHESION = [1 - Σ(p_ij) / Σ(p_ij * √a_ij)] * [1 - 1/√A] * 100

 4. CLI subcomandos

 Nuevo grupo top-level landscape:

 # Connected components
 surtgis landscape label-patches classification.tif labels.tif --connectivity 8

 # Per-patch metrics (CSV output)
 surtgis landscape patch-metrics classification.tif --output patches.csv

 # Class metrics (table output)
 surtgis landscape class-metrics classification.tif

 # Global landscape metrics (single line)
 surtgis landscape landscape-metrics classification.tif

 # All-in-one: label + metrics + CSV
 surtgis landscape analyze classification.tif --output-labels labels.tif --output-csv metrics.csv

 5. Module exports

 En landscape/mod.rs:
 mod connected_components;
 mod class_metrics;
 mod patch_metrics;

 pub use connected_components::{label_patches, Connectivity};
 pub use patch_metrics::{patch_metrics, PatchStats};
 pub use class_metrics::{landscape_metrics, class_metrics, LandscapeMetrics, ClassMetrics};

 Tests

 connected_components.rs

 1. Checkerboard — cada pixel es su propio parche (4-connected)
 2. Checkerboard 8-connected — todo un solo parche
 3. 3 blobs separados — 3 labels distintos
 4. Con nodata — NaN pixels no se labelan
 5. Single class — todo un solo parche

 patch_metrics.rs

 1. Cuadrado 10x10 — area=100, perimeter=36, PARA conocido
 2. Rectángulo 5x20 — perimeter/area diferente
 3. Parche de 1 pixel — edge case FRAC

 class_metrics.rs

 1. 50/50 dos clases — SHDI = ln(2) ≈ 0.693
 2. Una sola clase — SHDI = 0
 3. AI para clase compacta — AI alto (~95+)
 4. AI para clase dispersa — AI bajo

 Verificación

 # Generar clasificación de prueba
 surtgis terrain geomorphons benchmarks/results/dems/fbm_1000_raw.tif /tmp/geomorph.tif

 # Label patches
 surtgis landscape label-patches /tmp/geomorph.tif /tmp/labels.tif

 # Compute metrics
 surtgis landscape analyze /tmp/geomorph.tif --output-csv /tmp/metrics.csv

 # Build + test
 cargo build --release -p surtgis
 cargo test -p surtgis-algorithms -- landscape

 Estimación

 ~400 líneas Rust (3 archivos nuevos) + ~100 líneas CLI + ~150 líneas tests = ~650 total

  Plan: Streaming I/O para SurtGIS (feature/streaming-io)                                                                                                                             
   
 Contexto                                                                                                                                                                            
                                                        
 SurtGIS carga rasters completos en memoria (Array2<f64>). Un DEM 10K² = 800MB, 20K² = 3.2GB, 50K² = 20GB. El stac composite con 10 escenas de 14430×10403 usa 5.6GB. Esto es el
 blocker #1 para producción.

 El tiff crate 0.10 ya soporta:
 - Lectura por strips: decoder.read_chunk(strip_index) — lee un strip individual
 - Escritura por strips: image.write_strip(&data) — escribe secuencialmente
 - NO soporta: tiled writing (solo strips)

 Alcance: Nivel 1 — Strip-based processing

 Cubrir los ~20 algoritmos de ventana móvil que dominan el uso real:
 slope, aspect, hillshade, curvature, TPI, TRI, northness, eastness, geomorphons, openness±, SVF, DEV, VRM, convergence, advanced curvatures, multi-hillshade.

 Estos solo necesitan ver 2*radius + chunk_rows filas a la vez. Para kernel 3×3, solo 3 filas en buffer.

 Los algoritmos globales (fill, flow direction, flow accumulation, watershed) siguen in-memory.

 Archivos a crear/modificar

 ┌────────────────────────────────────────────┬─────────────────────────────────────────────────────┐
 │                  Archivo                   │                       Acción                        │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/core/src/io/strip_reader.rs         │ CREAR — StripReader que lee strips de un GeoTIFF    │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/core/src/io/strip_writer.rs         │ CREAR — StripWriter que escribe strips              │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/core/src/streaming.rs               │ CREAR — StripProcessor + WindowAlgorithm trait      │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/core/src/io/mod.rs                  │ MODIFICAR — exportar nuevos módulos                 │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/core/src/lib.rs                     │ MODIFICAR — exportar streaming                      │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/algorithms/src/terrain/slope.rs     │ MODIFICAR — impl WindowAlgorithm                    │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/algorithms/src/terrain/aspect.rs    │ MODIFICAR — impl WindowAlgorithm                    │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/algorithms/src/terrain/hillshade.rs │ MODIFICAR — impl WindowAlgorithm                    │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ crates/cli/src/main.rs                     │ MODIFICAR — auto-detect streaming para DEMs grandes │
 └────────────────────────────────────────────┴─────────────────────────────────────────────────────┘

 Implementación

 1. StripReader (crates/core/src/io/strip_reader.rs)

 pub struct StripReader {
     decoder: Decoder<BufReader<File>>,
     rows: usize,
     cols: usize,
     rows_per_strip: usize,
     strip_count: usize,
     transform: GeoTransform,
     crs: Option<CRS>,
     nodata: Option<f64>,
 }

 impl StripReader {
     pub fn open(path: &Path) -> Result<Self>
     pub fn rows(&self) -> usize
     pub fn cols(&self) -> usize
     pub fn strip_count(&self) -> usize
     pub fn rows_in_strip(&self, strip_idx: usize) -> usize
     pub fn transform(&self) -> &GeoTransform
     pub fn crs(&self) -> Option<&CRS>

     /// Read a single strip into an Array2<f64>
     pub fn read_strip(&mut self, strip_idx: usize) -> Result<Array2<f64>>

     /// Read a range of rows (may span multiple strips)
     pub fn read_rows(&mut self, start_row: usize, count: usize) -> Result<Array2<f64>>
 }

 Internamente usa decoder.read_chunk(strip_idx) y convierte a f64.

 2. StripWriter (crates/core/src/io/strip_writer.rs)

 pub struct StripWriter {
     encoder: TiffEncoder<BufWriter<File>>,
     image: ImageEncoder<...>,
     rows_per_strip: u32,
     cols: usize,
     total_rows: usize,
     rows_written: usize,
 }

 impl StripWriter {
     pub fn create(
         path: &Path,
         rows: usize,
         cols: usize,
         transform: GeoTransform,
         crs: Option<&CRS>,
         compress: bool,
     ) -> Result<Self>

     /// Write a strip of rows (f64 → f32 conversion)
     pub fn write_rows(&mut self, data: &Array2<f64>) -> Result<()>

     /// Finalize and close the file
     pub fn finish(self) -> Result<()>
 }

 Internamente configura rows_per_strip, escribe GeoTIFF tags, y usa image.write_strip().

 3. WindowAlgorithm trait + StripProcessor (crates/core/src/streaming.rs)

 /// Trait para algoritmos que operan con una ventana móvil.
 pub trait WindowAlgorithm: Send + Sync {
     /// Radio del kernel (1 para 3×3, 10 para 21×21)
     fn kernel_radius(&self) -> usize;

     /// Procesar un bloque de filas del input, producir filas de output.
     /// `input` tiene (2*radius + chunk_rows) filas × cols columnas.
     /// `output` debe tener chunk_rows filas × cols columnas.
     fn process_chunk(
         &self,
         input: &Array2<f64>,
         output: &mut Array2<f64>,
         nodata: Option<f64>,
         cell_size_x: f64,
         cell_size_y: f64,
     );
 }

 /// Procesador streaming que lee → computa → escribe en chunks.
 pub struct StripProcessor {
     chunk_rows: usize, // filas por chunk (default: 256)
 }

 impl StripProcessor {
     pub fn new(chunk_rows: usize) -> Self

     /// Procesar un archivo completo con streaming.
     pub fn process<A: WindowAlgorithm>(
         &self,
         input_path: &Path,
         output_path: &Path,
         algorithm: &A,
         compress: bool,
     ) -> Result<()>
 }

 El process() hace:
 1. Abrir StripReader(input)
 2. Crear StripWriter(output, same dims/transform/crs)
 3. radius = algorithm.kernel_radius()
 4. Para cada chunk de filas:
    a. Leer (chunk_rows + 2*radius) filas del input (con overlap)
    b. Crear buffer output de chunk_rows filas
    c. algorithm.process_chunk(input_buffer, output_buffer, ...)
    d. writer.write_rows(output_buffer)
 5. writer.finish()

 Buffer de memoria: (chunk_rows + 2*radius) × cols × 8 bytes para input + chunk_rows × cols × 8 para output.
 Para chunk_rows=256, cols=50000, radius=1: (258 + 256) × 50000 × 8 = ~200MB. Manejable.

 4. Implementar WindowAlgorithm para slope

 // En crates/algorithms/src/terrain/slope.rs
 impl WindowAlgorithm for SlopeStreaming {
     fn kernel_radius(&self) -> usize { 1 } // 3×3 kernel

     fn process_chunk(&self, input: &Array2<f64>, output: &mut Array2<f64>,
                      nodata: Option<f64>, dx: f64, dy: f64) {
         let radius = 1;
         let (in_rows, cols) = input.dim();
         let out_rows = in_rows - 2 * radius;
         for r in 0..out_rows {
             for c in 0..cols {
                 // Horn 3×3 kernel centrado en (r+radius, c)
                 // ... misma lógica que slope actual pero sobre el buffer
             }
         }
     }
 }

 5. CLI: auto-detect o flag --streaming

 // En el handler de slope:
 let file_size = std::fs::metadata(&input)?.len();
 if file_size > 500_000_000 { // >500MB → streaming
     let algo = SlopeStreaming { units, z_factor };
     let processor = StripProcessor::new(256);
     processor.process(&input, &output, &algo, compress)?;
 } else {
     // In-memory (actual, más rápido para DEMs pequeños)
     let dem = read_dem(&input)?;
     let result = slope(&dem, params)?;
     write_result(&result, &output, compress)?;
 }

 Orden de implementación

 ┌─────┬────────────────────────────────────────┬──────────────────────────┐
 │  #  │                 Tarea                  │       Dependencia        │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 1   │ StripReader + tests                    │ Ninguna                  │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 2   │ StripWriter + tests                    │ Ninguna (paralelo con 1) │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 3   │ WindowAlgorithm trait + StripProcessor │ 1 + 2                    │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 4   │ Slope streaming + test end-to-end      │ 3                        │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 5   │ Aspect, hillshade streaming            │ 4 (mismo patrón)         │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 6   │ CLI auto-detect                        │ 4                        │
 ├─────┼────────────────────────────────────────┼──────────────────────────┤
 │ 7   │ Resto de algoritmos de ventana         │ 5                        │
 └─────┴────────────────────────────────────────┴──────────────────────────┘

 Tests

 StripReader

 1. Leer strips de un GeoTIFF existente (fbm_1000_raw.tif)
 2. read_rows() que cruza boundary de strips
 3. Verificar que metadatos (transform, CRS) se preservan

 StripWriter

 1. Escribir strips y re-leer con el reader estándar
 2. Verificar que el GeoTIFF resultante es válido (rasterio/GDAL lo abre)
 3. Compresión DEFLATE en modo strip

 StripProcessor + Slope

 1. Comparar resultado streaming vs in-memory en fbm_1000_raw.tif
 2. Deben ser idénticos bit-a-bit (mismo algoritmo Horn)
 3. Test con DEM sintético donde el resultado analítico es conocido

 Verificación

 git checkout feature/streaming-io
 git merge main  # traer todos los fixes de esta sesión

 # Build + test
 cargo test -p surtgis-core -- strip_reader strip_writer
 cargo test -p surtgis-algorithms -- streaming

 # Comparación end-to-end
 ./target/release/surtgis terrain slope --streaming benchmarks/results/dems/fbm_1000_raw.tif /tmp/slope_stream.tif
 ./target/release/surtgis terrain slope benchmarks/results/dems/fbm_1000_raw.tif /tmp/slope_mem.tif
 # Deben ser idénticos
 python3 -c "import rasterio; a=rasterio.open('/tmp/slope_stream.tif').read(1); b=rasterio.open('/tmp/slope_mem.tif').read(1); print('Match:', (a==b).all())"

 Qué NO incluye este plan

 - Tiled GeoTIFF writing (tiff crate no lo soporta)
 - Streaming para algoritmos globales (fill, flow acc, watershed)
 - Streaming para stac composite (requiere tiled COG reading, es paso posterior)
 - COG output (requiere tiles + overviews)

  Plan: 9 nuevos algoritmos derivados de la bibliografía 

 Contexto

 Revisión de los .bib del proyecto identificó algoritmos comúnmente citados que SurtGIS no implementa pero podría derivar de funciones existentes. Todos son fórmulas sobre datos
 que SurtGIS ya computa (slope, flow_acc, HAND, stream_network, watershed).

 Algoritmos a implementar

 Grupo A: Terrain (4 algoritmos)

 A1. LS-Factor (RUSLE slope length-steepness)
 - Fórmula: LS = (flow_acc * cell_size / 22.13)^m * (sin(slope_rad) / 0.0896)^n
 - Donde m=0.4, n=1.3 (McCool 1987), o m variable según slope (Desmet & Govers 1996)
 - Input: flow_acc: Raster<f64>, slope_rad: Raster<f64>, cell_size: f64
 - Output: Raster<f64>
 - Archivo: crates/algorithms/src/terrain/ls_factor.rs

 A2. Valley Depth
 - Distancia vertical desde cada celda a la superficie de ridge interpolada
 - Algoritmo: invertir DEM → fill → diferencia con DEM original
 - Input: dem: Raster<f64>
 - Output: Raster<f64> (metros, ≥0)
 - Archivo: crates/algorithms/src/terrain/valley_depth.rs
 - Reusa: priority_flood() del módulo hydrology

 A3. Relative Slope Position (Normalized Height)
 - Fórmula: RSP = HAND / (HAND + valley_depth) → rango [0, 1]
 - 0 = fondo de valle, 1 = cresta
 - Input: hand: Raster<f64>, valley_depth: Raster<f64>
 - Output: Raster<f64> [0, 1]
 - Archivo: crates/algorithms/src/terrain/relative_slope_position.rs

 A4. Surface Area Ratio
 - Ratio de área 3D real vs área planimetrica en ventana móvil
 - Para cada celda: calcular los 8 triángulos formados con vecinos, sumar áreas 3D
 - SAR = sum(triangle_areas_3d) / (cell_size² * window_area)
 - Input: dem: Raster<f64>, radius: usize
 - Output: Raster<f64> (≥1, donde 1 = perfectamente plano)
 - Archivo: crates/algorithms/src/terrain/surface_area_ratio.rs

 Grupo B: Hydrology (4 algoritmos)

 B1. Drainage Density
 - Longitud de red de drenaje por unidad de área en ventana móvil
 - DD = (stream_cells_in_window * cell_size) / (window_area * cell_size²)
 - Input: stream_network: Raster<u8>, radius: usize, cell_size: f64
 - Output: Raster<f64> (m/m²)
 - Archivo: crates/algorithms/src/hydrology/drainage_density.rs

 B2. Hypsometric Integral
 - Por cuenca: HI = (mean_elev - min_elev) / (max_elev - min_elev)
 - Valor entre 0-1: >0.6 = joven (convexo), 0.35-0.6 = maduro (S), <0.35 = viejo (cóncavo)
 - Input: dem: Raster<f64>, watersheds: Raster<i32>
 - Output: HashMap<i32, f64> (watershed_id → HI)
 - Archivo: crates/algorithms/src/hydrology/hypsometric.rs
 - Reusa: zonal_statistics() del módulo statistics

 B3. Sediment Connectivity Index (Borselli 2008)
 - IC = log₁₀(D_up / D_dn)
 - D_up = W̄ · S̄ · √A (upslope componen
 - D_dn = Σ(dᵢ / (Wᵢ · Sᵢ)) (downslope path to channel)
 - W = weighting factor (1.0 por defecto, o NDVI-based)
 - Input: dem: Raster<f64>, slope: Raster<f64>, flow_acc: Raster<f64>, flow_dir: Raster<u8>, opcionalmente weight: Raster<f64>
 - Output: Raster<f64> (log-scale, típicamente [-10, 2])
 - Archivo: crates/algorithms/src/hydrology/sediment_connectivity.rs
 - Más complejo: requiere trazar caminos downslope hasta el canal

 B4. Basin Morphometrics
 - Para cada cuenca: área, perímetro, circularity ratio, elongation ratio, form factor, compactness
 - Circularity = 4π·A / P²
 - Elongation = (2/P) · √(A/π)
 - Form factor = A / L² (L = basin length)
 - Input: watersheds: Raster<i32>, cell_size: f64
 - Output: Vec<BasinMorphometry> struct con todas las métricas
 - Archivo: crates/algorithms/src/hydrology/basin_morphometry.rs
 - Reusa: lógica de perimeter counting de patch_metrics.rs

 Grupo C: Imagery (1 algoritmo)

 C1. dNBR (differenced Normalized Burn Ratio)
 - dNBR = NBR_pre - NBR_post
 - Ya tenemos nbr() y raster_difference() — esto es composición
 - Pero agregar como convenience function con clasificación de severidad:
   - Enhanced regrowth: < -0.25
   - Unburned: -0.25 to 0.1
   - Low: 0.1 to 0.27
   - Moderate-low: 0.27 to 0.44
   - Moderate-high: 0.44 to 0.66
   - High: > 0.66
 - Archivo: crates/algorithms/src/imagery/burn_severity.rs

 Archivos a crear/modificar

 ┌────────────────────────────────────┬────────────────────────────────┐
 │              Archivo               │             Acción             │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ terrain/ls_factor.rs               │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ terrain/valley_depth.rs            │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ terrain/relative_slope_position.rs │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ terrain/surface_area_ratio.rs      │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ terrain/mod.rs                     │ MODIFICAR — 4 nuevos exports   │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ hydrology/drainage_density.rs      │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ hydrology/hypsometric.rs           │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ hydrology/sediment_connectivity.rs │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ hydrology/basin_morphometry.rs     │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ hydrology/mod.rs                   │ MODIFICAR — 4 nuevos exports   │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ imagery/burn_severity.rs           │ CREAR                          │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ imagery/mod.rs                     │ MODIFICAR — 1 nuevo export     │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ cli/src/commands.rs                │ MODIFICAR — nuevos subcomandos │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ cli/src/handlers/terrain.rs        │ MODIFICAR — 4 handlers         │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ cli/src/handlers/hydrology.rs      │ MODIFICAR — 4 handlers         │
 ├────────────────────────────────────┼────────────────────────────────┤
 │ cli/src/handlers/imagery.rs        │ MODIFICAR — 1 handler          │
 └────────────────────────────────────┴────────────────────────────────┘

 CLI subcomandos nuevos

 # Terrain
 surtgis terrain ls-factor --flow-acc facc.tif --slope slope.tif output.tif
 surtgis terrain valley-depth dem.tif output.tif
 surtgis terrain relative-slope-position --hand hand.tif --valley-depth vd.tif output.tif
 surtgis terrain surface-area-ratio dem.tif output.tif --radius 1

 # Hydrology
 surtgis hydrology drainage-density --streams streams.tif output.tif --radius 10
 surtgis hydrology hypsometric-integral --dem dem.tif --watersheds ws.tif
 surtgis hydrology sediment-connectivity --dem dem.tif --slope slope.tif --flow-acc facc.tif --flow-dir fdir.tif output.tif
 surtgis hydrology basin-morphometry --watersheds ws.tif

 # Imagery
 surtgis imagery dnbr --pre-nir pre_nir.tif --pre-swir pre_swir.tif --post-nir post_nir.tif --post-swir post_swir.tif output.tif

 Orden de implementación

 1. LS-Factor — fórmula pura, 1 archivo, ~40 líneas
 2. Valley Depth — reusa priority_flood, ~50 líneas
 3. Relative Slope Position — fórmula pura, ~30 líneas
 4. Drainage Density — focal sobre stream network, ~50 líneas
 5. Hypsometric Integral — reusa zonal_statistics, ~40 líneas
 6. Surface Area Ratio — ventana 3×3 con trigonometría, ~80 líneas
 7. Basin Morphometry — reusa patch_metrics pattern, ~100 líneas
 8. dNBR / Burn Severity — composición de funciones existentes, ~60 líneas
 9. Sediment Connectivity — el más complejo (tracing downslope), ~150 líneas

 Verificación

 cargo build --release -p surtgis
 cargo test -p surtgis-algorithms -- ls_factor valley_depth relative_slope drainage hypsometric surface_area basin_morphometry burn_severity sediment

 # Test funcional
 surtgis terrain ls-factor --flow-acc facc.tif --slope slope.tif /tmp/ls.tif
 surtgis info /tmp/ls.tif

 Estimación

 ~600 líneas Rust (9 archivos nuevos) + ~200 líneas CLI + ~200 líneas tests = ~1000 total
 SurtGIS pasa de 127 a 136 algoritmos.

  Plan: Soporte Shapefile y GeoPackage para SurtGIS                                                                                                                                
                                                        
 Contexto

 SurtGIS solo lee GeoJSON para operaciones vectoriales (clip, rasterize). Muchos proyectos GIS trabajan con Shapefile (.shp) y GeoPackage (.gpkg). Sin soporte para estos
 formatos, un usuario con un shapefile de cuencas no puede usar surtgis clip sin convertir primero a GeoJSON — una barrera de entrada innecesaria.

 GeoPackage es el formato estándar OGC que reemplaza al Shapefile como formato de intercambio vector.

 Enfoque: auto-detección por extensión

 El CLI detecta el formato por extensión de archivo y usa el reader apropiado:
 - .geojson / .json → GeoJSON reader (existente)
 - .shp → Shapefile reader (nuevo)
 - .gpkg → GeoPackage reader (nuevo)

 La función unificada read_vector(path) devuelve FeatureCollection independientemente del formato.

 Crates a usar

 ┌───────────┬─────────┬───────────────────────────┬───────────────────┐
 │   Crate   │ Versión │         Para qué          │     Pure Rust     │
 ├───────────┼─────────┼───────────────────────────┼───────────────────┤
 │ shapefile │ latest  │ Leer .shp + .dbf + .prj   │ Sí                │
 ├───────────┼─────────┼───────────────────────────┼───────────────────┤
 │ gpkg      │ latest  │ Leer .gpkg (SQLite + WKB) │ Sí (usa rusqlite) │
 └───────────┴─────────┴───────────────────────────┴───────────────────┘

 Ambos feature-gated para no agregar dependencias a quien no las necesite. El CLI activa ambos por defecto.

 Archivos a crear/modificar

 ┌────────────────────────────────────────────┬─────────────────────────────────────────────────────────┐
 │                  Archivo                   │                         Acción                          │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ crates/core/src/vector/shapefile_reader.rs │ CREAR — read_shapefile()                                │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ crates/core/src/vector/gpkg_reader.rs      │ CREAR — read_gpkg()                                     │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ crates/core/src/vector/mod.rs              │ MODIFICAR — agregar módulos + read_vector() unificado   │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ crates/core/Cargo.toml                     │ MODIFICAR — agregar deps feature-gated                  │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ Cargo.toml (workspace)                     │ MODIFICAR — agregar deps al workspace                   │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ crates/cli/Cargo.toml                      │ MODIFICAR — activar features                            │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
 │ crates/cli/src/handlers/clip.rs            │ MODIFICAR — usar read_vector() en vez de read_geojson() │
 └────────────────────────────────────────────┴─────────────────────────────────────────────────────────┘

 Implementación

 1. shapefile_reader.rs

 use std::path::Path;
 use crate::error::Result;
 use super::{Feature, FeatureCollection, AttributeValue};

 pub fn read_shapefile(path: &Path) -> Result<FeatureCollection>

 Flujo:
 1. shapefile::Reader::from_path(path)?
 2. Iterar reader.iter_shapes_and_records()
 3. Convertir Shape → Geometry<f64> (feature geo-types del crate shapefile)
 4. Convertir dbase::Record fields → HashMap<String, AttributeValue>
 5. Construir Feature + acumular en FeatureCollection

 Mapeo de atributos dbase:
 - FieldValue::Character(s) → AttributeValue::String(s)
 - FieldValue::Numeric(f) → AttributeValue::Float(f)
 - FieldValue::Integer(i) → AttributeValue::Int(i)
 - FieldValue::Logical(b) → AttributeValue::Bool(b)
 - FieldValue::None → AttributeValue::Null

 2. gpkg_reader.rs

 pub fn read_gpkg(path: &Path, layer: Option<&str>) -> Result<FeatureCollection>

 Flujo:
 1. Abrir con gpkg::GeoPackage::open(path)?
 2. Si layer es None, usar la primera tabla de features
 3. Consultar geometrías + atributos via SQL
 4. Convertir WKB → Geometry<f64>
 5. Mapear columnas SQLite → AttributeValue

 3. Función unificada read_vector()

 /// Read vector features from any supported format (auto-detected by extension).
 pub fn read_vector(path: &Path) -> Result<FeatureCollection> {
     let ext = path.extension()
         .and_then(|e| e.to_str())
         .unwrap_or("")
         .to_lowercase();

     match ext.as_str() {
         "geojson" | "json" => read_geojson(path),
         #[cfg(feature = "shapefile")]
         "shp" => shapefile_reader::read_shapefile(path),
         #[cfg(feature = "geopackage")]
         "gpkg" => gpkg_reader::read_gpkg(path, None),
         _ => Err(Error::Other(format!(
             "Unsupported vector format: .{}. Supported: .geojson, .shp, .gpkg", ext
         ))),
     }
 }

 4. Cargo.toml changes

 Workspace Cargo.toml:
 shapefile = { version = "0.6", features = ["geo-types"] }
 gpkg = "0.1"

 Core Cargo.toml:
 [features]
 default = ["parallel"]
 shapefile = ["dep:shapefile"]
 geopackage = ["dep:gpkg"]

 [dependencies]
 shapefile = { workspace = true, optional = true }
 gpkg = { workspace = true, optional = true }

 CLI Cargo.toml:
 surtgis-core = { version = "0.2.0", path = "../core", features = ["shapefile", "geopackage"] }

 5. CLI: actualizar clip y rasterize

 Cambiar read_geojson por read_vector en handlers/clip.rs:
 // Antes
 let features = surtgis_core::vector::read_geojson(&polygon)?;

 // Después
 let features = surtgis_core::vector::read_vector(&polygon)?;

 Ahora surtgis clip --polygon cuenca.shp funciona directamente.

 Tests

 shapefile_reader

 1. Crear un shapefile de prueba en el test (o usar el crate shapefile para escribir uno temporal)
 2. Leer y verificar geometría + atributos

 gpkg_reader

 1. Crear un gpkg temporal con rusqlite
 2. Leer y verificar

 read_vector

 1. Verificar dispatch por extensión (.geojson, .shp, .gpkg)
 2. Extensión no soportada → error descriptivo

 Verificación

 cargo build --release -p surtgis
 cargo test -p surtgis-core -- vector::shapefile vector::gpkg

 # Test funcional
 surtgis clip --polygon cuenca.shp dem.tif dem_clipped.tif
 surtgis clip --polygon cuenca.gpkg dem.tif dem_clipped.tif
 surtgis clip --polygon cuenca.geojson dem.tif dem_clipped.tif

 Plan: Soporte Shapefile y GeoPackage para SurtGIS                                                                                                                                      
                                                                                                                                                                                        
 Contexto                                                                                                                                                                               
                                                                                                                                                                                        
 SurtGIS solo lee GeoJSON para operaciones vectoriales (clip, rasterize). Muchos proyectos GIS trabajan con Shapefile (.shp) y GeoPackage (.gpkg). Sin soporte para estos formatos, un  
 usuario con un shapefile de cuencas no puede usar surtgis clip sin convertir primero a GeoJSON — una barrera de entrada innecesaria.                                                   
                                                                                                                                                                                        
 GeoPackage es el formato estándar OGC que reemplaza al Shapefile como formato de intercambio vector.                                                                                   
                                                                                                                                                                                        
 Enfoque: auto-detección por extensión                                                                                                                                                  
                                                                                                                                                                                        
 El CLI detecta el formato por extensión de archivo y usa el reader apropiado:                                                                                                          
 - .geojson / .json → GeoJSON reader (existente)                                                                                                                                        
 - .shp → Shapefile reader (nuevo)                                                                                                                                                      
 - .gpkg → GeoPackage reader (nuevo)                                                                                                                                                    
                                                                                                                                                                                        
 La función unificada read_vector(path) devuelve FeatureCollection independientemente del formato.                                                                                      
                                                                                                                                                                                        
 Crates a usar                                                                                                                                                                          
                                                                                                                                                                                        
 ┌───────────┬─────────┬───────────────────────────┬───────────────────┐                                                                                                                
 │   Crate   │ Versión │         Para qué          │     Pure Rust     │                                                                                                                
 ├───────────┼─────────┼───────────────────────────┼───────────────────┤                                                                                                                
 │ shapefile │ latest  │ Leer .shp + .dbf + .prj   │ Sí                │                                                                                                                
 ├───────────┼─────────┼───────────────────────────┼───────────────────┤                                                                                                                
 │ gpkg      │ latest  │ Leer .gpkg (SQLite + WKB) │ Sí (usa rusqlite) │                                                                                                                
 └───────────┴─────────┴───────────────────────────┴───────────────────┘                                                                                                                
                                                                                                                                                                                        
 Ambos feature-gated para no agregar dependencias a quien no las necesite. El CLI activa ambos por defecto.                                                                             
                                                                                                                                                                                        
 Archivos a crear/modificar                                                                                                                                                             
                                                                                                                                                                                        
 ┌────────────────────────────────────────────┬─────────────────────────────────────────────────────────┐                                                                               
 │                  Archivo                   │                         Acción                          │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ crates/core/src/vector/shapefile_reader.rs │ CREAR — read_shapefile()                                │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ crates/core/src/vector/gpkg_reader.rs      │ CREAR — read_gpkg()                                     │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ crates/core/src/vector/mod.rs              │ MODIFICAR — agregar módulos + read_vector() unificado   │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ crates/core/Cargo.toml                     │ MODIFICAR — agregar deps feature-gated                  │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ Cargo.toml (workspace)                     │ MODIFICAR — agregar deps al workspace                   │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ crates/cli/Cargo.toml                      │ MODIFICAR — activar features                            │                                                                               
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤                                                                               
 │ crates/cli/src/handlers/clip.rs            │ MODIFICAR — usar read_vector() en vez de read_geojson() │                                                                               
[ias] 0:[tmux]*Z                                                                                                                                 "✳ expand-cli-imagery-" 16:40 25-mar-26

Implementación

 1. shapefile_reader.rs

 use std::path::Path;
 use crate::error::Result;
 use super::{Feature, FeatureCollection, AttributeValue};

 pub fn read_shapefile(path: &Path) -> Result<FeatureCollection>

 Flujo:
 1. shapefile::Reader::from_path(path)?
 2. Iterar reader.iter_shapes_and_records()
 3. Convertir Shape → Geometry<f64> (feature geo-types del crate shapefile)
 4. Convertir dbase::Record fields → HashMap<String, AttributeValue>
 5. Construir Feature + acumular en FeatureCollection

 Mapeo de atributos dbase:
 - FieldValue::Character(s) → AttributeValue::String(s)
 - FieldValue::Numeric(f) → AttributeValue::Float(f)
 - FieldValue::Integer(i) → AttributeValue::Int(i)
 - FieldValue::Logical(b) → AttributeValue::Bool(b)
 - FieldValue::None → AttributeValue::Null

 2. gpkg_reader.rs

 pub fn read_gpkg(path: &Path, layer: Option<&str>) -> Result<FeatureCollection>

 Flujo:
 1. Abrir con gpkg::GeoPackage::open(path)?
 2. Si layer es None, usar la primera tabla de features
 3. Consultar geometrías + atributos via SQL
 4. Convertir WKB → Geometry<f64>
 5. Mapear columnas SQLite → AttributeValue

 3. Función unificada read_vector()

 /// Read vector features from any supported format (auto-detected by extension).
 pub fn read_vector(path: &Path) -> Result<FeatureCollection> {
     let ext = path.extension()
         .and_then(|e| e.to_str())
         .unwrap_or("")
         .to_lowercase();

     match ext.as_str() {
         "geojson" | "json" => read_geojson(path),
         #[cfg(feature = "shapefile")]
         "shp" => shapefile_reader::read_shapefile(path),

 Plan: --max-memory Flag para Auto-Streaming (Prioridad 1)                                                                                                                              
                                                                                                                                                                                        
 Contexto                                                                                                                                                                               
                                                                                                                                                                                        
 Problema de producción: SurtGIS freezea con cuencas grandes (>8GB decomprimido). El usuario encontró esto procesando 15 cuencas Chile + DEM Copernicus + Sentinel-2.                   
                                                                                                                                                                                        
 Root cause:                                                                                                                                                                            
 - terrain all, hydrology all, clip cargan rasters completos sin límite                                                                                                                 
 - read_geotiff() crea Vec<f64> del tamaño total sin bound checking                                                                                                                     
 - Heurística actual: solo streaming si archivo >500MB (incorrecto: 500MB archivo ≠ 500MB RAM con compresión)                                                                           
                                                                                                                                                                                        
 Solución: Flag global --max-memory 4G que fuerza streaming automático si DEM decomprimido excede límite.                                                                               
                                                                                                                                                                                        
 ---                                                                                                                                                                                    
 Enfoque                                                                                                                                                                                
                                                                                                                                                                                        
 Memory estimation + auto-streaming fallback:                                                                                                                                           
 1. Parser: Convierte "4G", "1024MB", "2GiB" → bytes                                                                                                                                    
 2. Estimador: Lee metadata TIFF (width, height) sin descomprimir, estima rows × cols × 8 (Float64)                                                                                     
 3. Decisión en handlers: use_streaming = --streaming OR (file_size > 500MB) OR (--max-memory Y AND estimated_size > Y)                                                                 
                                                                                                                                                                                        
 Archivos a crear/modificar                                                                                                                                                             
                                                                                                                                                                                        
 ┌──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┬──────────┐                                                       
 │               Archivo                │                                   Acción                                   │  Líneas  │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/memory.rs             │ CREAR — parse_memory_size(), estimate_decompressed_size(), should_stream() │ -        │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/commands.rs           │ MODIFICAR — agregar field max_memory: Option<String>                       │ ~20-25   │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/main.rs               │ MODIFICAR — parsear max_memory a bytes, pasar a handlers                   │ ~16-38   │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/handlers/terrain.rs   │ MODIFICAR — usar should_stream() con mem_limit                             │ ~26-80   │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/handlers/hydrology.rs │ MODIFICAR — usar should_stream() con mem_limit                             │ ~200-349 │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/handlers/clip.rs      │ MODIFICAR — validar DEM no excede limite                                   │ ~10-32   │                                                       
 ├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼──────────┤                                                       
 │ crates/cli/src/lib.rs                │ MODIFICAR — agregar pub mod memory;                                        │ -        │                                                       
 └──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┴──────────┘                                                       
                                                                                                                                                                                        
 Implementación concreta

 1. memory.rs (nuevo módulo)

 pub fn parse_memory_size(s: &str) -> Result<u64> {

    // "4G" → 4_000_000_000                                                                                                                                                            
     // "1024MB" → 1_073_741_824                                                                                                                                                        
     // "500MiB" → 524_288_000
     // Soporta: B, KB/K, MB/M, GB/G, TB/T
     //         KIB, MIB, GIB, TIB
 }

 pub fn estimate_decompressed_size(path: &Path) -> Result<u64> {
     // Lee metadata TIFF con StripReader (ya existe en core)
     // Retorna: rows × cols × 8 bytes (asume Float64)
 }

 pub fn should_stream(
     file_path: &Path,
     max_memory_bytes: Option<u64>,
     force_streaming: bool,
 ) -> Result<bool> {
     // if force_streaming: return true
     // if file_size > 500MB: return true
     // if max_memory AND estimated_size > max_memory: return true
     // else: false
 }

 Tests incluidos para parse_memory_size() con casos edge.

 2. CLI: commands.rs

 Agregar después de streaming flag:

 /// Maximum memory to use (e.g., 4G, 1024MB, 500MiB).
 /// If raster would exceed this when decompressed, force streaming.
 #[arg(long, global = true)]
 pub max_memory: Option<String>,

 3. CLI: main.rs

 let mem_limit_bytes = cli.max_memory
     .as_ref()
     .map(|s| memory::parse_memory_size(s))
     .transpose()?;

 // Pasar a handlers

 4. Terrain handler

 let use_streaming = memory::should_stream(
     &input,
     mem_limit_bytes,
     cli.streaming,
 )?;

 // Resto de lógica igual: if use_streaming { ... } else { ... }

 5. Hydrology handler (línea ~277)

 let use_streaming = memory::should_stream(
     &dem_path,
     mem_limit_bytes,
     false,
 )?;

 // Si use_streaming=true: emitir warning

 6. Clip handler

 // Validar que DEM no excede límite
 if let Some(limit) = mem_limit_bytes {
     let est_size = memory::estimate_decompressed_size(&input)?;
     if est_size > limit {
         eprintln!("Warning: DEM exceeds --max-memory but clip requires in-memory processing");
     }
 }

 Testing

 Unit tests

 - ✅ parse_memory_size con todas las unidades
 - ✅ Whitespace, case-insensitivity
 - ✅ Números flotantes
 - ✅ Error handling

 Integration

 - ✅ Verificar should_stream() con diferentes límites
 - ✅ terrain slope con --max-memory 500M (fuerza streaming)
 - ✅ terrain slope con --max-memory 2G (no streaming)

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: MVP surtgis pipeline susceptibility - Comando End-to-End                                                                                                                         
                                                                                                                                                                                        
 Contexto                                                                                                                                                                               
                                                                                                                                                                                        
 Problema: En producción, los usuarios necesitan hacer:                                                                                                                                 
 # Script bash con 3+ pasos manuales                                                                                                                                                    
 surtgis stac composite --collection copernicus-dem-glo-30 ... dem.tif                                                                                                                  
 surtgis terrain all dem.tif terrain_out/                                                                                                                                               
 surtgis hydrology all dem.tif hydrology_out/                                                                                                                                           
 surtgis stac composite --collection sentinel-2-l2a ... s2_composite.tif                                                                                                                
 surtgis resample s2_composite.tif dem.tif s2_aligned.tif                                                                                                                               
 surtgis imagery ndvi s2_aligned_nir.tif s2_aligned_red.tif ndvi.tif                                                                                                                    
 # ... más índices ...                                                                                                                                                                  
 # Luego combinar en Python                                                                                                                                                             
                                                                                                                                                                                        
 Solución MVP: Un comando unificado que hace todo en una pasada:                                                                                                                        
 surtgis pipeline susceptibility \                                                                                                                                                      
   --dem copernicus-dem-glo-30 \                                                                                                                                                        
   --s2 sentinel-2-l2a \                                                                                                                                                                
   --bbox -70.5,-33.6,-70.2,-33.3 \                                                                                                                                                     
   --datetime 2024-01-01/2024-12-31 \                                                                                                                                                   
   --outdir ./factors/ \                                                                                                                                                                
   --max-memory 4G                                                                                                                                                                      
                                                                                                                                                                                        
 Salida esperada:                                                                                                                                                                       
 factors/                                                                                                                                                                               
 ├── dem.tif                    # DEM descargado y clipeado
 ├── terrain/
 │   ├── slope.tif
 │   ├── aspect.tif
 │   ├── hand.tif
 │   ├── flow_accumulation.tif
 │   └── ... (13 más de terrain all)
 ├── hydrology/
 │   ├── flow_direction_d8.tif
 │   ├── flow_accumulation.tif
 │   ├── twi.tif
 │   └── ... (5 más de hydrology all)
 └── imagery/
     ├── ndvi.tif               # (NIR - Red) / (NIR + Red)
     ├── ndbi.tif               # (SWIR - NIR) / (SWIR + NIR)
     ├── ndwi.tif               # (Green - NIR) / (Green + NIR)
     └── ... (7 más)

 ---

 Enfoque: Orchestration simplificada                                                                                                                                                    
                                                                                                                                                                                        
 Responsabilidad del pipeline:                                                                                                                                                          
 1. Orquestar los pasos existentes (terrain all, hydrology all, imagery indices)
 2. Manejar dependencias (DEM → terrain/hydrology, S2 → índices)
 3. Alinear grillas (S2 a DEM)
 4. Normalizar outputs
 5. Organizados en subdirectorios

 Reutilizar todo lo posible:
 - ✅ STAC composite existente (para DEM y S2)
 - ✅ Terrain all handler existente
 - ✅ Hydrology all handler existente
 - ✅ Imagery indices existentes
 - ✅ Resample logic existente
 - ✅ Memory estimation existente

 ---
 Archivos a crear/modificar

 ┌────────────────────────────────────────────────────────┬───────────────────────────────────────────┬─────────┐
 │                        Archivo                         │                  Acción                   │ Líneas  │
 ├────────────────────────────────────────────────────────┼───────────────────────────────────────────┼─────────┤
 │ crates/cli/src/commands.rs                             │ MODIFICAR — agregar PipelineCommands enum │ +20     │
 ├────────────────────────────────────────────────────────┼───────────────────────────────────────────┼─────────┤
 │ crates/cli/src/handlers/pipeline.rs                    │ CREAR — handle_susceptibility()           │ 200-250 │
 ├────────────────────────────────────────────────────────┼───────────────────────────────────────────┼─────────┤
 │ crates/cli/src/handlers/mod.rs                         │ MODIFICAR — agregar pub mod pipeline      │ +1      │
 ├────────────────────────────────────────────────────────┼───────────────────────────────────────────┼─────────┤
 │ crates/cli/src/main.rs                                 │ MODIFICAR — match cli.command Pipeline    │ +3      │
 ├────────────────────────────────────────────────────────┼───────────────────────────────────────────┼─────────┤
 │ crates/algorithms/tests/integration_production_bugs.rs │ MODIFICAR — agregar test_pipeline_e2e     │ +150    │
 └────────────────────────────────────────────────────────┴───────────────────────────────────────────┴─────────┘

 ---

 Fase 1: CLI plumbing (1h)                                                                                                                                           21:50:49 [235/1803]

 commands.rs:
 pub enum Commands {
     // ... existing ...
     Pipeline { action: PipelineCommands },
 }

 #[derive(Subcommand)]
 pub enum PipelineCommands {
     /// Compute susceptibility factors from DEM + S2 imagery
     Susceptibility {
         /// DEM source: "copernicus-dem-glo-30" or local path
         #[arg(long)]
         dem: String,

         /// S2 source: "sentinel-2-l2a" or "earth-search"
         #[arg(long)]
         s2: String,

         /// Bounding box: "west,south,east,north"
         #[arg(long)]
         bbox: String,

         /// Date range: "YYYY-MM-DD/YYYY-MM-DD"
         #[arg(long)]
         datetime: String,

         /// Output directory (will be created)
         #[arg(long)]
         outdir: PathBuf,

         /// Max scenes for S2 (default: 12)
         #[arg(long, default_value = "12")]
         max_scenes: usize,

         /// Cloud mask threshold for S2 (default: "4,5,6,11")
         #[arg(long, default_value = "4,5,6,11")]
         scl_keep: String,
     },
 }

 main.rs:
 Commands::Pipeline { action } => {
     handlers::pipeline::handle(action, cli.compress, mem_limit_bytes)?
 }

 ---

 Fase 2: Core orchestration (6-8h)                                                                                                                                                      
                                                                                                                                                                                        
 handlers/pipeline.rs:                                                                                                                                                                  
                                                                                                                                                                                        
 pub fn handle(                                                                                                                                                                         
     command: PipelineCommands,                                                                                                                                                         
     compress: bool,                                                                                                                                                                    
     mem_limit_bytes: Option<u64>,                                                                                                                                                      
 ) -> Result<()> {                                                                                                                                                                      
     match command {                                                                                                                                                                    
         PipelineCommands::Susceptibility {                                                                                                                                             
             dem,                                                                                                                                                                       
             s2,                                                                                                                                                                        
             bbox,                                                                                                                                                                      
             datetime,                                                                                                                                                                  
             outdir,                                                                                                                                                                    
             max_scenes,                                                                                                                                                                
             scl_keep,                                                                                                                                                                  
         } => handle_susceptibility(                                                                                                                                                    
             &dem, &s2, &bbox, &datetime, &outdir,                                                                                                                                      
             max_scenes, &scl_keep, compress, mem_limit_bytes,                                                                                                                          
         ),                                                                                                                                                                             
     }                                                                                                                                                                                  
 }                      

  pub fn handle_susceptibility(                                                                                                                                                          
     dem_source: &str,                                                                                                                                                                  
     s2_source: &str,                                                                                                                                                                   
     bbox: &str,                                                                                                                                                                        
     datetime: &str,                                                                                                                                                                    
     outdir: &Path,                                                                                                                                                                     
     max_scenes: usize,                                                                                                                                                                 
     scl_keep: &str,                                                                                                                                                                    
     compress: bool,                                                                                                                                                                    
     mem_limit_bytes: Option<u64>,                                                                                                                                                      
 ) -> Result<()> {                                                                                                                                                                      
     // Create output structure                                                                                                                                                         
     std::fs::create_dir_all(outdir)?;                                                                                                                                                  
     std::fs::create_dir_all(&outdir.join("terrain"))?;                                                                                                                                 
     std::fs::create_dir_all(&outdir.join("hydrology"))?;                                                                                                                               
     std::fs::create_dir_all(&outdir.join("imagery"))?;                                                                                                                                 
                                                                                                                                                                                        
     let pb = spinner("Pipeline: DEM");                                                                                                                                                 

     // STEP 1: Download DEM (via STAC or local)
     let dem_path = if dem_source.contains("/") {
         PathBuf::from(dem_source) // local file
     } else {
         // Download from STAC
         let temp_dem = outdir.join("dem_temp.tif");
         stac::fetch_and_mosaic(dem_source, bbox, None, &temp_dem)?;
         temp_dem
     };

     let dem = read_dem(&dem_path)?;
     std::fs::copy(&dem_path, &outdir.join("dem.tif"))?;
     pb.finish_and_clear();

     // STEP 2: Terrain all (17 products)
     pb.set_message("Pipeline: Terrain");
     terrain::handle_terrain_all(&dem_path, &outdir.join("terrain"), compress, mem_limit_bytes)?;
     pb.finish_and_clear();

     // STEP 3: Hydrology all (8 products)
     pb.set_message("Pipeline: Hydrology");
     hydrology::handle_hydrology_all(&dem_path, &outdir.join("hydrology"), compress, mem_limit_bytes)?;
     pb.finish_and_clear();

     // STEP 4: S2 Composite (aligned to DEM)
     pb.set_message("Pipeline: Sentinel-2");
     let s2_path = outdir.join("s2_composite.tif");
     stac::fetch_stac_composite(
         s2_source, bbox, datetime, max_scenes, &s2_path,
         Some(&dem_path), compress, // align_to DEM
     )?;
     pb.finish_and_clear();

     // STEP 5: Imagery indices (extract bands + compute)
     pb.set_message("Pipeline: Imagery indices");
     compute_s2_indices(&s2_path, &outdir.join("imagery"), compress)?;
     pb.finish_and_clear();

     println!("✅ Pipeline complete: {}", outdir.display());
     println!("Outputs:");
     println!("  - DEM: dem.tif");
     println!("  - Terrain factors (17): terrain/*.tif");
     println!("  - Hydrology factors (8): hydrology/*.tif");
     println!("  - S2 indices (10): imagery/*.tif");

     Ok(())
 }

 fn compute_s2_indices(s2_path: &Path, outdir: &Path, compress: bool) -> Result<()> {                                                                                                   
     // Read S2 composite (assumed: B02=Blue, B03=Green, B04=Red, B08=NIR, B11=SWIR)
     let s2 = read_dem(s2_path)?;

     // For S2 composite, all bands are in same raster
     // Need to extract individual bands from multiband S2
     // Option 1: User provides pre-processed NIR, Red, etc.
     // Option 2: Use gdal/gdalinfo to extract bands
     // SIMPLIFICATION: Assume S2 is single-band NIR or already processed

     // Placeholder: just write S2 composite to imagery dir
     write_result(&s2, &outdir.join("s2_composite.tif"), compress)?;

     Ok(())
 }

 ---
 Fase 3: Reuse existing handlers (2h)

 Key functions to reuse:

 1. From stac.rs:
   - StacClientBlocking::search_all() for DEM/S2 discovery
   - CogReaderBlocking for windowed reads
   - Existing handle_stac_composite() for S2 download
 2. From terrain.rs:
   - Extract loop from handle_terrain_all() (lines 597-711)
   - Reutilizar sin cambios
 3. From hydrology.rs:
   - Extract loop from handle_hydrology_all() (lines 278-369)
   - Reutilizar sin cambios
 4. From imagery.rs:
   - Index computation (ndvi, ndbi, ndwi, etc.)
   - Reusable as imagery::handle_ndvi(), etc.

 ---

 Fase 4: Testing (2h)                                                                                                                                                                   
                                                                                                                                                                                        
 Test ubicación: crates/algorithms/tests/integration_production_bugs.rs                                                                                                                 
                                                                                                                                                                                        
 #[test]                                                                                                                                                                                
 fn test_pipeline_susceptibility_end_to_end() {                                                                                                                                         
     // Use Río Salado DEM from Agentes                                                                                                                                                 
     let dem_path = "/home/franciscoparrao/proyectos/Agentes/salado_utm_cropped.tif";                                                                                                   
                                                                                                                                                                                        
     // Create temp output dir                                                                                                                                                          
     let outdir = tempdir()?;                                                                                                                                                           
                                                                                                                                                                                        
     // Run pipeline                                                                                                                                                                    
     surtgis_cli::handlers::pipeline::handle_susceptibility(                                                                                                                            
         dem_path,                    // local DEM                                                                                                                                      
         "skip",                       // no S2 for MVP test                                                                                                                            
         "-70.5,-33.6,-70.2,-33.3",  // bbox (not used in local mode)                                                                                                                   
         "2024-01-01/2024-12-31",     // datetime                                                                                                                                       
         &outdir,                                                                                                                                                                       
         12,                                                                                                                                                                            
         "4,5,6,11",                                                                                                                                                                    
         true,  // compress                                                                                                                                                             
         Some(1_000_000_000),  // 1GB limit                                                                                                                                             
     )?;                                                                                                                                                                                
                                                                                                                                                                                        
     // Validate outputs exist                                                                                                                                                          
     assert!(outdir.join("dem.tif").exists());                                                                                                                                          
     assert!(outdir.join("terrain/slope.tif").exists());                                                                                                                                
     assert!(outdir.join("terrain/hand.tif").exists());                                                                                                                                 
     assert!(outdir.join("hydrology/flow_accumulation.tif").exists());                                                                                                                  
                                                                                                                                                                                        
     // Validate metadata                                                                                                                                                               
     let slope = read_geotiff(&outdir.join("terrain/slope.tif"), None)?;                                                                                                                
     assert_eq!(slope.crs(), Some(CRS::from_epsg(32719))); // UTM 19S                                                                                                                   
     assert!(slope.statistics().min.unwrap() >= 0.0);                                                                                                                                   
     assert!(slope.statistics().max.unwrap() <= 90.0);                                                                                                                                  
                                                                                                                                                                                        
     println!("✅ Pipeline end-to-end test passed");                                                                                                                                    
     Ok(())
 }

 ---

 Timeline                                                                                                                                                                               
                                                                                                                                                                                        
 ┌───────┬─────────────────────────────────────┬───────┐                                                                                                                                
 │ Phase │                Task                 │ Hours │
 ├───────┼─────────────────────────────────────┼───────┤
 │ 1     │ CLI plumbing (commands.rs, main.rs) │ 1     │
 ├───────┼─────────────────────────────────────┼───────┤
 │ 2     │ Core orchestration (pipeline.rs)    │ 8     │
 ├───────┼─────────────────────────────────────┼───────┤
 │ 3     │ Refactor handlers for reuse         │ 2     │
 ├───────┼─────────────────────────────────────┼───────┤
 │ 4     │ Testing + debugging                 │ 2     │
 ├───────┼─────────────────────────────────────┼───────┤
 │ Total │                                     │ 13h   │
 └───────┴─────────────────────────────────────┴───────┘

 ---
 Verificación

 Manual test

 # Local DEM (no STAC download for speed)
 surtgis pipeline susceptibility \
   --dem /path/to/dem.tif \
   --s2 skip \
   --bbox -70.5,-33.6,-70.2,-33.3 \
   --datetime 2024-01-01/2024-12-31 \
   --outdir ./test_output/ \
   --max-memory 2G

 # Should complete in ~2-3 min (Río Salado size)
 # Output: test_output/terrain/*.tif, hydrology/*.tif, s2_composite.tif

 CI test

 cargo test -p surtgis-algorithms test_pipeline_susceptibility_end_to_end

 Success criteria

 - ✅ Command accepts arguments
 - ✅ Terrain products (17) generated
 - ✅ Hydrology products (8) generated
 - ✅ S2 composite downloaded + aligned (if not skipped)
 - ✅ All outputs have correct CRS + geotransform
 - ✅ No OOM with --max-memory 1G
 - ✅ Completes in <5 min for 5k×3.6k DEM

 ---
 Notas de implementación

 1. S2 multiband handling: MVP simplificado — si S2 es multiband, extracto de bandas individualmente (gdal required for complex case, skip for now)
 2. Error recovery: Si falla S2 download, continúa con terrain/hydrology solamente (no fatal)
 3. Feature gates: Sin features adicionales (reutilizar shapefile, geopackage ya existentes)
 4. Performance: Streaming automático vía --max-memory existente
 5. Logging: Progress bars para cada step, resumen final con rutas de outputs

 Plan: S2 Imagery Integration en SurtGIS Pipeline                                                                                                                                       
                                                                                                                                                                                        
 Contexto                                                                                                                                                                               
                                                                                                                                                                                        
 Objetivo: Agregar descarga automática de Sentinel-2 L2A + cómputo de índices espectrales al pipeline MVP surtgis pipeline susceptibility.                                              
                                                                                                                                                                                        
 Resultado esperado:                                                                                                                                                                    
 surtgis pipeline susceptibility \                                                                                                                                                      
   --dem dem.tif \                                                                                                                                                                      
   --s2 sentinel-2-l2a \                                                                                                                                                                
   --bbox -70.5,-33.6,-70.2,-33.3 \                                                                                                                                                     
   --datetime 2024-01-01/2024-12-31 \                                                                                                                                                   
   --outdir ./factors/ \                                                                                                                                                                
   --max-memory 4G                                                                                                                                                                      
                                                                                                                                                                                        
 # Output:                                                                                                                                                                              
 # factors/                                                                                                                                                                             
 # ├── dem.tif                                                                                                                                                                          
 # ├── terrain/ (17 productos)                                                                                                                                                          
 # ├── hydrology/ (8 productos)                                                                                                                                                         
 # └── imagery/ (10+ índices)                                                                                                                                                           
 #     ├── s2_composite_B04.tif (red)                                                                                                                                                   
 #     ├── s2_composite_B08.tif (nir)                                                                                                                                                   
 #     ├── ndvi.tif
 #     ├── ndwi.tif
 #     ├── mndwi.tif
 #     ├── nbr.tif
 #     ├── savi.tif
 #     ├── evi.tif
 #     ├── bsi.tif
 #     ├── ndbi.tif
 #     └── ndmi.tif

 ---
 Enfoque: Reutilizar máximo, agregar mínimo

 Bandas requeridas para MVP (7 bandas):
 - B02 (Blue) → para EVI, BSI
 - B03 (Green) → para NDWI, MNDWI, NDSI, GNDVI
 - B04 (Red) → para NDVI, SAVI, EVI, BSI, NDBI, NGRDI
 - B08 (NIR) → para todos los índices de vegetación
 - B11 (SWIR 1) → para MNDWI, NDMI
 - B12 (SWIR 2) → para NBR, NDMI, BSI

 Enfoque: Reutilizar máximo, agregar mínimo                                                                                                                                             
                                                                                                                                                                                        
 Bandas requeridas para MVP (7 bandas):                                                                                                                                                 
 - B02 (Blue) → para EVI, BSI                                                                                                                                                           
 - B03 (Green) → para NDWI, MNDWI, NDSI, GNDVI                                                                                                                                          
 - B04 (Red) → para NDVI, SAVI, EVI, BSI, NDBI, NGRDI                                                                                                                                   
 - B08 (NIR) → para todos los índices de vegetación                                                                                                                                     
 - B11 (SWIR 1) → para MNDWI, NDMI                                                                                                                                                      
 - B12 (SWIR 2) → para NBR, NDMI, BSI                                                                                                                                                   
                                                                                                                                                                                        
 Índices a computar en MVP (10 indices):                                                                                                                                                
 1. NDVI = (NIR - Red) / (NIR + Red)                                                                                                                                                    
 2. NDWI = (Green - NIR) / (Green + NIR)                                                                                                                                                
 3. MNDWI = (Green - SWIR1) / (Green + SWIR1)                                                                                                                                           
 4. NBR = (NIR - SWIR2) / (NIR + SWIR2)                                                                                                                                                 
 5. SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)                                                                                                                                      
 6. EVI = 2.5 * (NIR - Red) / (NIR + 6Red - 7.5Blue + 1)                                                                                                                                
 7. BSI = ((SWIR2 + Red) - (NIR + Blue)) / ((SWIR2 + Red) + (NIR + Blue))                                                                                                               
 8. NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)                                                                                                                                                
 9. NDMI = (NIR - SWIR1) / (NIR + SWIR1)                                                                                                                                                
 10. NDSI = (Green - SWIR1) / (Green + SWIR1)                                                                                                                                           
                                                                                                                                                                                        
 ---                                                                                                                                                                                    
 Archivos a modificar                                                                                                                                                                   
                                                                                                                                                                                        
 ┌──────────────────────────────────────────┬──────────────────────────────────────────────────┬──────────┐                                                                             
 │                 Archivo                  │                      Acción                      │  Líneas  │                                                                             
 ├──────────────────────────────────────────┼──────────────────────────────────────────────────┼──────────┤                                                                             
 │ crates/cli/src/handlers/pipeline.rs      │ MODIFICAR — agregar step S2 + índices            │ +200-250 │                                                                             
 ├──────────────────────────────────────────┼──────────────────────────────────────────────────┼──────────┤                                                                             
 │ crates/cli/src/handlers/imagery.rs       │ MODIFICAR — extraer en función compute_indices() │ +50      │                                                                             
 ├──────────────────────────────────────────┼──────────────────────────────────────────────────┼──────────┤                                                                             
 │ crates/cli/src/streaming.rs              │ USAR EXISTENTE — resolve_asset_key() ya existe   │ -        │                                                                             
 ├──────────────────────────────────────────┼──────────────────────────────────────────────────┼──────────┤                                                                             
 │ crates/algorithms/src/imagery/indices.rs │ USAR EXISTENTE — todas las funciones listas      │ -        │                                                                             
 └──────────────────────────────────────────┴──────────────────────────────────────────────────┴──────────┘                                                                             
                                                                                                                                                                                        
 ---                                                           

  Implementación en fases                                                                                                                                                                
                                                                                                                                                                                        
 Fase 1: Modificar pipeline.rs (2-3h)                                                                                                                                                   
                                                                                                                                                                                        
 Reemplazar el STEP 4 actual (que era placeholder):                                                                                                                                     
                                                                                                                                                                                        
 // STEP 4: S2 Composite + Band extraction (aligned to DEM)                                                                                                                             
 let pb = helpers::spinner("Pipeline: Sentinel-2 (6 bands)");                                                                                                                           
                                                                                                                                                                                        
 // Fetch S2 bands (B02, B03, B04, B08, B11, B12)                                                                                                                                       
 let bands = download_s2_bands(                                                                                                                                                         
     s2_source,                                                                                                                                                                         
     bbox_str,                                                                                                                                                                          
     datetime_str,                                                                                                                                                                      
     max_scenes,                                                                                                                                                                        
     &scl_keep_str,                                                                                                                                                                     
     &output_dem,  // align_to DEM                                                                                                                                                      
 )?;                                                                                                                                                                                    

 pb.finish_and_clear();

 // STEP 5: Compute indices
 let pb = helpers::spinner("Pipeline: Imagery indices (10 indices)");

 compute_imagery_indices(&bands, &outdir.join("imagery"), compress)?;

 pb.finish_and_clear();

 Nuevas funciones a agregar en pipeline.rs:

 struct S2Bands {
     blue: Raster<f64>,    // B02
     green: Raster<f64>,   // B03
     red: Raster<f64>,     // B04
     nir: Raster<f64>,     // B08
     swir1: Raster<f64>,   // B11
     swir2: Raster<f64>,   // B12
     scl: Option<Raster<f64>>,
 }

 fn download_s2_bands(
     s2_source: &str,
     bbox_str: &str,
     datetime_str: &str,
     max_scenes: usize,
     scl_keep_str: &str,
     align_to: &Path,
 ) -> Result<S2Bands> {
     // 1. Para cada banda (B02, B03, B04, B08, B11, B12):
     //    - Llamar handle_stac_composite() con --asset B0X
     //    - Retorna composite.tif (cloud-masked, alineado)
     // 2. Retornar struct S2Bands
 }

 fn compute_imagery_indices(
     bands: &S2Bands,
     outdir: &Path,
     compress: bool,
 ) -> Result<()> {
     // Computar 10 índices
     // Guardar en imagery/*.tif
 }

 ---
 Fase 2: Refactorizar download (1-2h)                                                                                                                                                   
                                                                                                                                                                                        
 Crear función helper que reutiliza handle_stac_composite():                                                                                                                            
                                                                                                                                                                                        
 fn fetch_s2_band(
     collection: &str,          // "sentinel-2-l2a"
     band: &str,                // "B04", "B08", etc.
     bbox: &str,
     datetime: &str,
     max_scenes: usize,
     align_to: &Path,
     output_path: &Path,
 ) -> Result<Raster<f64>> {
     // Reusa stac::handle() internamente
     // Retorna Raster<f64> alineado
 }

 Pseudocódigo:
 pub fn fetch_s2_band(...) -> Result<Raster<f64>> {
     let temp_output = PathBuf::from("temp.tif");

     // Llamar internamente a stac handler
     stac::handle_stac_composite(
         "pc",                  // catalog (Planetary Computer)
         bbox,
         collection,            // "sentinel-2-l2a"
         band,                  // "B04"
         None,                  // scl_asset (None para bandas normales)
         None,                  // scl_keep (N/A)
         max_scenes,
         None,                  // max_scenes
         &temp_output,
         Some(align_to),        // align_to DEM grid
         true,                  // compress
     )?;

     let result = read_geotiff(&temp_output, None)?;
     std::fs::remove_file(&temp_output)?;

     Ok(result)
 }

 ---

 Fase 3: Computar índices (1-2h)                                                                                                                                      00:13:11 [100/325]
                                                                                                                                                                                        
 En compute_imagery_indices(), iterar sobre índices y guardar:                                                                                                                          
                                                                                                                                                                                        
 fn compute_imagery_indices(                                                                                                                                                            
     bands: &S2Bands,                                                                                                                                                                   
     outdir: &Path,
     compress: bool,
 ) -> Result<()> {
     // NDVI
     let ndvi = surtgis_algorithms::imagery::ndvi(&bands.nir, &bands.red)?;
     helpers::write_result(&ndvi, &outdir.join("ndvi.tif"), compress)?;

     // NDWI
     let ndwi = surtgis_algorithms::imagery::ndwi(&bands.green, &bands.nir)?;
     helpers::write_result(&ndwi, &outdir.join("ndwi.tif"), compress)?;

     // MNDWI
     let mndwi = surtgis_algorithms::imagery::mndwi(&bands.green, &bands.swir1)?;
     helpers::write_result(&mndwi, &outdir.join("mndwi.tif"), compress)?;

     // NBR
     let nbr = surtgis_algorithms::imagery::nbr(&bands.nir, &bands.swir2)?;
     helpers::write_result(&nbr, &outdir.join("nbr.tif"), compress)?;

     // SAVI (default L=0.5)
     let savi_params = surtgis_algorithms::imagery::SaviParams { l: 0.5 };
     let savi = surtgis_algorithms::imagery::savi(&bands.nir, &bands.red, savi_params)?;
     helpers::write_result(&savi, &outdir.join("savi.tif"), compress)?;

     // EVI (default gains)
     let evi_params = surtgis_algorithms::imagery::EviParams::default();
     let evi = surtgis_algorithms::imagery::evi(&bands.nir, &bands.red, &bands.blue, evi_params)?;
     helpers::write_result(&evi, &outdir.join("evi.tif"), compress)?;

     // BSI
     let bsi = surtgis_algorithms::imagery::bsi(&bands.swir2, &bands.red, &bands.nir, &bands.blue)?;
     helpers::write_result(&bsi, &outdir.join("bsi.tif"), compress)?;

     // NDBI
     let ndbi = surtgis_algorithms::imagery::ndbi(&bands.swir1, &bands.nir)?;
     helpers::write_result(&ndbi, &outdir.join("ndbi.tif"), compress)?;

     // NDMI
     let ndmi = surtgis_algorithms::imagery::ndmi(&bands.nir, &bands.swir1)?;
     helpers::write_result(&ndmi, &outdir.join("ndmi.tif"), compress)?;

     // NDSI
     let ndsi = surtgis_algorithms::imagery::ndsi(&bands.green, &bands.swir1)?;
     helpers::write_result(&ndsi, &outdir.join("ndsi.tif"), compress)?;

     Ok(())
 }

 ---
 Fase 4: Testing (1-2h)

 Test ubicación: crates/algorithms/tests/integration_production_bugs.rs

 Agregar test E2E con datos reales:

 #[test]
 fn test_pipeline_with_s2_imagery() {
     // Use Río Salado DEM
     let dem_path = "/home/franciscoparrao/proyectos/Agentes/salado_utm_cropped.tif";
     let outdir = TempDir::new().unwrap();

     // MVP: skip S2 download, test con bandas sintéticas
     // TODO: Usar datos STAC cuando esté lista la descarga

     // Verificar outputs
     assert!(outdir.path().join("imagery/ndvi.tif").exists());
     assert!(outdir.path().join("imagery/ndwi.tif").exists());
 }

 ---
 Timeline

 ┌───────┬───────────────────────────────────────────┬───────┐
 │ Phase │                   Task                    │ Hours │
 ├───────┼───────────────────────────────────────────┼───────┤
 │ 1     │ Modificar pipeline.rs (S2 step + índices) │ 2-3   │
 ├───────┼───────────────────────────────────────────┼───────┤
 │ 2     │ Refactorizar download helper              │ 1-2   │
 ├───────┼───────────────────────────────────────────┼───────┤
 │ 3     │ Implementar compute_imagery_indices()     │ 1-2   │
 ├───────┼───────────────────────────────────────────┼───────┤
 │ 4     │ Testing + debugging                       │ 1-2   │
 ├───────┼───────────────────────────────────────────┼───────┤
 │ Total │                                           │ 6-9h  │
 └───────┴───────────────────────────────────────────┴───────┘


 Verificación

 Manual test

 surtgis pipeline susceptibility \
   --dem /path/to/dem.tif \
   --s2 sentinel-2-l2a \
   --bbox -70.5,-33.6,-70.2,-33.3 \
   --datetime 2024-06-01/2024-08-31 \
   --outdir ./test_pipeline_s2/ \
   --max-memory 4G

 # Should complete in ~15-20 min:
 # - Terrain: ~10 min
 # - Hydrology: ~1 min
 # - S2 download (6 bands): ~3-4 min
 # - Índices (10): ~1 min

 Success criteria

 - ✅ 6 bandas S2 descargadas y alineadas
 - ✅ 10 índices computados
 - ✅ Todos tienen CRS + geotransform correctos
 - ✅ Valores en rangos esperados ([-1, 1] para índices)
 - ✅ Alineados con DEM grid
 - ✅ No OOM con --max-memory 4G

 ---
 Notas de implementación

 1. Bandas "sintéticas" en MVP: Para no depender de conexión STAC en tests, generar bandas sintéticas (rasters aleatorios) en test
 2. Cloud masking: Ya está incluido en handle_stac_composite() via SCL layer
 3. Alineamiento: Ya está en STAC composite via --align-to
 4. Resampling: B02/B03/B04/B08 @ 10m alineadas directo; B11/B12 @ 20m se resamplearán automático nearest-neighbor
 5. Memoria: 6 bandas × 5k×3.6k × 8 bytes = ~864MB → dentro de --max-memory 4G
 6. Índices: Todos ya existen, solo iterar y guardar

 Dependencies

 Todas las funciones ya existen:
 - ✅ surtgis_algorithms::imagery::ndvi() y 14 más
 - ✅ handle_stac_composite() para descargas
 - ✅ resample_to_grid() para alineamiento
 - ✅ write_result() para outputs
