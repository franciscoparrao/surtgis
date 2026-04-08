# Soporte Zarr/NetCDF para Datasets Climáticos — Roadmap

> Feature Request: 2026-04-07
> Status: Planificación

## Contexto

SurtGIS solo lee COGs. Tres colecciones climáticas clave en Planetary Computer usan formatos incompatibles:

| Colección | Uso | Formato | Fase |
|-----------|-----|---------|------|
| era5-pds | Reanalysis histórico (precip, temp, 1979-hoy) | Zarr | 1 |
| terraclimate | Clima mensual (~4km, 1958-2021) | Zarr/NetCDF | 1 |
| nasa-nex-gddp-cmip6 | Proyecciones SSP (26 GCMs, 2015-2100) | NetCDF | 4 |

## Fases

### Fase 1: ZarrReader con `zarrs` crate
- [ ] Agregar `zarrs` + `zarrs_object_store` como dependencias feature-gated (`zarr`)
- [ ] Crear `crates/cloud/src/zarr_reader.rs` — ZarrReader struct
  - [ ] Abrir Zarr v2/v3 stores (HTTP, Azure Blob via `object_store`)
  - [ ] Detección automática de formato (Zarr vs COG) por STAC asset media type
  - [ ] Parsing de CF Conventions (time axis, lat/lon, variable names)
  - [ ] Subsetting por bbox (lat/lon → chunk ranges)
  - [ ] Subsetting por time slice / time range
  - [ ] Devolver `Raster<f64>` compatible con pipeline existente
  - [ ] Manejar CRS (climate data suele ser WGS84/EPSG:4326)
- [ ] Integrar con STAC pipeline
  - [ ] `stac_reader.rs`: detectar asset type → dispatch a CogReader o ZarrReader
  - [ ] Signing: Azure Blob SAS para Planetary Computer Zarr stores
- [ ] CLI: `surtgis stac fetch` y `stac composite` aceptan assets Zarr
- [ ] Tests con ERA5-PDS y TerraClimate reales

### Fase 2: Trait `CloudRasterReader`
- [ ] Definir trait abstracto para lectura cloud
  ```rust
  pub trait CloudRasterReader: Send + Sync {
      async fn read_bbox(&self, bbox: &BBox, options: &ReadOptions) -> Result<Raster<f64>>;
      fn metadata(&self) -> &RasterMetadata;
  }
  ```
- [ ] Implementar para `CogReader`
- [ ] Implementar para `ZarrReader`
- [ ] Refactorizar `stac_reader.rs` para usar trait en vez de struct concreto
- [ ] Refactorizar composite handler para ser format-agnostic

### Fase 3: CLI `stac download-climate`
- [ ] Nuevo subcomando con flags climate-specific:
  - `--variable` (pr, tas, hurs, sfcWind...)
  - `--model` (MIROC6, MPI-ESM1-2-HR, multi-model ensemble)
  - `--scenario` (historical, ssp245, ssp585)
  - `--aggregate` (yearly-mean, monthly-mean, seasonal)
- [ ] Multi-model ensemble: descargar N modelos → estadísticas (mean, std, percentiles)
- [ ] Agregación temporal: reducir time series a estadístico por período
- [ ] Exportar como GeoTIFF (1 archivo por período/modelo)

### Fase 4: NetCDF reader
- [ ] Feature-gated: `netcdf` (georust bindings) para targets nativos
- [ ] Evaluar `netcdf-reader` / `hdf5-pure` cuando maduren (para WASM)
- [ ] Implementar `CloudRasterReader` para NetCdfReader
- [ ] Tests con NASA NEX-GDDP-CMIP6

## Crates evaluados

| Crate | Versión | Madurez | WASM | Cloud | Decisión |
|-------|---------|---------|------|-------|----------|
| `zarrs` | 0.23.9 | Excelente (130K dl) | Sí | Sí (object_store) | **Usar en Fase 1** |
| `zarrs_object_store` | 0.6.2 | Buena | Sí | S3/Azure/GCS/HTTP | **Usar en Fase 1** |
| `netcdf` (georust) | 0.12.0 | Buena (257K dl) | No (libnetcdf-c) | No | Fase 4 (native only) |
| `netcdf-reader` | 0.2.0 | Muy nueva (249 dl) | Dudoso | No | Monitorear |
| `hdf5-pure` | 0.1.1 | Experimental | Sí | No | Monitorear |

## Cobertura esperada

| Dataset | Formato | Fase 1 | Fase 4 |
|---------|---------|:------:|:------:|
| ERA5-PDS | Zarr | **Sí** | - |
| TerraClimate | Zarr/NetCDF | **Sí** (Zarr) | Sí |
| NEX-GDDP-CMIP6 | NetCDF | Parcial* | **Sí** |

*CMIP6 disponible como Zarr en Pangeo STAC (no PC), lo que amplía cobertura de Fase 1.
