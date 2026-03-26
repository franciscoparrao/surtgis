# SurtGIS — Roadmap from Production Feedback

**Fuente**: Sesión de postdoctorado procesando 15 cuencas de Chile (Marzo 2026)
**Estado**: Feedback recibido después de procesar 472M píxeles en producción

## Pendientes por prioridad

### P1: Integration tests con cuenca real
- **Estado**: En progreso
- Test suite con una cuenca pequeña (ej. Río Salado, ~7500 km²)
- Descarga datos reales de STAC (DEM + S2)
- Verifica: CRS propagation, SCL resampleo, extents diferentes, grid alignment, cloud-mask
- Atrapa los bugs que los unit tests no capturan

### P2: Reportes de progreso informativos
- Cuando `stac composite` tarda 15 min, mostrar:
  - Tiempo estimado restante (basado en strips procesados)
  - MB descargados / total estimado
  - % de cobertura clear del composite parcial
- Genera confianza de que no se colgó

### P3: Memory cap / auto-detect streaming
- `--max-memory 4G` que fuerce streaming automático
- O auto-detect basado en RAM disponible (`sysinfo` crate)
- Safety net para usuarios menos técnicos
- El streaming ya resuelve el 90%, esto es el 10% restante

### P4: Pipeline susceptibility end-to-end
```bash
surtgis pipeline susceptibility \
  --dem copernicus-dem-glo-30 \
  --optical sentinel-2-l2a \
  --bbox ... --datetime ... \
  --outdir ./factors/
```
- Internaliza los 3 scripts bash que la otra sesión usa
- Requiere que el caso de uso se estabilice primero
- Depende de: ML nativo (logistic regression + random forest)
- **Después del paper**

### P5: Tutorial "Susceptibility mapping at national scale"
- Basado en el caso real: 15 cuencas de Chile
- DEM + 25 terrain + 8 hydro + 10 bandas S2 + 16 índices
- Más valioso que 20 páginas de API reference
- **Cuando publiquen resultados del postdoctorado**

### P6: ML nativo (Logistic Regression + Random Forest)
- Zero Python pipeline: terrain → hydrology → imagery → train → predict
- Paper potencial: "End-to-end susceptibility mapping from a single CLI"
- Estimación: ~2 semanas
- **Después del paper EMS**

## Ya resueltos en esta sesión

- [x] CRS no propagado en outputs → `read_crs()` en native.rs
- [x] Asset alias PC/ES → `resolve_asset_key()` bidireccional
- [x] SCL resampleo 20m→10m → nearest-neighbor en cloud_mask_scl
- [x] Extents diferentes en median composite → union bbox path
- [x] Grid alignment DEM vs S2 → `--align-to` + `resample_to_grid()`
- [x] Compresión DEFLATE en streaming writer
- [x] CLI monolítico → 14 módulos
- [x] Unsafe en core algos → safe indexing (zero overhead)
- [x] 4/5 CVEs parcheados → `cargo update`
