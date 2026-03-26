# Code Review Checklist — SurtGIS

**Fecha**: 2026-03-21
**Calidad general**: Buena | **Deuda técnica**: Baja

## Hallazgos

- [x] **[HIGH] H1**: Descomponer `main.rs` (3,752→35 líneas) en 14 módulos — commit `6368948`
- [x] **[HIGH] H2**: Migrar unsafe a safe indexing en 6 core algos (48 bloques, zero perf impact) — commit `6ea2ec9`. 111 restantes en algos secundarios
- [ ] **[MEDIUM] M1**: Strip writer acumula todo en memoria por limitación de `tiff` crate (fix upstream o flate2)
- [ ] **[MEDIUM] M2**: `fetch_stac_asset` dead code — refactorizar para que `FetchMosaic` lo use
- [ ] **[MEDIUM] M3**: 13 variables de una letra en `terrain all` — renombrar
- [ ] **[MEDIUM] M4**: `SceneInfo.date` no usado — usar en output de progreso
- [ ] **[LOW] L1**: 4 `.unwrap()` en main.rs (L3677, L3687, L3723) — reemplazar por manejo explícito
- [ ] **[LOW] L2**: GUI crate tiene 22 `#[allow(dead_code)]` — limpiar cuando GUI madure
- [ ] **[LOW] L3**: `vector/mod.rs` TODO vacío — eliminar o implementar

## Arquitectura: OK
- DAG unidireccional, core como raíz, 0 circular deps, 0 warnings, 595 tests passing
