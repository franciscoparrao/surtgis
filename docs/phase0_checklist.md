# Fase 0: Merge y Limpieza Pre-Release — Checklist

> Estado: **En progreso**
> Fecha inicio: 2026-02-04
> Objetivo: `main` en estado publicable, 0 warnings, README+CHANGELOG+LICENSE actualizados

---

## 0.1 Limpieza de código

- [x] Fix clippy en `surtgis-colormap` (2 errores: manual_clamp, collapsible_if)
- [x] Fix clippy en `surtgis-cloud` (14 errores: div_ceil, collapsible_if, extend→append)
- [x] Fix clippy en `surtgis-algorithms` (20 errores: needless_range_loop, dead_code, manual contains, derivable impl)
- [x] Fix clippy en `surtgis-gui` (~50 errores: dead_code, clone on Copy, collapsible_if, struct update, div_ceil)
- [x] Fix clippy en `surtgis-python` (CurvatureFormula import, GeoTransform dereference)
- [x] `cargo clippy --workspace -- -D warnings` pasa limpio (0 errores)
- [ ] Eliminar `#[allow(dead_code)]` innecesarios después de refactoring
- [ ] Revisar `cargo clippy --workspace -- -W clippy::pedantic` (informativo, no bloquea)

## 0.2 Archivos de proyecto

- [x] Crear `LICENSE-MIT` en raíz del proyecto
- [x] Crear `LICENSE-APACHE` en raíz del proyecto
- [x] Crear `CHANGELOG.md` (formato Keep a Changelog)
- [x] Actualizar `README.md` (105 algoritmos, 9 crates, 9 categorías, GUI, WASM, Cloud)
- [x] Verificar `license = "MIT OR Apache-2.0"` en todos los `Cargo.toml` (workspace + 9 crates via `license.workspace = true`)
- [x] Corregir author: "Francisco Parrao" → "Francisco Parra" en `Cargo.toml` raíz

## 0.3 Tests y validación

- [x] `cargo test --workspace` — todos los tests pasan (637 passed, 0 failed)
- [x] `cargo build --workspace --release` — compila sin errores (7m22s)
- [x] `cargo build -p surtgis-gui --release` — GUI compila (incluido en workspace release)
- [x] `cargo build -p surtgis-wasm --target wasm32-unknown-unknown` — WASM compila (fix: `IntoParallelRefMutIterator` en maybe_rayon)
- [x] Verificar que no hay fixtures/datos de test > 1 MB en git (tests/fixtures, output, biblio en .gitignore)
- [x] Verificar que no hay secretos/credenciales en el repositorio

## 0.4 Merge a main

- [x] Revisar diferencias `main` vs `feature/gui-desktop` (13 commits, 203 archivos, fast-forward)
- [x] Sin conflictos (main no tenía commits divergentes)
- [x] Merge con `--no-ff` (preserva historia de commits)
- [x] `cargo test --workspace` en main post-merge — 637 passed, 0 failed
- [ ] Tag `v0.1.0-rc1` (release candidate)

## 0.5 Verificación post-merge

- [x] `cargo clippy --workspace -- -D warnings` en main — 0 warnings
- [x] `cargo test --workspace` en main — 637 passed, 0 failed
- [ ] `cargo doc --workspace --no-deps` genera sin warnings
- [ ] README renderiza correctamente en GitHub
- [ ] CHANGELOG refleja el estado actual

---

# Checklist General del Proyecto (Fases 0-6)

## Estado por Fase

| # | Fase | Estado | Bloqueado por |
|---|------|--------|---------------|
| 0 | Merge y limpieza | **En progreso** | — |
| 1 | Publicar registros | Pendiente | Fase 0 |
| 2 | Paper académico | Pendiente | Fase 1.1 |
| 3 | Python usable | Pendiente | Fase 1.3 |
| 4 | Documentación | Pendiente | Fase 1 |
| 5 | LiDAR pipeline | Pendiente | — (independiente) |
| 6 | GPU selectivo | Pendiente | — (independiente) |

## Fase 1: Publicar en Registros

### 1.1 crates.io
- [ ] Metadata completa en cada Cargo.toml (description, keywords, categories, repository)
- [ ] `cargo publish --dry-run` exitoso para cada crate
- [ ] Publicar en orden: core → colormap → parallel → algorithms → cloud → cli
- [ ] Verificar docs.rs post-publish
- [ ] `cargo install surtgis` funcional

### 1.2 npm
- [ ] `wasm-pack build --target bundler --release` exitoso
- [ ] `npm publish` en npmjs.com
- [ ] Test de importación en Vite + vanilla TS

### 1.3 PyPI
- [ ] Configurar maturin + pyproject.toml
- [ ] Build wheels multiplataforma (linux, macos, windows)
- [ ] Publicar en TestPyPI primero
- [ ] `pip install surtgis` funcional

## Fase 2: Paper Académico

- [ ] Borrador redactado (estructura: introduction, architecture, algorithms, benchmarks, WASM, cloud, case study)
- [ ] Benchmarks formales (3 DEMs × 5 algos × 5 herramientas × 10 repeticiones)
- [ ] Benchmarks WASM vs servidor (browser Chrome/Firefox)
- [ ] Cross-validation extendida (RMSE, MAE, correlación)
- [ ] Caso de estudio reproducible (notebook)
- [ ] Submission a JOSS o SoftwareX

## Fase 3: Python Usable

- [ ] API Pythonic: `surtgis.Raster` con from_file, save, to_numpy, from_numpy
- [ ] Funciones de alto nivel: `surtgis.slope()`, `surtgis.twi()`, etc.
- [ ] Integración xarray/rioxarray
- [ ] 4 notebooks de ejemplo (terrain, hydrology, spectral, classification)
- [ ] Benchmark surtgis vs rasterio+richdem

## Fase 4: Documentación Pública

- [ ] docs.rs completo (módulos documentados, ejemplos en docstrings)
- [ ] Landing page / GitHub Pages
- [ ] Documentación JavaScript/TypeScript
- [ ] CONTRIBUTING.md + issue templates

## Fase 5: LiDAR Pipeline

- [ ] Lector LAS/LAZ nativo
- [ ] Clasificación de suelo (PMF)
- [ ] Rasterización punto → DEM
- [ ] Pipeline CLI + GUI

## Fase 6: GPU Selectivo

- [ ] Crate surtgis-gpu (slope/hillshade en WGSL)
- [ ] Focal statistics genérico en GPU
- [ ] WebGPU en browser

---

## Métricas de Éxito

| Métrica | Target | Actual |
|---------|--------|--------|
| Clippy warnings | 0 | **0** |
| Tests passing | 637+ | 637 |
| Algoritmos GUI | 105 | 105 |
| Categorías | 9 | 9 |
| Crates | 9 | 9 |
| LICENSE files | 2 | 2 |
| CHANGELOG | 1 | 1 |
| README actualizado | Si | Si |
| Paper draft | 1 | 0 |
| crates.io publicados | 6 | 0 |
| npm publicado | 1 | 0 |
| PyPI publicado | 1 | 0 |
| GitHub stars | >100 | 0 |
