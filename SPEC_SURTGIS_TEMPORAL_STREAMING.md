# SPEC — Analítica temporal *streaming* sobre `Cube` / STAC

> **Propósito:** convertir los reducers temporales que hoy operan en memoria
> (`algorithms::temporal::{trend, phenology, anomaly, statistics}`) en un motor
> que consuma una **serie temporal STAC completa por chunks**, con la RAM
> acotada por construcción (patrón R9), y la exponga como un comando de una línea:
> "dame el mapa de tendencia de NDVI de esta cuenca en 5 años".
> **Estado:** propuesta de diseño (2026-07-12), **verificada contra código real
> y corregida el 2026-07-16** (§1, §2.1, §2.3 — ver notas inline). Supuestos
> corregidos: los reducers usan `f64` (no `f32`) y convención NaN (no máscara
> `bool`); el trait opera per-pixel sobre un núcleo que hay que extraer de los
> reducers actuales (no son wrappers gratis). **Baseline objetivo:** SurtGIS
> v0.19.0. **NO es implementación** — es el contrato para el trait
> `TemporalReducer` + `CubeStreamer`.

---

## 0. Por qué esta feature (y por qué es barata)

La mayoría del trabajo **ya está hecho**, sólo desconectado:

- `core::Cube<T>` + `CubeChunk` ya existen, con `chunks(chunk_rows)`,
  `pixel_series()`, `band_values()`, `into_slices()` y `CubeChunk::view()`.
- `algorithms::temporal` ya tiene `trend`, `phenology`, `anomaly`, `statistics`
  — pero asumen los arrays completos en memoria.
- El crate `cloud` ya sabe: bajar por STAC, decode con **byte-budget** (R9),
  `MemoryBudget`, `CompositeEngine` y `StreamingTiffSink`.

El salto killer no es escribir algoritmos nuevos: es **enchufar los reducers
existentes a un iterador de chunks alimentado por STAC**, de modo que la serie
temporal nunca se materialice entera. Es la diferencia entre "cabe si tienes
128 GB" y "corre en un laptop sobre 5 años de Sentinel-2".

---

## 1. Verificar primero (no asumir) — ✅ verificado 2026-07-16 contra el código real

- [x] Firma real de cada reducer en `temporal/*.rs`: **NO** operan sobre
      `&[f32]` de un pixel. Operan sobre **stacks completos de rasters**,
      `&[&Raster<f64>]` (ej. `linear_trend`, `mann_kendall`, `sens_slope` en
      `trend.rs`; `temporal_mean`/`std`/`min`/`max`/`percentile` en
      `statistics.rs`; `temporal_anomaly` en `anomaly.rs`;
      `vegetation_phenology` en `phenology.rs`). Internamente cada uno
      itera pixel a pixel y arma la serie con un helper interno tipo
      `collect_valid(rasters, row, col) -> Vec<f64>` — ese núcleo per-pixel
      existe pero no está expuesto como API pública. **Decisión (2026-07-16):
      se refactorizan los reducers para exponer ese núcleo per-pixel en
      `f64`**, y el trait streaming envuelve ese núcleo (no los wrappers
      "delgados" que asumía la v1 de esta SPEC — ver §2.1 corregida).
- [x] `Cube::pixel_series(row, col, band)` — confirmado, existe en
      `core/src/cube.rs:197`, devuelve un iterador de la serie alineada en
      orden temporal. **Ojo**: deja pasar `NaN` sin filtrar (no hace su propio
      manejo de nodata), igual que el resto del código.
- [x] `Cube::chunks(chunk_rows)` — confirmado, cada `CubeChunk` trae **todas
      las fechas/bandas** de una franja de filas (franja externa, tiempo
      interno). Test `chunks_cover_all_rows_aligned` en `cube.rs` lo verifica.
      Orden correcto para el budget: se puede descartar la franja completa
      (todas las fechas) antes de pasar a la siguiente.
- [x] Cliente STAC (`cloud/src/stac_client.rs`) — `search_all()` **sí** entrega
      `Vec<StacItem>` con `datetime` preservado por escena. Pero el único
      orquestador productivo (`CompositeEngine::run`) colapsa todo a un
      raster compuesto por banda; no existe hoy nada que entregue un `Cube`
      con el eje temporal intacto. Confirma que `StacCubeSource` es trabajo
      nuevo real, no un rewire.
- [x] Manejo de nodata / nubes por fecha: **NO** hay máscara `bool` explícita
      ni `Option<f32>` en ningún reducer actual. El patrón uniforme en todo
      el código es **`f64::NAN` + `.is_finite()`** para descartar muestras
      inválidas (`collect_valid` en `statistics.rs`/`anomaly.rs`,
      `Cube::pixel_series` deja pasar NaN tal cual). El trait streaming debe
      seguir esta convención en vez de introducir un `mask: &[bool]` nuevo
      (ver §2.1 corregida) — máscaras de nube, si existen, se aplican aguas
      arriba (en `MaskApplier` del pipeline de composite) escribiendo NaN,
      no se pasan como parámetro separado al reducer.

Nota adicional no prevista en la v1: el doc-comment de `core/src/cube.rs`
declara explícitamente que el análisis temporal (regresiones, detección de
quiebres) *no* debe vivir en `core` — debe vivir en un crate consumidor. Esto
**no** contradice esta SPEC: `TemporalReducer`/`reduce_temporal` ya estaban
planeados para `crates/algorithms` (no `core`), que es justamente ese
consumidor dentro del propio motor SurtGIS. Sin conflicto, solo se deja
registrado para no reabrir la pregunta más adelante.

---

## 2. Modelo de datos y API propuesta

### 2.1 Trait — `TemporalReducer` — ⚠️ corregida 2026-07-16 (v1 asumía series `f32` + máscara `bool`; el código real usa `f64` + convención NaN, ver §1)

En `crates/algorithms/src/temporal/mod.rs`:

```rust
/// Reduce la serie temporal de UN pixel a uno o varios valores de salida.
/// El orquestador provee valores + timestamps ya alineados; las muestras
/// inválidas llegan como NaN (misma convención que el resto de `temporal/*.rs`
/// y de `Cube::pixel_series`) — NO se pasa una máscara `bool` separada.
pub trait TemporalReducer: Send + Sync {
    /// Nombres de las bandas de salida (ej. ["slope", "intercept", "pvalue"]).
    fn outputs(&self) -> &[&str];
    /// Reduce una serie de un pixel. `values[i].is_nan()` ⇒ muestra inválida;
    /// el reducer es responsable de filtrar (mismo patrón que `collect_valid`
    /// en `statistics.rs`/`anomaly.rs` hoy).
    fn reduce(&self, values: &[f64], times: &[i64]) -> SmallVec<[f64; 4]>;
    /// Nº mínimo de muestras válidas para producir salida (si no, nodata).
    fn min_valid(&self) -> usize { 3 }
}
```

**Precondición de refactor (no es trabajo gratis, corrige el "wrapping delgado"
de la v1 de esta SPEC)**: los reducers actuales (`linear_trend`, `mann_kendall`,
`sens_slope` en `trend.rs`; `temporal_mean`/`std`/`min`/`max`/`percentile` en
`statistics.rs`; `temporal_anomaly` en `anomaly.rs`; `vegetation_phenology` en
`phenology.rs`) toman `&[&Raster<f64>]` completos, no series de un pixel. Cada
uno ya tiene internamente un núcleo per-pixel (tipo `collect_valid` +
matemática sobre `Vec<f64>`) pero no está expuesto. Antes de escribir los
adaptadores hay que **extraer ese núcleo a una función pública `f64`-en
`f64`-out** por reducer, y que tanto la función de stack completo (API actual,
sin romperla) como el nuevo adaptador `TemporalReducer` la reusen.

Adaptadores delgados (ahora sí delgados, una vez hecha la extracción):
`TheilSenTrend`, `MannKendall`, `PhenologyMetrics` (SOS/EOS/peak), `ZScoreAnomaly`,
`HarmonicFit` (amplitud/fase estacional). Cada uno = wrapper sobre el núcleo
per-pixel extraído, sin reimplementar la matemática.

### 2.2 Orquestador — `CubeStreamer`

```rust
/// Aplica un `TemporalReducer` a un Cube chunk-por-chunk, produciendo un
/// raster multibanda (una banda por output del reducer). RAM acotada por
/// `chunk_rows`, no por el nº de fechas ni el tamaño total.
pub fn reduce_temporal<R: TemporalReducer>(
    cube: &Cube<f64>,        // f64, no f32 — matchea el dtype real de todos
                              // los reducers existentes en temporal/*.rs (ver §1)
    reducer: &R,
    chunk_rows: usize,
) -> Result<Vec<Raster<f64>>>;
```

Paraleliza filas dentro del chunk con `maybe_rayon`. Salida vía
`StreamingTiffSink` (R9) para no bufferear las bandas de salida.

### 2.3 Fuente streaming — `CubeSource` (clave para STAC grande)

`Cube` en memoria sirve para el caso chico y para tests. Para 5 años de S2 hace
falta una **fuente perezosa** que produzca `CubeChunk` bajo demanda desde STAC,
reusando el decode byte-budget de R9:

```rust
pub trait CubeSource {
    fn shape(&self) -> (usize, usize);
    /// Añadido en la implementación (2026-07-16, no estaba en la v1 de esta
    /// SPEC): sin esto `reduce_temporal` no puede georreferenciar su salida
    /// a partir de un `CubeSource` genérico.
    fn transform(&self) -> GeoTransform;
    fn times(&self) -> &[i64];
    /// Entrega la franja [row0, row0+rows) con TODAS las fechas alineadas.
    fn chunk(&self, row0: usize, rows: usize) -> Result<CubeChunk<'_, f64>>;
}
```

Impl `StacCubeSource` en `cloud`: resuelve escenas fechadas vía
`StacClient::search_all()` (ya existe y preserva `datetime` por item, ver §1)
por bbox+colección, reproyecta/alinea a la grilla objetivo, aplica el
enmascarado de nubes por fecha escribiendo NaN (mismo mecanismo que
`MaskApplier` en el pipeline de composite hoy — no una máscara `bool`
separada), y decodifica solo la franja pedida dentro del `MemoryBudget`.
`Cube` en memoria implementa el mismo trait trivialmente.

---

## 3. Superficie (CLI / Python)

- **CLI:**
  `surtgis temporal trend --collection sentinel-2-l2a --bbox <…> --index ndvi
  --start 2020-01-01 --end 2025-01-01 --method theil-sen --out trend.tif`
  (bandas de salida: slope, intercept, pvalue). Verbos hermanos:
  `temporal phenology`, `temporal anomaly`, `temporal harmonic`.
- **Python:** `surtgis.temporal_reduce(cube_or_source, reducer="theil-sen",
  chunk_rows=256) -> dict[str, ndarray]`. Acepta un `Cube` armado desde numpy o
  un `StacCubeSource` configurado.

## 4. Modelo de memoria

- Pico `O(chunk_rows · n_cols · n_times · 4 bytes + salida de chunk)`; **no**
  escala con el total de la serie ni con el bbox completo.
- `chunk_rows` es el dial RAM↔I/O (análogo a `--band-chunk-size` del composite).
- Documentar el pico y un test nightly (como `stac-ram-bench`) que asserte
  ausencia de crecimiento monótono sobre una serie larga.

## 5. Criterios de aceptación (Definition of Done del scaffold) — ✅ scaffold completo 2026-07-16

- [x] `TemporalReducer` + `reduce_temporal` (cuerpo real, genérico sobre
      `CubeSource` — no solo in-memory `Cube`) + `TheilSenTrend` adaptando
      `sens_slope_series`, el núcleo per-pixel extraído de `sens_slope`
      (`crates/algorithms/src/temporal/streaming.rs`, `trend.rs`).
- [x] `CubeSource` trait definido en `core::cube` (con `transform()`
      añadido, ver §2.3); `Cube<f64>` lo implementa (rechaza cubos
      multibanda); `StacCubeSource` = stub documentado (`todo!()`) en
      `crates/cloud/src/stac_cube_source.rs`, gateado tras `unstable`.
- [x] Test analítico: cube sintético con tendencia lineal conocida → Theil-Sen
      recupera la pendiente exacta (`theil_sen_recovers_exact_slope`);
      streaming (chunk_rows=1) vs in-memory `sens_slope` comparado bit a bit
      vía `to_bits()` (`streaming_matches_in_memory_bit_identical`) — también
      verificado a nivel CLI end-to-end (`temporal trend --method theil-sen`
      vs `--method sens`, `np.array_equal` sobre los GeoTIFF de salida).
- [x] Test de gaps: `gaps_respect_min_valid` (1 muestra válida → NaN, no hay
      par para pendiente) y `gaps_still_recover_slope_with_enough_valid_samples`
      (2 fechas con nube/NaN de 6 → pendiente exacta igual) — convención NaN,
      no máscara `bool` (ver corrección §1/§2.1).
- [x] CLI: `surtgis temporal trend --method theil-sen` (alias `streaming`)
      registrado y funcional sobre un stack local de GeoTIFFs fechados
      (construye un `Cube`, llama `reduce_temporal`) — no solo un stub,
      valida end-to-end el criterio "corre sobre cualquier `CubeSource`,
      incluido GeoTIFFs locales" del §6. CI verde: `cargo fmt --check` limpio,
      853 tests `algorithms` + 160 `core` en verde, `cargo build --all-features`
      en `cloud` compila el stub `StacCubeSource` sin warnings nuevos.

**Fuera del scaffold** (siguiente iteración, no bloquea este cierre):
`StacCubeSource::new`/`chunk` siguen siendo `todo!()` — la superficie CLI/Python
con `--collection`/`--bbox`/`--start`/`--end` de §3 depende de esa
implementación real y no se construyó todavía.

## 6. No-objetivos

- **NO** reimplementar la matemática temporal — sólo envolver `temporal/*.rs`.
- **NO** meter análisis temporal conceptual pesado (BFAST completo, LandTrendr):
  eso puede vivir en **datacube-rs**; SurtGIS aporta el **contenedor + iterador
  alineado + reducers base**, exactamente como dice el ecosystem SPEC (P1.2).
- **NO** obligar STAC: el motor corre sobre cualquier `CubeSource`, incluido un
  stack local de GeoTIFFs fechados.

## 7. Preguntas abiertas

- ¿El eje de reducción es siempre t, o se quiere reducir sobre (t, band) juntos
  (ej. índice derivado on-the-fly antes de reducir)? Probable: permitir un
  `pre-map` band→escalar por fecha (calcular NDVI por escena y luego reducir).
- ¿Alinear fechas irregulares a una grilla temporal, o pasar timestamps crudos al
  reducer? (Theil-Sen/MK toleran irregular; harmonic necesita el t real — pasar
  crudo y que cada reducer decida).

## 8. Consumidores

datacube-rs (núcleo), la monitorización multi-temporal de Ñuble (tendencia /
anomalía de vegetación y agua), unmix-rs, y cualquier flujo que hoy baja un
composite pero querría la **dinámica**, no solo el snapshot.
