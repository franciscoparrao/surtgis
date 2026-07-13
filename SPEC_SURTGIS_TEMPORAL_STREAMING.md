# SPEC — Analítica temporal *streaming* sobre `Cube` / STAC

> **Propósito:** convertir los reducers temporales que hoy operan en memoria
> (`algorithms::temporal::{trend, phenology, anomaly, statistics}`) en un motor
> que consuma una **serie temporal STAC completa por chunks**, con la RAM
> acotada por construcción (patrón R9), y la exponga como un comando de una línea:
> "dame el mapa de tendencia de NDVI de esta cuenca en 5 años".
> **Estado:** propuesta de diseño (2026-07-12). **Baseline objetivo:** SurtGIS
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

## 1. Verificar primero (no asumir)

- [ ] Firma real de cada reducer en `temporal/*.rs`: ¿operan sobre
      `&[f32]` (serie de un pixel), sobre `ArrayView`, o sobre `Raster`
      completos? Esto decide si el trait envuelve *serie-por-pixel* o
      *reduce-de-chunk*.
- [ ] `Cube::pixel_series(row, col, band)` — ¿devuelve la serie temporal de un
      pixel ya alineada? Si sí, el trait natural es `reduce(&[f32], &[i64]) -> T`
      (valores + timestamps).
- [ ] `Cube::chunks(chunk_rows)` — ¿cada `CubeChunk` trae **todas las fechas**
      de una franja de filas? Confirmar el orden de iteración (t externo vs
      franja externa) para no romper el budget.
- [ ] ¿El cliente STAC del crate `cloud` puede entregar una **lista de escenas
      fechadas** (no solo un composite colapsado)? Si hoy solo compone, hace
      falta un modo "serie" que preserve el eje t.
- [ ] Manejo de nodata / nubes por fecha: ¿los reducers reciben una máscara o un
      `Option<f32>` por muestra? El contrato debe ser explícito (gaps son la regla
      en series ópticas).

---

## 2. Modelo de datos y API propuesta

### 2.1 Trait — `TemporalReducer`

En `crates/algorithms/src/temporal/mod.rs`:

```rust
/// Reduce la serie temporal de UN pixel a uno o varios valores de salida.
/// El orquestador provee la serie ya alineada (valores + timestamps + máscara).
pub trait TemporalReducer: Send + Sync {
    /// Nombres de las bandas de salida (ej. ["slope", "intercept", "pvalue"]).
    fn outputs(&self) -> &[&str];
    /// Reduce una serie de un pixel. `mask[i] == false` ⇒ muestra inválida.
    fn reduce(&self, values: &[f32], times: &[i64], mask: &[bool]) -> SmallVec<[f32; 4]>;
    /// Nº mínimo de muestras válidas para producir salida (si no, nodata).
    fn min_valid(&self) -> usize { 3 }
}
```

Adaptadores delgados que envuelven lo ya escrito:
`TheilSenTrend`, `MannKendall`, `PhenologyMetrics` (SOS/EOS/peak), `ZScoreAnomaly`,
`HarmonicFit` (amplitud/fase estacional). Cada uno = wrapper sobre la función
existente en `temporal/*.rs`, sin reimplementar la matemática.

### 2.2 Orquestador — `CubeStreamer`

```rust
/// Aplica un `TemporalReducer` a un Cube chunk-por-chunk, produciendo un
/// raster multibanda (una banda por output del reducer). RAM acotada por
/// `chunk_rows`, no por el nº de fechas ni el tamaño total.
pub fn reduce_temporal<R: TemporalReducer>(
    cube: &Cube<f32>,        // o un CubeSource streaming (ver 2.3)
    reducer: &R,
    chunk_rows: usize,
) -> Result<Vec<Raster<f32>>>;
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
    fn times(&self) -> &[i64];
    /// Entrega la franja [row0, row0+rows) con TODAS las fechas alineadas.
    fn chunk(&self, row0: usize, rows: usize) -> Result<CubeChunk<'_, f32>>;
}
```

Impl `StacCubeSource` en `cloud`: resuelve escenas fechadas por bbox+colección,
reproyecta/alinea a la grilla objetivo, aplica máscara de nubes por fecha, y
decodifica solo la franja pedida dentro del `MemoryBudget`. `Cube` en memoria
implementa el mismo trait trivialmente.

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

## 5. Criterios de aceptación (Definition of Done del scaffold)

- [ ] `TemporalReducer` + `reduce_temporal` (cuerpo real, in-memory `Cube`) +
      al menos `TheilSenTrend` adaptando la función existente.
- [ ] `CubeSource` trait definido; `Cube` lo implementa; `StacCubeSource` = stub
      documentado (`todo!()`) en `cloud`.
- [ ] Test analítico: cube sintético con tendencia lineal conocida → Theil-Sen
      recupera la pendiente exacta; comparar streaming vs in-memory (bit-idéntico).
- [ ] Test de gaps: serie con nodata/nubes → respeta `min_valid` y máscara.
- [ ] CLI `surtgis temporal trend` registrada (handler stub). CI verde.

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
