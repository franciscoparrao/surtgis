# SPEC — Inferencia de modelos (ONNX, pure-Rust) sobre raster

> **Propósito:** dotar a SurtGIS de inferencia *tiled + stitched* de modelos
> entrenados (segmentación / clasificación / regresión densa) **dentro del
> binario**, sin Python ni CUDA, cerrando el loop que hoy queda abierto:
> `extract-patches` produce los tensores de entrenamiento pero el modelo se
> corre afuera.
> **Estado:** propuesta de diseño (2026-07-12). **Baseline objetivo:** SurtGIS
> v0.19.0 (feature `onnx`, opcional; no altera el single-binary por defecto).
> **NO es implementación** — es el contrato para andamiar `algorithms::inference`.

---

## 0. Por qué esta feature (y por qué encaja)

SurtGIS ya tiene **la mitad del loop de deep learning geoespacial**:

- `extract-patches` (CLI `crates/cli/src/handlers/extract_patches.rs`) genera
  chips `[N, bands, H, W] f32` + labels + `manifest.csv` + `meta.json`, con
  **profiles de Foundation Model** (`gfm_profiles.rs`) y metadata **STAC MLM /
  ML-AOI** (`stac_writer.rs`). Es una salida CNN-ready de primer nivel.
- `predict_raster([band0, band1, …], callable)` en `crates/python/src/lib.rs`
  aplica un modelo **pixel-wise** (sklearn/XGBoost) llamando de vuelta a Python.

Lo que falta es **inferencia densa nativa**: correr un modelo convolucional (que
mira un vecindario, no un pixel) sobre un raster grande, en mosaico con halo,
recomponiendo la salida — sin salir de Rust y sin materializar todo en RAM.
[`tract`](https://github.com/sonos/tract) es un runtime **ONNX 100% Rust** (sin
C++/CUDA/onnxruntime), lo que preserva la propiedad single-binary y compila a
WASM. Nadie en el espacio "un binario, cero deps de sistema" ofrece esto.

**Diferenciador honesto:** no competimos con TensorRT en throughput; competimos
en *portabilidad y cero fricción* — un `.onnx` + un COG + un comando, en un
servidor sin GPU, en un Raspberry Pi o en el navegador.

---

## 1. Verificar primero (no asumir)

Antes de andamiar, confirmar contra el código actual:

- [ ] `TileGrid::new(rows, cols, tile_size, overlap)` y `Tile` exponen `core`
      vs `read` extent con `core_offset_row/col()` — confirmar que el halo es
      suficiente para el *receptive field* del modelo (halo ≥ (rf−1)/2).
- [ ] `Raster<f32>` multibanda: ¿el proyecto representa N bandas como
      `Vec<Raster<f32>>` (como `predict_raster`) o existe un raster multibanda?
      Definir la unidad de entrada del modelo en consecuencia.
- [ ] ¿`StripProcessor`/`WindowAlgorithm` sirve para el caso *tile 2-D con halo
      en ambos ejes*, o conviene iterar `TileGrid` directamente? (probable: el
      streaming por filas sirve para modelos 1-D de vecindario; para CNN 2-D usar
      `TileGrid`).
- [ ] `meta.json` de `extract-patches`: qué normalización/orden de bandas
      registra, para que inferencia consuma **el mismo contrato** (mean/std,
      band order, dtype). Reusar esa estructura, no inventar otra.
- [ ] Licencia y peso de `tract` en el árbol de deps; confirmar que queda tras
      `--features onnx` y NO entra al build por defecto ni a la wheel base.

Reportar hallazgos y ajustar el alcance antes de tocar código.

---

## 2. Modelo de datos y API propuesta

### 2.1 Trait de extensión — `TileModel`

Punto de enganche nuevo en `crates/algorithms/src/inference/mod.rs`:

```rust
/// Un modelo que transforma un tile multibanda (con halo) en un tile de salida.
/// El orquestador se encarga del mosaico, el halo y el stitching; el impl solo
/// ve un chip y devuelve un chip del mismo tamaño de *core*.
pub trait TileModel: Send + Sync {
    /// Nº de bandas de entrada que el modelo espera (para validar el stack).
    fn in_bands(&self) -> usize;
    /// Nº de bandas de salida (1 = clasificación/regresión densa; K = logits).
    fn out_bands(&self) -> usize;
    /// Halo mínimo requerido (celdas) = (receptive_field − 1) / 2.
    fn required_halo(&self) -> usize;
    /// Inferencia sobre un tile [in_bands, H+2·halo, W+2·halo] → [out_bands, H, W].
    fn infer(&self, tile: &TileInput) -> Result<TileOutput>;
}
```

> **Ajuste de diseño (encontrado al andamiar, no anticipado arriba):**
> `TileGrid` **clampa** (no rellena con ceros) el halo en los bordes de la
> grilla, por lo que el halo puede ser **asimétrico** alrededor del core en
> los tiles de borde/esquina. Un modelo no puede inferir el tamaño del core
> de forma confiable a partir de `read_rows − 2·halo` (ese cálculo asume halo
> simétrico y falla exactamente en los bordes). Por eso `TileInput` expone
> explícitamente `core_offset_row/col` + `core_rows/cols` — la ubicación y
> tamaño del rectángulo de core dentro de `bands` — en vez de forzar a cada
> implementación de `TileModel` a re-derivarlo (e implementarlo mal, como
> pasó en el primer intento del modelo de prueba del scaffold). Esto también
> es lo que un backend ONNX real necesitaría de todos modos para hacer
> padding-a-tamaño-fijo antes de correr la red y recortar exactamente al
> core esperado después.

- Implementación concreta feature-gated: `OnnxModel` (envuelve `tract`), que
  carga un `.onnx`, infiere shape/rf desde el grafo y valida contra
  `in_bands/out_bands`.
- El **orquestador** `run_tiled(model, &[Raster<f32>], params) -> Raster<…>` NO
  depende de `tract`: recorre `TileGrid` (con `overlap = model.required_halo()`),
  arma el `TileInput`, llama `model.infer`, y escribe **solo el core** al raster
  de salida. Paralelizable con el patrón `maybe_rayon` existente.

### 2.2 Post-proceso

- `out_bands == 1` → raster de clase (u8/i16) o valor continuo (f32).
- `out_bands == K` → argmax opcional (`--argmax`) o escribir el cubo de logits
  (reusar `StreamingTiffSink` de R9 para no bufferear K bandas en RAM).
- Softmax/temperature opcionales como flags, no en el trait.

### 2.3 Contrato de normalización

Reusar la estructura de `meta.json` de `extract-patches`: mean/std por banda,
orden de bandas, dtype. Inferencia debe **rechazar** (error honesto) un stack
cuyo orden/dtype no matchee el `meta.json` asociado al modelo. Simetría exacta
entre entrenamiento y despliegue = menos bugs silenciosos.

---

## 3. Superficie (CLI / Python / WASM)

- **CLI:** `surtgis infer --model m.onnx --features slope.tif aspect.tif ndvi.tif
  --out pred.tif [--tile-size 512] [--halo N] [--argmax] [--profile <gfm>]`.
  `--out` es nombrado, no posicional — un posicional aquí sería ambiguo con la
  lista variádica `--features` (clap se la traga; confirmado al andamiar). Falta
  aún: autodescubrir bandas y normalización desde un `meta.json` hermano del
  modelo (§2.3) — el handler de hoy no lo lee, ver §5.
- **Python:** `surtgis.infer_raster(model_path, [b0, b1, …], tile_size=512,
  argmax=True) -> ndarray`. Convive con `predict_raster` (pixel-wise, callback
  Python) — este es el camino *denso, nativo, sin callback*.
- **WASM:** `infer_tile(model_bytes, tile_f32, shape)` a nivel de un tile
  (el navegador orquesta el mosaico). Habilita "corre tu modelo sobre este COG"
  en el visor sin backend.

---

## 4. Modelo de memoria

RAM acotada por construcción, siguiendo R9:

- Nunca cargar el raster completo: `TileGrid` da tiles de `tile_size²`; el pico
  es `O(tile_size² · (in_bands + out_bands) + memoria interna de tract)`.
- Salida vía `StreamingTiffSink` (ya existe) — no se bufferea el raster de salida.
- El grafo de tract se carga una vez y se reusa entre tiles (thread-safe).
- Documentar el pico como función de `tile_size` (dial RAM↔overhead de halo),
  igual que el `--band-chunk-size` del composite.

---

## 5. Criterios de aceptación (Definition of Done del scaffold)

> **Estado (2026-07-12): andamiaje base completo**, en
> `crates/algorithms/src/inference/{mod,onnx}.rs` +
> `crates/cli/src/handlers/infer.rs`. Verificado localmente: `cargo build`
> default y `--features onnx` compilan ambos limpio (solo warnings
> preexistentes, ninguno nuevo); `cargo clippy --features onnx -- -D
> clippy::correctness -W warnings` sin hallazgos nuevos; `cargo test -p
> surtgis-algorithms --lib inference::` 5/5 verde; smoke test end-to-end real
> (`surtgis infer --model /tmp/fake.onnx --features demo_dem.tif --out
> /tmp/out.tif`) lee el raster real y falla con el mensaje controlado en
> `OnnxModel::load`, sin panic.

- [x] Módulo `algorithms::inference` con `TileModel`, `TileInput/TileOutput`,
      `run_tiled` (cuerpo real) y `OnnxModel` con `infer` = `todo!()` documentado.
- [x] Firma CLI `surtgis infer` registrada (handler que hace la I/O real y
      retorna "not yet implemented" en `OnnxModel::load`, tras feature `onnx`).
- [x] Test de orquestación **sin tract**: `IdentityModel` (halo=0) +
      `BoxSumModel` (halo>0) demostrando halo + stitching exacto (sin costuras),
      verificado en **cada celda** — incluyendo bordes/esquinas con halo
      clampeado asimétrico — de un raster 64×71 con `TileGrid` tile_size=16
      (ragged, ≥2×2 tiles). Encontrado al escribir el test: un modelo no puede
      inferir el tamaño del core desde `read_rows - 2·halo` porque el halo se
      clampea (no se rellena con ceros) en los bordes — por eso `TileInput`
      ahora expone `core_offset_row/col` + `core_rows/cols` explícitos (ver
      nota de diseño en §2.1). Tests adicionales: rechazo de band-count
      incorrecto, de bands desalineados (`check_aligned`), y de un modelo que
      devuelve una shape de salida incorrecta.
- [ ] SPEC de normalización enlazada al `meta.json` de `extract-patches` (§2.3)
      — **documentado, no implementado**: el handler de hoy no lee
      `meta.json`/`GfmProfileSpec`/`apply_band_norm` (de `gfm_profiles.rs`) ni
      tiene `--profile`. Próximo paso natural, junto con `--argmax` real (hoy
      es un error explícito) y la carga real del grafo tract en
      `OnnxModel::load`.
- [x] `cargo build` por defecto NO compila `tract`; `--features onnx` sí. CI
      verde localmente (no se corrió el workflow de GitHub Actions en sí).

## 6. No-objetivos

- **NO** entrenar modelos (SurtGIS infiere; el training vive en Python/otros).
- **NO** soportar todo el opset ONNX — documentar el subconjunto validado
  (conv/pool/bn/relu/upsample típicos de segmentación) y fallar claro fuera de él.
- **NO** GPU en v1 (tract es CPU). Dejar `TileModel` agnóstico para que un backend
  wgpu futuro (ver SPEC de compute backend) implemente el mismo trait.
- **NO** obligatorio en el binario base ni en la wheel — siempre feature `onnx`.

## 7. Preguntas abiertas

- ¿Unidad de entrada: `Vec<Raster<f32>>` o un raster multibanda de primera clase?
  (afecta también a `composite`/`extract-patches`; decidir consistente).
- ¿Descubrir `required_halo` desde el grafo ONNX es fiable, o se declara en el
  `meta.json`/flag? (probable: declararlo, validar contra el grafo si se puede).
- ¿Vale la pena un caché de tiles para modelos con stride > 1? Diferir a demanda.

## 8. Consumidores

geoembed-rs (embeddings densos), smelt-ml, la clasificación multi-hazard de
Ñuble (correr su modelo de susceptibilidad como mapa, no como tabla), y cualquier
usuario que hoy exporta chips con `extract-patches` y no tiene cómo volver a mapa
sin Python.
