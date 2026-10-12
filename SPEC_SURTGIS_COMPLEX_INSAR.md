# SPEC — `Complex<f32>` en `Raster` (fundación InSAR)

> **Propósito:** habilitar rasters de fase compleja para interferometría, como
> **fundación pura** sobre la que arranca `insar-rs`, sin meter dominio InSAR en
> `surtgis-core`. Es la P2.1 del `SPEC_SURTGIS_ECOSYSTEM_FOUNDATION.md`,
> desarrollada como diseño concreto.
> **Estado: CERRADA (2026-07-16).** El grueso de esta SPEC ya estaba
> implementado desde antes de que este documento existiera — PR #40
> (`61c3dbd`, 2026-06-10, "complex rasters for InSAR — RasterCell/RasterElement
> split", v0.15.4) — bajo `feature = "complex"` (no `"insar"` como proponía
> este texto originalmente). Esta revisión documenta el diseño **tal como
> quedó implementado**, con dos divergencias deliberadas respecto a la
> propuesta original (§2.2, §2.3), y cierra los gaps que sí faltaban
> (`from_amp_phase`, smoke test de fundación). Ver §4 para el estado real de
> cada criterio de aceptación.

---

## 0. Por qué (y por qué es scaffold, no feature completa)

El ecosystem SPEC es explícito: agregar `Complex` **solo con el consumidor a la
vista** (insar-rs), y NO poner lógica InSAR (coregistro, unwrapping, coherencia
conceptual) en core. Esta SPEC hace exactamente eso: deja `Raster<Complex<f32>>`
como ciudadano de primera clase para I/O y operaciones elementales, y **nada
más**. Cuando insar-rs arranque, hace `surtgis-core = "0.19"` y ya tiene el
contenedor + lectura/escritura, sin duplicar scaffolding.

---

## 1. Verificado (resultado real, no la pregunta original)

- `RasterElement`/`RasterCell` (`core::raster::element`): `Complex<f32>` y
  `Complex<f64>` implementan `RasterCell` vía macro (`impl_raster_cell_complex!`),
  gateado tras `feature = "complex"`. No requiere `DataType` — ver siguiente punto.
- `DataType` (`core::raster::any_raster`) **no tiene** variante compleja. Decisión
  de diseño (divergencia de §2.2 original): en vez de `CFloat32`/SampleFormat=3
  nativo, un raster complejo se persiste como **dos rasters `f32` reales
  independientes** (re, im) vía `complex_to_parts`/`complex_from_parts`. Evita
  tocar el I/O nativo (`io/native.rs`) y el match exhaustivo de `AnyRaster` por
  completo — el ahorro de complejidad se juzgó mayor que el de un solo archivo
  por interferograma.
- `AnyRaster`: sin cambios, sin variante compleja — consistente con el punto
  anterior.
- `num-complex` ya estaba en el árbol de dependencias del workspace (`Cargo.toml`
  raíz, `num-complex = "0.4"`); en `crates/core` es opcional
  (`complex = ["dep:num-complex"]`).
- Nodata: `Complex::new(NaN, NaN)` como `default_nodata()`; `is_nodata` también
  trata "ambas partes NaN" como nodata aunque no coincida bit a bit con el
  sentinel explícito (mismo criterio laxo que los `f32`/`f64` reales).

---

## 2. Diseño (tal como quedó implementado)

### 2.1 El tipo

- `impl RasterCell for Complex<f32>` y `Complex<f64>` (repr `num_complex::Complex<T>`),
  tras `feature = "complex"` (no `"insar"` — el nombre de feature real difiere
  de la propuesta original; se mantiene porque ya es superficie pública desde
  v0.15.4 y renombrarla ahora sería un breaking change sin beneficio).
- Nodata = `Complex::new(NaN, NaN)`.
- **No hay `DataType::CFloat32`** — ver §1, divergencia deliberada.

### 2.2 I/O — DIVERGENCIA vs. propuesta original

En vez de SampleFormat=3 nativo en `io/native.rs`, el camino de persistencia es
**split a dos bandas reales**: `complex_to_parts(&Raster<Complex<T>>) -> (Raster<T>, Raster<T>)`
escribe/lee cada parte con `write_geotiff`/`read_geotiff` normales, y
`complex_from_parts(&Raster<T>, &Raster<T>) -> Result<Raster<Complex<T>>>`
las reensambla (validando shape/transform/CRS iguales). Round-trip bit-exacto
verificado en `geotiff_roundtrip_via_parts` (`crates/core/src/raster/complex.rs`).
CRS/transform/nodata se preservan porque cada parte es un `Raster<T>` real
completo. Costo: dos archivos por raster complejo en vez de uno; beneficio: cero
código nuevo en el I/O nativo ni en `AnyRaster`.

### 2.3 Operaciones elementales (mínimas, no dominio)

- Constructores: `complex_from_parts(re, im)` y `complex_from_amp_phase(amp, phase)`
  (agregado 2026-07-16 — usa `Complex::from_polar`).
- Accesores: `complex_to_parts(&raster) -> (re, im)`, `magnitude(&raster)`,
  `phase(&raster)` (todas en `crates/core/src/raster/complex.rs`).
- Helpers FFT: **descartados**, no solo diferidos. La implementación real
  documenta la decisión explícitamente (doc comment del módulo): "FFT helpers
  are deliberately out of scope — spectral processing belongs to insar-rs".
  Es consistente con el espíritu de esta SPEC (§0) pero más estricto que la
  propuesta original, que los pedía como opcionales bajo feature.

Todo lo demás (interferograma = producto conjugado, coherencia, coregistro,
phase unwrapping, filtro Goldstein) **NO está aquí** — vive en `insar-rs`. El
producto conjugado en particular se probó como responsabilidad del consumidor
en el smoke test de fundación (§4, `insar_conjugate_product`).

---

## 3. Superficie

- **Rust:** el tipo + I/O + constructores/accesores. Es la superficie principal;
  el consumidor es otro crate Rust.
- **Python/WASM:** diferido. Exponer complejo por PyO3 solo cuando insar-rs pida
  el puente (se alinea con P2.2, patrón de bindings). No bloquear con esto.

## 4. Criterios de aceptación (Definition of Done del scaffold) — estado real

- [x] `impl RasterCell for Complex<f32>` (y `f64`) compila tras
      `--features complex`, sin afectar el build por defecto (verificado:
      `cargo check -p surtgis-core` sin la feature compila limpio).
- [x] ~~`DataType::CFloat32` + match exhaustivos~~ — **no aplica**, decisión de
      diseño fue split re/im (§2.2). Cerrado como no-objetivo, no como pendiente.
- [x] Round-trip GeoTIFF complejo bit-exacto (test) — `geotiff_roundtrip_via_parts`,
      vía split re/im.
- [x] `from_parts` / `from_amp_phase` / `magnitude` / `phase` con tests
      analíticos (`parts_roundtrip_with_nodata`, `from_amp_phase_matches_polar_form`,
      `magnitude_and_phase`).
- [x] ~~Helpers FFT~~ — **descartados** (§2.3), no forman parte del DoD final.
- [x] **Smoke test de fundación**: `crates/core/examples/insar_conjugate_product.rs`
      — dos rasters SLC sintéticos (master/slave) → GeoTIFF (re,im) → read →
      `complex_from_parts` → producto conjugado (implementado en el ejemplo,
      *no* en core) → `magnitude`/`phase` → write. Corre con
      `cargo run -p surtgis-core --example insar_conjugate_product --features complex`.
      Prueba que un consumidor externo arranca sin reimplementar tipos ni I/O.

## 5. No-objetivos

- **NO** algoritmos InSAR en core (coregistro, unwrapping, coherencia). Core =
  tipo + I/O (split re/im) + primitivas de contenedor.
- **NO** activar complejo en el binario/wheel por defecto — siempre feature
  `complex`.
- **NO** `DataType::CFloat32` ni SampleFormat=3 nativo — split re/im cubre el
  caso de uso sin tocar el I/O nativo (§2.2).
- **NO** helpers FFT en core — quedan enteramente en `insar-rs` (§2.3).
- `Complex<f64>` **sí** se implementó junto con `Complex<f32>` (mismo costo de
  macro) — la restricción original de "no sobre-abstraer" se relaja porque no
  agregó superficie nueva, solo otra instancia del mismo trait.

## 6. Consumidores

`insar-rs` (único, pero definido). Esta SPEC es el desbloqueo de fundación para
que ese motor arranque sin reimplementar I/O ni tipos raster.
