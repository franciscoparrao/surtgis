# SPEC — `Complex<f32>` en `Raster` (fundación InSAR)

> **Propósito:** habilitar rasters de fase compleja para interferometría, como
> **fundación pura** sobre la que arranca `insar-rs`, sin meter dominio InSAR en
> `surtgis-core`. Es la P2.1 del `SPEC_SURTGIS_ECOSYSTEM_FOUNDATION.md`,
> desarrollada como diseño concreto.
> **Estado:** propuesta de diseño (2026-07-12). **Baseline objetivo:** SurtGIS
> v0.19.0, feature `insar` (opcional). **Scaffold puro** — el valor está en dejar
> el tipo y el I/O listos; los algoritmos InSAR viven en el motor hermano.

---

## 0. Por qué (y por qué es scaffold, no feature completa)

El ecosystem SPEC es explícito: agregar `Complex` **solo con el consumidor a la
vista** (insar-rs), y NO poner lógica InSAR (coregistro, unwrapping, coherencia
conceptual) en core. Esta SPEC hace exactamente eso: deja `Raster<Complex<f32>>`
como ciudadano de primera clase para I/O y operaciones elementales, y **nada
más**. Cuando insar-rs arranque, hace `surtgis-core = "0.19"` y ya tiene el
contenedor + lectura/escritura, sin duplicar scaffolding.

---

## 1. Verificar primero (no asumir)

- [ ] `RasterElement` (en `core::raster`): ¿qué exige exactamente (traits:
      `Copy`, `Zero`, nodata, conversión `DataType`)? Listar los bounds para saber
      qué debe cumplir `Complex<f32>`.
- [ ] `DataType` enum: ¿tiene variantes complejas (`CFloat32`)? GeoTIFF/GDAL las
      define (TIFFTAG SampleFormat = 3 complejo). Confirmar si el reader/writer
      nativo (`io/native.rs`) puede round-trippear complejo o solo real.
- [ ] `AnyRaster`: ¿cómo se enumeran los tipos dinámicos? Agregar la variante
      compleja sin romper el match exhaustivo existente.
- [ ] ¿`num-complex` ya está en el árbol (transitivo)? Elegir esa repr para
      `Complex<f32>` y feature-gate `insar` que la active.
- [ ] Manejo de nodata para complejo: ¿NaN en cualquier componente? Definir la
      convención y documentarla.

---

## 2. Diseño

### 2.1 El tipo

- `impl RasterElement for Complex<f32>` (repr `num_complex::Complex<f32>`), tras
  feature `insar`. Nodata = `Complex::new(NaN, NaN)` (o la convención que fije §1).
- `DataType::CFloat32` (nueva variante) + su mapeo a SampleFormat=3 en el I/O.

### 2.2 I/O

- Lectura/escritura GeoTIFF complejo en `io/native.rs` (SampleFormat=3, 2×f32
  intercalados por muestra). Round-trip bit-exacto es el único criterio duro.
- Preservar CRS/transform/nodata como cualquier raster.

### 2.3 Operaciones elementales (mínimas, no dominio)

Solo lo que es genuinamente "primitiva de contenedor", NO InSAR:

- Constructores: `from_parts(re: &Raster<f32>, im: &Raster<f32>)`,
  `from_amp_phase(amp, phase)`.
- Accesores: `.real() -> Raster<f32>`, `.imag()`, `.amplitude()`, `.phase()`.
- Helpers FFT **opcionales** (feature `insar`, vía `rustfft` pure-Rust) como
  utilidad de contenedor — el uso InSAR (espectro, filtrado) lo decide el motor.

Todo lo demás (interferograma = producto conjugado, coherencia, coregistro,
phase unwrapping, filtro Goldstein) **NO va aquí** — vive en `insar-rs`.

---

## 3. Superficie

- **Rust:** el tipo + I/O + constructores/accesores. Es la superficie principal;
  el consumidor es otro crate Rust.
- **Python/WASM:** diferido. Exponer complejo por PyO3 solo cuando insar-rs pida
  el puente (se alinea con P2.2, patrón de bindings). No bloquear con esto.

## 4. Criterios de aceptación (Definition of Done del scaffold)

- [ ] `impl RasterElement for Complex<f32>` compila tras `--features insar`,
      sin afectar el build por defecto.
- [ ] `DataType::CFloat32` + match exhaustivos actualizados (sin `todo!()` que
      rompa otros tipos).
- [ ] Round-trip GeoTIFF complejo bit-exacto (test) — escribir y releer un raster
      complejo sintético y comparar.
- [ ] `from_parts` / `amplitude` / `phase` con un test analítico simple.
- [ ] Helpers FFT tras feature, con un test de identidad FFT→IFFT.
- [ ] Un **smoke test de fundación**: un stub `insar-rs` (o un ejemplo en el
      repo) hace read→conjugate-product→write usando solo la superficie de core.

## 5. No-objetivos

- **NO** algoritmos InSAR en core (coregistro, unwrapping, coherencia). Core =
  tipo + I/O + primitivas de contenedor.
- **NO** activar complejo en el binario/wheel por defecto — siempre feature
  `insar`.
- **NO** sobre-abstraer a `Complex<f64>` u otros hasta que un consumidor lo pida.

## 6. Consumidores

`insar-rs` (único, pero definido). Esta SPEC es el desbloqueo de fundación para
que ese motor arranque sin reimplementar I/O ni tipos raster.
