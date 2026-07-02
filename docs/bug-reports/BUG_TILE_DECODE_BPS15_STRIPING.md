---
De: Sesión postdoc (susceptibilidad multi-cuenca, paper 1)
Para: Sesión SurtGIS
Fecha: 2026-05-08
Versión: surtgis 0.7.0 (compilado en VM Azure con features `cloud,projections`)
Catálogo: Microsoft Planetary Computer (Sentinel-2 L2A)
Prioridad: **CRÍTICA — los outputs son inutilizables para análisis científico**
---

# v0.7.0: tiles `bps=15` decodificados parcialmente → striping severo en composite (Maule + Biobío, PC)

## TL;DR

`stac composite` con catálogo PC produjo **outputs visualmente quebrados** (bandeo horizontal/vertical, bloques saturados, discontinuidades por strip) en dos cuencas grandes (Maule 7274×5725 px; Biobío 7279×8247 px) procesadas exitosamente *según el log* (todas las bandas escritas, validez >90 %). El root cause aparenta ser:

1. **El decodificador COG/TIFF no maneja correctamente tiles con `bps=15`** (15 bits per sample, no estándar pero presente en algunos COGs L2A de PC).
2. SurtGis loguea cada vez `expected 262144 px, got 245760` (faltan exactamente 16384 px = 6.25 % por tile, consistente con la ratio 15/16) y **continúa la corrida**, escribiendo los píxeles faltantes con un valor por defecto (0 o último valor decodificado).
3. Cuando ese 6.25 % se repite por miles de tiles a través de muchas escenas, **la mediana inter-escena se contamina sistemáticamente** y aparece como striping/saturación en el output final.

**Conteo de eventos en logs reales** (corrida 4-7 mayo 2026, Azure D8s_v6, PC):

| Cuenca | `bps=15` events loggeados | tile-size mismatches | output dims | bandas con valid <95 % |
|---|---:|---:|---|---|
| 11 Maule | 87,102 | 65,220 | 7274 × 5725 | B06, B07 (90.5 %) |
| 12 Biobío | 141,428 | 106,071 | 7279 × 8247 | múltiples 96-97 % |

## Evidencia visual

Quicklook (stretch p2-p98) de las 10 bandas para ambas cuencas en `quicklook_11_rio_maule.png` y `quicklook_12_rio_biobio.png` (adjuntos en este directorio):

- **Maule B06 / B07**: bloques rectangulares saturados de tamaño aproximadamente strip-sized en zonas centrales del raster. Esas dos bandas son las únicas que cayeron a 90.5 % de validez (el resto >99 %).
- **Maule B08, B8A**: franjas saturadas anchas en la parte superior del raster.
- **Maule B11, B12**: striping horizontal claramente visible en zonas planas.
- **Biobío todas las bandas**: striping continuo horizontal y vertical. B11/B12 son las peores. B02 muestra striping vertical particularmente en la mitad sur.

Para una corrida de benchmarking ML (23 algoritmos × 15 cuencas × 58 factores) este nivel de artefacto sería **devastador**: cualquier aprendizaje "geográfico" del modelo se confundiría con artefactos de strip-y-tile del compositor.

## Reproducción exacta

### Comando

```bash
SURTGIS=/home/azureuser/surtgis/target/release/surtgis
FORCE_CATALOG=pc $SURTGIS --compress stac composite \
  --catalog=pc \
  --bbox=-72.687,-36.579,-70.319,-35.080 \
  --collection=sentinel-2-l2a \
  --asset=B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 \
  --datetime=2022-01-01/2024-12-31 \
  --max-scenes=50 \
  --naming=asset \
  --align-to=dem_30m.tif \
  composite.tif
```

(idéntico para Biobío con su BBOX y DEM correspondientes).

### Entorno

- Azure D8s_v6 (8 vCPU, 32 GB RAM, premium SSD), westus3
- Ubuntu 24.04 LTS, kernel 6.x
- SurtGis built from source con `--features cloud,projections`
- libhdf5-dev, libnetcdf-dev, libeccodes-dev instalados
- Conectividad Azure→Planetary Computer ~0 ms (muy bueno)
- 50 escenas Sentinel-2 L2A 2022-2024 sobre la cuenca

### Estado de las corridas

- Maule: ✅ completó (24 h 4 min wallclock), 10 bandas escritas, **inutilizables**
- Biobío: ✅ completó (41 h 38 min wallclock), 10 bandas escritas, **inutilizables**
- Bueno: ❌ interrumpido en strip 9/11 después de detectar el problema en las dos previas

## Análisis del bug

### Patrón en el log

Cada evento problema tiene esta forma:

```
[cog] pred=1 comp=8 bps=15 sf=1 raw_len=491520 expected=524288
[cog] tile(X,Y) idx=N expected 262144 px, got 245760 (raw=491520 bytes, bps=15 tw=512 th=512)
```

Aritmética:
- `expected = 524288 bytes = 262144 * 2 bytes` (uint16, 1 banda, 512×512)
- `raw_len = 491520 bytes = 245760 * 2 bytes`
- Ratio `245760 / 262144 = 15/16` ⇒ los datos comprimidos fueron escritos asumiendo **15 bits per sample** en lugar de 16
- Faltan **16384 píxeles** por tile (6.25 %)

### Hipótesis de la causa

El header IFD del COG declara `BitsPerSample = 15`, lo cual **es válido en TIFF** aunque infrecuente. Microsoft Planetary Computer reescribe COGs Sentinel-2 con compresión optimizada y aparentemente algunos tiles quedan con `bps=15` (probablemente por el predictor horizontal sobre datos con dynamic range que cabe en 15 bits, ahorrando 6.25 % de espacio).

El decodificador de SurtGis aparenta:
1. Leer correctamente `bps=15` del header
2. Calcular el tamaño esperado en bytes basado en `bps=16` (`width * height * 2`) — **bug aquí**
3. Detectar el mismatch de tamaño
4. Loguear el warning
5. **Continuar con datos parciales**, dejando los 16384 píxeles faltantes como ceros o no inicializados

Esto se vuelve **silencioso en términos de corrida** (no aborta, no hay error fatal) pero **catastrófico para el output**: cada tile mal decodificado deja una lesión rectangular en cada escena, y la mediana sobre 50 escenas no consigue lavar la inconsistencia porque el patrón se repite por la misma estructura de tiles en todas las escenas.

### Por qué B06/B07 son las peores

B06 y B07 (red edge 2 y 3, 20 m nativo, resampleado a 10 m por SurtGis) tienen, en PC, una proporción mayor de tiles con `bps=15` (probablemente porque su rango dinámico es menor que el visible). Esto correlaciona con la caída a 90.5 % de validez en Maule.

### Por qué Biobío es peor que Maule

Biobío es ~42 % más grande (8247 vs 5725 filas) → más tiles totales → más oportunidades de encontrar `bps=15` → más errores acumulados (141,428 vs 87,102). El striping también es más visible porque hay más filas para que el patrón se exprese.

## Lo que SurtGis debería hacer

### Mínimo (no rompe API ni performance)

1. **Soporte correcto de `bps=15`** en el decodificador COG: leer 15 bits por sample, des-empaquetar correctamente. Es algo que `tifffile` (Python) y GDAL hacen bien.
2. Si decidir no soportarlo: **abortar la corrida** con error claro en lugar de continuar con datos parciales. Un fallo ruidoso es preferible a un output silenciosamente roto.

### Recomendado

3. **Validación post-decode**: si `decoded_bytes != expected_for_bps16`, marcar el tile como inválido y excluirlo de la mediana en lugar de promediarlo con datos parciales.
4. **Métrica de salud por banda**: imprimir al final del job el % de tiles con decode issues por banda. Hubiera permitido detectar el bug *antes* de bajar 1.4+1.9 GB de outputs corruptos por rsync.
5. **Test de regresión** con un fixture COG `bps=15` (extraíble de cualquier escena Sentinel-2 L2A actual de PC).

### Avanzado

6. **Degradación graceful**: si una escena tiene >X % de tiles `bps=15` con decode failure, excluirla completamente del composite y avisar.

## Comparación con bug previo de RAM (`BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md`)

El bug de RAM (peak real 12-13 GB vs predicho 8 GB) era una molestia operacional pero **los outputs eran correctos**. Este bug es **categóricamente distinto y peor**: la corrida termina "exitosamente" según el log, las stats parecen aceptables (mean en rangos razonables, validez >90 %), pero los rasters son **científicamente inutilizables**.

## Workaround mientras se arregla

El stack Python (`pystac-client` + `odc-stac` + `rasterio`) decodifica `bps=15` correctamente vía libtiff. Para los próximos basins (13-15) se va a hacer la composición con un script Python en la misma VM Azure (32 GB RAM, conectividad Azure→PC ~0 ms, mismo costo $0.40/h). Estimo 30-60 min por cuenca con `dask.compute()` paralelo en lugar de las 24-40 h de SurtGis.

## Mensaje al equipo SurtGIS

Esto es **aprendizaje crítico**: una corrida de 65 h de wallclock total (Maule + Biobío) y ~$25 USD de Azure entregó dos cuencas inutilizables. El log mostraba 87k+141k = 228,530 warnings que no llamaron mi atención porque (a) eran indistinguibles de logs de debug normales, (b) el job nunca abortó, (c) las stats agregadas parecían razonables. El primer indicio del problema fue **abrir las imágenes en QGIS** después de bajarlas localmente.

Recomendaciones de proceso:
- Tratar los warnings de tile decode como errores (al menos elevar verbosidad y conteo final).
- Agregar test de integración con un COG real de Planetary Computer L2A (no solo fixtures sintéticos).
- En el `--help` de `stac composite`, mencionar las limitaciones conocidas del decodificador cuando se descubran.

## Adjuntos

- `quicklook_11_rio_maule.png` — 10 bandas Maule, stretch p2-p98 (~1 MB)
- `quicklook_12_rio_biobio.png` — 10 bandas Biobío, stretch p2-p98 (~1.3 MB)
- Logs completos disponibles en VM Azure: `/home/azureuser/work/factors/{11_rio_maule,12_rio_biobio}/spectral/download.log` (200k+ líneas cada uno; ofrezco extracto si se quiere reproducir el patrón).
