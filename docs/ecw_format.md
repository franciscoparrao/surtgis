# El formato ECW v2 y el decodificador nativo `surtgis-ecw`

**Fecha**: 2026-09-05 · **Estado**: decodificador v2 funcional, validado contra
el ortomosaico de Maipú (Geomag/ENAP) · **Crate**: `crates/ecw`

Este documento existe para que el conocimiento del formato no quede solo en el
código: es la especificación contra la que está escrito `surtgis-ecw`, más la
revisión de licencias y patentes que respalda la decisión de implementarlo.

## 1. Procedencia del conocimiento y decisión de licenciamiento

ECW (Enhanced Compressed Wavelet) es un formato propietario de ER Mapper
(luego ERDAS, hoy Hexagon). En 2006 ER Mapper liberó el **ECW JPEG 2000 SDK
3.3** (`libecwj2-3.3`) bajo su "ECW Public Use License" (EPUL); mirrors
públicos sobreviven en GitHub (`sasgis/libecwj2`, `rouault/libecwj2-3.3-builds`
— este último de Even Rouault, mantenedor de GDAL).

**Decisión: el SDK se usó exclusivamente como especificación del formato.**
No se portó ni tradujo código. Razones:

1. La EPUL es una licencia tipo GPL con cláusulas propias (obligación de
   fuente para todo lo que enlace, prohibición de modificar el formato,
   cesión de derechos a ERM). Un port sería obra derivada y arrastraría esas
   cláusulas, incompatibles con el dual MIT/Apache-2.0 de SurtGIS.
2. Dentro del SDK, el range coder (`rangecode.c`) y el modelo cuasiestático
   (`qsmodel`) son de Michael Schindler bajo **GPL v2+** — otra razón para no
   portar. La matemática subyacente (range coding, G.N.N. Martin 1979) es de
   dominio público.
3. Las estructuras de datos, constantes de formato y algoritmos *como tales*
   no son protegibles por copyright (solo su expresión); leerlos para
   interoperar y reimplementar desde la comprensión es la práctica estándar
   (misma vía por la que existen los lectores libres de otros formatos
   propietarios).

### Revisión de patentes (verificada 2026-09-05)

Los fuentes del SDK citan tres patentes de EE.UU. (los encabezados dicen
"#6,102,897", un typo por 6,201,897 — otros archivos citan la correcta):

| Patente | Título | Prioridad | Vencimiento | Estado |
|---|---|---|---|---|
| US 6,201,897 | Transformation and selective inverse transformation of large digital images | 1998-11-09 | **2018-11-09** | Expired – Lifetime |
| US 6,442,298 | (continuación de la anterior) | 1998-11-09 | **2018-11-09** | Expired – Lifetime |
| US 6,633,688 | Method system and apparatus for providing image data in client/server systems (streaming ECWP) | 2000-04-28 | **2020-04-28** | Expired – Lifetime |

Fuente: Google Patents ([6201897](https://patents.google.com/patent/US6201897B1/en),
[6442298](https://patents.google.com/patent/US6442298B1/en),
[6633688](https://patents.google.com/patent/US6633688B1/en)).
**Las tres están vencidas hace más de seis años.** La técnica patentada (DWT
recursiva línea a línea con síntesis selectiva) es hoy de uso libre.

### Alcance

- **Cubierto**: ECW versión 2 (el formato clásico, todo lo que emitieron los
  compresores 2.x–4.x). Lectura completa, por ventana y por nivel de
  resolución. Sin `unsafe`, sin FFI.
- **Fuera de alcance**: escritura ECW; ECW v3 (SDK 5.x, 2013+: cabecera
  distinta, más profundidades de bit) — el decodificador lo rechaza con
  `UnsupportedVersion`; JPEG 2000 (para eso está `JP2OpenJPEG` vía GDAL);
  tablas de bloques comprimidas (jamás observadas en archivos reales; el
  parser falla explícito si aparecen).

## 2. Estructura del archivo

### 2.1 Advertencia de byte order

Cita literal del SDK (`fileio_decompress.c`):

> *"DUE TO A COMPLETE COCKUP THE INTS IN THE HEADER ARE STORED ON DISC AS MSB
> WHILST FLOATS, THE BLOCK TABLE AND THE REST OF THE DATA IS LSB!"*

Es decir: **los enteros de la cabecera son big-endian** (herencia Sun);
los `f64`, la tabla de bloques (u64) y todos los payloads de bloques son
**little-endian**. Única excepción: los offsets de sideband *dentro* de un
bloque vuelven a ser big-endian (§2.5).

### 2.2 Cabecera fija

| Offset | Tipo | Campo |
|---|---|---|
| 0 | u8 | magic `'e'` (0x65) |
| 1 | u8 | versión (1 o 2; 3 = formato nuevo, no soportado) |
| 2 | u8 | blocking format (siempre 1 = BLOCKING_LEVEL) |
| 3 | u8 | compress format: 1=UINT8 (gris), 2=YUV, 3=MULTIBAND |
| 4 | u8 | número de niveles QMF (sin contar el nivel-archivo virtual) |
| 5 | u8 | sidebands por nivel (siempre 4) |
| 6 | u32 BE | ancho de la imagen (celdas) |
| 10 | u32 BE | alto |
| 14 | u16 BE | número de bandas |
| 16 | u16 BE | scale factor (v1; en v2 siempre 1) |
| 18 | u16 BE | ancho de bloque (típico 64) |
| 20 | u16 BE | alto de bloque (típico 64) |

**Solo si versión ≥ 2** sigue la georreferencia:

| Tipo | Campo |
|---|---|
| u16 BE | compression rate objetivo |
| u8 | unidades de celda: 1=metros, 2=grados, 3=pies |
| f64 LE | cell increment X |
| f64 LE | cell increment Y (negativo = norte arriba) |
| f64 LE | origen X |
| f64 LE | origen Y |
| char[16] | datum ER Mapper (p. ej. `WGS84`), NUL-terminado |
| char[16] | proyección ER Mapper (p. ej. `SUTM19`) |

Mapeo CRS implementado: `WGS84` + `NUTM##`/`SUTM##` → EPSG 326##/327##;
`WGS84` + `GEODETIC` → 4326. Cualquier otro par queda expuesto como strings.

### 2.3 Cadena de niveles

Por cada nivel `0..num_levels` (nivel 0 = el más pequeño):

| Tipo | Campo |
|---|---|
| u8 | número de nivel (debe venir en orden) |
| u32 BE | ancho del nivel |
| u32 BE | alto del nivel |
| u32 BE × bandas | **binsize** por banda (paso de cuantización) |

Las dimensiones se duplican por nivel: `nivel[k].x = ceil(nivel[k+1].x / 2)`.
El "nivel archivo" (la imagen completa) es el doble del último nivel real.
La dequantización es `valor = bin_i16 × binsize / scale_factor` (v2: ×binsize).

### 2.4 Tabla de offsets de bloques

| Tipo | Campo |
|---|---|
| u32 BE | packed length (incluye el byte de formato que sigue) |
| u8 | encode format de la tabla (1 = RAW en la práctica) |
| u64 LE × (N+1) | offsets de los N bloques + centinela al final |

N = Σ por nivel de `ceil(w/bs) × ceil(h/bs)`. Los bloques se numeran nivel 0
primero, dentro de cada nivel por filas (`id = first_block_del_nivel +
y_block × nr_x_blocks + x_block`). Los offsets son relativos al **primer byte
tras la tabla** (`blocks_start`); el largo del bloque `i` es
`tabla[i+1] − tabla[i]` (por eso el centinela). Largo 0 = bloque vacío
(equivale a todo ceros). El centinela debe coincidir con el final del archivo
— chequeo de integridad gratis.

### 2.5 Formato de un bloque

Un bloque del nivel L contiene, para cada banda, S sidebands: **S=4 en el
nivel 0** (LL, LH, HL, HH) y **S=3 en los demás** (LH, HL, HH — el LL sale de
reconstruir el nivel menor). Con n = bandas × S:

```
u32 BE × (n−1)   offsets acumulados de los sidebands 1.. relativos al área
                 de datos (el sideband 0 empieza en 0)
área de datos:
  por sideband (orden band-major): u8 encode format + payload
```

Cada sideband es un stream independiente de `ancho_bloque × alto_bloque`
valores i16 cuantizados (los bloques de borde almacenan solo las
celdas reales). Codificaciones:

| # | Nombre | Payload |
|---|---|---|
| 1 | RAW | i16 LE crudos |
| 2 | HUFFMAN | árbol serializado + bitstream LSB-first (§2.6) |
| 3 | RANGE | range coder, 2 símbolos por valor (byte bajo, byte alto) |
| 4 | RANGE8 | valor inicial de 16 bits (byte alto primero) + diffs i8 acumulados |
| 5 | ZEROS | sin payload: todo el sideband es cero |
| 6 | RUN_ZERO | u16 LE: bit15=run de (v & 0x7fff) ceros; si no, bit14=signo sobre magnitud de 14 bits |

### 2.6 Huffman

Prefijo: u16 LE (conteo de nodos − 1; redundante, el árbol es
auto-delimitado). Serialización pre-orden: byte `0x00` = nodo interno (sigue
el subárbol del bit 0 y luego el del bit 1); otro byte = hoja:

- **hoja pequeña** (bit 6): `valor = ((byte & 0x30) << 10) | (byte & 0x0f)` —
  los dos bits desplazados caen exactamente sobre los flags run (0x8000) y
  signo (0x4000);
- **hoja grande** (`0x80`): los 2 bytes siguientes son el valor, u16 LE.

Semántica del valor: igual que RUN_ZERO, con el largo del run **almacenado
menos 1**. El bitstream lee bits LSB-first dentro de cada byte; cada
sideband Huffman trae su propio árbol.

### 2.7 Range coder y modelo cuasiestático

Range coder byte a byte (Martin 1979, variante de renormalización por bytes):
CODE_BITS=32, umbral de renormalización 2²³, EXTRA_BITS=7. `start_decoding`
consume 1 byte de cabecera (valor arbitrario) + 1 byte al buffer;
`low = buffer >> 1`, `range = 1 << 7`. Normalización:
`low = (low<<8) | ((buffer<<7) & 0xff); buffer = próximo byte; low |= buffer>>1`.

Modelo: **257 símbolos** (256 bytes + marcador fin), total 4096 (12 bits),
rescale objetivo 2000, tabla de búsqueda de 2⁷ buckets. El modelo arranca
uniforme (los primeros 4096 mod 257 = 241 símbolos con frecuencia 16, el
resto 15) y se rescala duplicando el intervalo hasta 2000, dividiendo las
frecuencias a la mitad (con piso 1) y repartiendo el sobrante como
incremento por aparición. RANGE decodifica 2 símbolos por i16 (bajo, alto);
RANGE8 arranca con un i16 (alto, bajo) y sigue con diferencias i8.

### 2.8 Síntesis (la DWT inversa)

La asimetría es el corazón del formato: el análisis usa un banco de filtros
de 11 taps, pero la **síntesis usa un filtro 1-2-1 de 3 taps** que necesita
solo dos líneas de sidebands por nivel en memoria (la técnica de la patente
6,201,897, hoy vencida). La memoria escala con el *ancho*, nunca con el área.

Con `line0`/`line1` = línea anterior/actual de cada sideband, y las cuatro
fases por paridad de (fila, columna) de salida:

```
fila par,  col par : O = LL1[1] − (LH0[1]+LH1[1]+HL1[0]+HL1[1])/2
                         + (HH0[0]+HH0[1]+HH1[0]+HH1[1])/4
fila par,  col impar: O = (LL1[0]+LL1[1] − HH0[0] − HH1[0])/2
                         − (LH0[0]+LH0[1]+LH1[0]+LH1[1])/4 + HL1[0]
fila impar, col par : O = (LL0[1]+LL1[1] − HH0[0] − HH0[1])/2
                         + LH0[1] − (HL0[0]+HL0[1]+HL1[0]+HL1[1])/4
fila impar, col impar: O = (LL0[0]+LL0[1]+LL1[0]+LL1[1])/4
                         + (LH0[0]+LH0[1]+HL0[0]+HL1[0])/2 + HH0[0]
```

Los índices avanzan tras cada columna par. La salida N necesita las entradas
`(N−1)/2` y `(N−1)/2 + 1`; en los bordes se refleja (`I[−1] := I[0]` tras el
mapeo, e igual al final y en Y). Tras una fila par se lee una línea nueva de
sidebands (roll del ring buffer); las filas impares reutilizan las mismas.

El LL del nivel L viene de reconstruir recursivamente el nivel L−1 (excepto
el nivel 0, que lo trae almacenado). Leer a resolución reducida = elegir como
nivel de partida el más pequeño cuya salida cubra 2× el tamaño pedido, y
remuestrear nearest con paso `size/number` en punto fijo 32.32.

### 2.9 Espacio de color

- **MULTIBAND / UINT8**: cada banda reconstruye directo a 0..255; se redondea
  (round-half-to-even, como el x87 del decodificador de referencia) y satura.
- **YUV** (RGB de 3 bandas): YCbCr estándar JPEG con crominancias centradas
  en cero (son salida wavelet con signo, sin offset +128):
  `R = Y + 1.402·V; G = Y − 0.34414·U − 0.71414·V; B = Y + 1.772·U`.
- El decodificador de referencia opcionalmente añade "texture noise"
  (dithering) — `surtgis-ecw` no lo hace: la salida es determinista.

## 3. El archivo de Maipú (verificación byte a byte)

`ORTOOMOSAICO ENAP DEMO  MAIPU.ecw` (Geomag, corredor ENAP, 19 MB):

- versión 2, MULTIBAND, **4 bandas** (RGB + máscara de opacidad 0/255),
  8 niveles, bloques 64×64, rate 30:1
- **31.666 × 41.817 px** (~1,3 gigapíxeles; 5,3 GB descomprimido)
- GSD 0,0405 m, origen UTM 336.382 E / 6.289.010 N, `WGS84`/`SUTM19`
  → EPSG:32719
- niveles 124×164 → 15.833×20.909 (duplicando); 108.241 bloques; tabla RAW
  de 865.937 bytes (108.242 × 8 + 1) cuyo centinela calza con el final del
  archivo
- binsize 1 en todas las bandas y niveles

Validación actual (tests `crates/ecw/tests/maipu_real_file.rs`): cabecera
exacta contra el análisis manual del hexdump, overview 1:256 con
estadísticas plausibles y georreferencia correcta, ventana 512×512 a 1:1 con
textura fina, consistencia entre niveles consecutivos (MAE < 20 entre
reconstrucciones a 1:128 y 1:256), e inspección visual del corredor
completo y de detalle a 4 cm (techos corrugados, demarcación vial).

**Pendiente**: comparación por tolerancia (PSNR / MAE por banda) contra la
exportación GeoTIFF del mismo mosaico que hará Geomag con su software (se le
pidió a Carlos). ECW es con pérdida: la comparación correcta es por
tolerancia, nunca bit a bit, más igualdad exacta de dimensiones y
georreferencia. Mientras llegue, el oráculo alternativo es un QGIS de
Windows con lectura ECW.

## 4. Rendimiento medido (Maipú, un hilo, release)

| Operación | Tiempo |
|---|---|
| Abrir + parsear cabecera y tabla | < 10 ms |
| Overview 1:32 (989×1306 × 4 bandas) | ~330 ms |
| Ventana 512×512 a resolución completa | ~58 ms |

La lectura por ventana/nivel toca solo los bloques necesarios — abrir un
mosaico de gigapíxeles a escala de overview es interactivo.
