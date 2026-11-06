# Cuencas de lagos en permafrost con SurtGIS: respuesta al brief del paper ártico

**Fecha**: 2026-11-05
**Pedido**: brief "delimitación masiva de cuencas de lagos en terreno de permafrost" (proyecto smelt, paper de química de lagos árticos).
**Base**: SurtGIS 1.7.0 + PR #151; prototipo y resultados crudos en `~/proyectos/smelt/artic/results/lake_catchments_surtgis/`.

**Leyenda**: *verificado* = medido o comprobado en esta sesión; *inferido* = juicio razonado, con su confianza; *supuesto* = no comprobado.

---

## 1. Conclusión

SurtGIS ya reemplaza a HydroLAKES para esta muestra: el módulo `surtgis hydrology lake-catchment` está construido (guía en el libro: *Delineate lake catchments*). Recibe polígonos de lagos, uno o varios DEM (STAC o teselas locales como FABDEM) y escribe una tabla por lago y DEM. La condición sigue siendo que los lagos entren como polígonos; con solo coordenadas hace falta una máscara de agua, y esa detección es frágil.

El criterio del brief ("área de cuenca dentro de un 20 % de HydroLAKES en la mayoría") **no se cumple** (3 de 20 con GLO-30), pero no sirve para decidir: al norte de 60°N, HydroLAKES calcula las cuencas con HydroSHEDS v1 sobre el DEM HYDRO1k de 1 km (verificado en las dos documentaciones técnicas).

1. **Cobertura**: solo el 20 % de las 1.539 ubicaciones cae dentro de un lago de HydroLAKES (verificado).
2. **Robustez**: con el mismo lago, la cuenca conectada coincide dentro del 20 % entre GLO-30 y FABDEM en 18 de 20 lagos, y entre GLO-30 y AW3D30 en 11 de 20 (verificado).
3. **Dos cuencas**: en terreno de permafrost hay que distinguir la cuenca **conectada** (toda cubeta que rebalsa al lago) de la **local** (las depresiones reales retienen su agua). Difieren hasta en dos órdenes de magnitud; el módulo reporta ambas.
4. **DEM**: FABDEM en la franja boreal, GLO-30 por encima de 80°N, AW3D30 como segunda opinión (sección 4).

Pendiente: volumen del lago y tiempo de residencia (sección 5), y una validación contra una referencia independiente de alta resolución.

## 2. Los puntos 1 a 5 del brief

| Punto | ¿Se puede hoy? | Cómo, o qué falta |
|---|---|---|
| 1. Los lagos son las depresiones | **Sí**, con un cambio de orden | Definir el lago **antes** de rellenar y pasar **todas sus celdas** como puntos de salida a `watershed`; luego `priority_flood` y D8 sobre el resto. El relleno ya no borra el lago, porque el lago es la salida. `nested_depressions` no hace falta para esto: sirve para jerarquías de depresiones, no para separar un lago conocido. |
| 2. La salida no es un píxel | **Parcial** | `watershed` acepta una lista de píxeles, y basta pasarle el lago completo (verificado). Falta: rasterizar un polígono de lago sobre la grilla del DEM como conjunto de salida, y la detección automática desde una máscara de agua cuando no hay polígono. |
| 3. Escala continental | **Sí, por ventanas** | El prototipo baja Copernicus GLO-30 por STAC solo alrededor de cada lago, reproyecta a UTM y agranda la ventana (16 → 32 → 64 → 128 km) mientras la cuenca o el lago toquen el borde. Falta el orquestador dentro de SurtGIS; hoy lo hace un script. |
| 4. Muchos lagos por corrida | **No** | Falta el modo batch: paralelismo por lago, reintentos y salida tabular. |
| 5. Tiempo de residencia | **Parcial** | La acumulación ponderada existe desde 1.6.0 (`hydrology flow-accumulation --weights`), pero con la cuenca ya delimitada basta el caudal medio = escorrentía media de la cuenca × área. Falta el volumen: no hay un modelo de profundidad de lago en SurtGIS. |

**Medición del prototipo (verificado)**: mediana de 9 s por lago con GLO-30. Una ventana de 16 km bastó en 15 de 20 lagos; un lago necesitó 128 km. Para unos 2.300 lagos y 8 en paralelo, el costo sería del orden de una hora (inferido, probable; domina la descarga).

---

## 3. Benchmark: 20 lagos contra HydroLAKES

### Método

1. **Cruce.** Las 2.849 filas de la muestra (1.539 ubicaciones únicas, EPSG:3995) se cruzaron con los 765.442 polígonos de HydroLAKES al norte de 59°N.
2. **Selección.** 20 lagos estratificados por área (de 0,13 a 38 km²) y sector de longitud: Alaska, Canadá, Groenlandia y Siberia.
3. **Delimitación.** DEM por STAC de Planetary Computer, reproyectado a UTM con vecino más cercano y 30 m. Lago como conjunto de salida, priority-flood (ε = 1e-4), D8 y `watershed`.
4. **Dos variantes de lago:**
   - **A**: el lago se detecta desde ESA WorldCover 2021 (clase 80), la componente conectada más cercana a la coordenada.
   - **B**: el lago es el polígono de HydroLAKES rasterizado. Así se compara solo la cuenca, con el mismo lago en ambos lados.
5. **Segunda opinión.** La variante B se repitió con ALOS AW3D30.

### Cobertura de HydroLAKES sobre la muestra (verificado)

| Distancia al lago de HydroLAKES más cercano | Ubicaciones | % |
|---|---|---|
| Dentro del polígono | 308 | 20,0 |
| A 250 m o menos | 507 | 32,9 |
| A 1 km o menos | 843 | 54,8 |
| Sin lago de HydroLAKES a menos de unos 2 km | 395 | 25,7 |

### Área del lago (variante A, WorldCover)

Con la apertura morfológica final, la razón SurtGIS/HydroLAKES tiene mediana 1,07: 9 de 18 lagos dentro del 20 % y 14 de 18 dentro del 50 %. Dos teselas de WorldCover no se pudieron bajar de Planetary Computer.

La detección automática es la parte frágil. WorldCover une lagos con ríos anchos: el lago 2 salió con 268 km² de cuenca antes de la apertura. La apertura corta esos ríos, pero parte lagos de forma estrecha: el lago 18 bajó de 15,4 a 1,0 km² de espejo. **Inferido (muy probable)**: el módulo debe recibir polígonos de lagos (HydroLAKES donde existan; una máscara propia u otro inventario donde no) y dejar la detección automática como respaldo con advertencia.

### Área de cuenca (variante B, mismo lago; módulo `lake-catchment`)

> **Corrección.** La primera versión de este documento reportaba cuencas calculadas con un relleno que trataba al lago como una depresión más: un lago en una cubeta cerrada quedaba rellenado hasta la cota de rebalse, y su cubeta drenaba hacia el rebalse en vez de hacia el lago. El módulo final trata al lago como salida del relleno (como el mar) y reporta dos cuencas. Las cifras de esta sección son las corregidas; las anteriores quedan obsoletas.

Dos cuencas por lago, en km²:

- **Conectada** (`catchment_area_km2`): toda depresión se llena y rebalsa; una cubeta cuyo rebalse llega al lago le pertenece. Es la convención de HydroSHEDS y HydroLAKES.
- **Local** (`local_catchment_area_km2`, `--sink-depth 1`): las depresiones de 1 m o más conservan su propio drenaje; las menores (ruido) se rellenan.

| Lago | HydroLAKES | GLO-30 conectada | GLO-30 local | FABDEM conectada | FABDEM local | AW3D30 conectada | AW3D30 local | `dem_spread` |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,20 | 145,4 | 2,08 | 182,9 | 18,63 | 151,1 | 9,55 | 1,258 |
| 2 | 0,50 | 1,03 | 1,03 | 0,92 | 0,92 | 150,9 | 3,35 | 164,810 |
| 3 | 48,30 | 72,50 | 2,65 | 75,69 | 5,50 | 79,64 | 0,70 | 1,099 |
| 4 | 138,0 | 0,43 | 0,43 | 0,43 | 0,43 | 0,40 | 0,39 | 1,081 |
| 5 | 1,50 | 9,52 | 3,89 | 9,46 | 4,28 | 6,63 | 3,15 | 1,436 |
| 6 | 3,20 | 15,31 | 12,51 | 14,70 | 14,70 | 14,41 | 2,13 | 1,062 |
| 7 | 1,00 | 2,13 | 2,13 | 2,47 | 2,47 | 2,34 | 1,09 | 1,159 |
| 8 | 59,50 | 61,52 | 28,63 | 63,73 | 63,73 | 77,24 | 6,85 | 1,256 |
| 9 | 3,40 | 2,99 | 2,99 | 3,05 | 3,05 | 3,94 | 2,76 | 1,320 |
| 10 | 12,70 | 6,22 | 5,03 | 6,15 | 4,98 | 6,16 | 2,66 | 1,012 |
| 11 | 10,60 | 7,27 | 4,38 | 271,4 | 5,43 | 88,12 | 1,60 | 37,316 |
| 12 | 123,5 | 11,06 | 9,67 | 12,16 | 12,16 | 12,20 | 2,49 | 1,104 |
| 13 | 12,60 | 59,29 | 18,46 | 64,05 | 23,68 | 94,36 | 13,49 | 1,592 |
| 14 | 33,70 | 50,88 | 31,87 | 59,40 | 44,62 | 184,7 | 6,96 | 3,630 |
| 15 | 11,70 | 25,03 | 11,64 | 24,54 | 17,92 | 22,44 | 8,25 | 1,115 |
| 16 | 56,60 | 134,6 | 44,86 | 159,5 | 94,57 | 132,6 | 49,67 | 1,203 |
| 17 | 1.197,5 | 1.559,9 | 133,6 | 1.650,7 | 276,3 | 2.809,9 | 56,12 | 1,801 |
| 18 | 152,8 | 150,9 | 87,56 | 147,2 | 88,25 | 153,8 | 27,94 | 1,044 |
| 19 | 35,10 | 89,76 | 24,64 | 94,28 | 71,56 | 159,2 | 18,51 | 1,774 |
| 20 | 57,60 | 263,2 | 111,6 | 247,5 | 136,4 | 291,5 | 63,83 | 1,177 |

Acuerdo entre pares (dentro del 20 % / dentro del 50 % / mediana de la razón):

| Par | Conectada | Local |
|---|---|---|
| GLO-30 contra FABDEM | **18/20** / 19/20 / 0,97 | 10/20 / 14/20 / 0,81 |
| GLO-30 contra AW3D30 | 11/20 / 14/20 / 1,10 | 4/20 / 10/20 / 0,57 |
| FABDEM contra AW3D30 | 11/20 / 15/20 / 1,04 | 2/20 / 7/20 / 0,46 |
| GLO-30 contra HydroLAKES | 3/20 / 5/20 | 3/20 / 7/20 |

Tiempo: los 20 lagos con los tres DEM (60 corridas) toman 9 minutos con 6 lagos en paralelo y 1,2 GB de RAM máxima (verificado).

![Seis lagos del benchmark](figures/lake_catchments_benchmark.png)

*Figura de la primera pasada (relleno anterior): ilustra las cuencas de GLO-30 y AW3D30 frente a HydroLAKES en seis lagos; las áreas de sus títulos son las anteriores a la corrección.*

### Interpretación

- **Verificado**: al norte de 60°N, HydroSHEDS v1 usó el DEM HYDRO1k de 1 km remuestreado a 15 segundos de arco (HydroSHEDS TechDoc, sec. 2.7 y 3.7), y `Wshd_area` de HydroLAKES sale de esa red (HydroLAKES TechDoc). Los polígonos de lagos de Rusia al norte de 60°N vienen de MODIS a 250 m. HydroLAKES no es una referencia válida para cuencas de pocos km² en esta muestra.
- **Verificado**: la cuenca **conectada** es robusta entre GLO-30 y FABDEM (18 de 20 dentro del 20 %) y razonablemente robusta con AW3D30 (11 de 20). La cuenca **local** depende mucho del DEM.
- **Inferido (probable)**: FABDEM da cuencas locales mayores que GLO-30 justo en los lagos boreales (1, 8, 12, 16 y 19, en Siberia y Yukón), y casi iguales en los de tundra (2, 4, 6, 7, 9 y 10). Es lo esperado si el dosel crea pozos falsos en GLO-30. AW3D30, un modelo de superficie, da las cuencas locales más pequeñas por la misma razón.
- **Inferido (muy probable)**: en el termokarst plano (lagos 1, 3, 11, 13, 17 y 19) la conectada y la local difieren en uno o dos órdenes de magnitud. Para el paper conviene reportar ambas, o justificar cuál es la relevante para la química del lago.
- **Supuesto**: que una referencia independiente de alta resolución confirmaría la lectura anterior. No se probó; candidatos: MERIT-Hydro (90 m, requiere registro) o NHDPlus HR en Alaska.

## 4. Qué DEM usar

| DEM | Uso recomendado | Por qué |
|---|---|---|
| **Copernicus GLO-30** | **Principal para cuencas** | Cobertura completa hasta los polos, sin fallas de teselas en esta prueba, en Planetary Computer con STAC. Ojo: **no aplana los lagos pequeños** (verificado en el lago 6: superficie de agua con ruido de ±0,5 m), así que no sirve para detectar lagos desde el DEM. |
| **ALOS AW3D30** | **Segundo DEM para medir la incertidumbre** | Independiente de GLO-30 y en el mismo catálogo. Es un modelo de superficie (incluye vegetación). Con 1.7.0 SurtGIS no lo podía leer; queda arreglado con el PR #151. |
| **FABDEM V1-2** | **Mejor que GLO-30 bajo bosque; igual en tundra** | Es GLO-30 con bosques y edificios removidos por random forest (Hawker et al. 2022; V1-2 corrige la alineación de teselas sobre 50°N). En taiga, el dosel crea falsas divisorias y depresiones que desvían el drenaje; FABDEM las quita. En tundra no hay dosel que quitar y queda prácticamente igual a GLO-30 (inferido). Límites: cobertura documentada hasta 80°N, y 46 de las 1.539 ubicaciones de la muestra están más al norte (verificado); no está en un catálogo STAC con búsqueda, se baja de data.bris en teselas COG; licencia no comercial (CC BY-NC-SA 4.0 según mirrors; el registro de data.bris no se pudo leer), suficiente para un paper. No se probó en este benchmark. |
| **ArcticDEM** | **Solo para la ribera o una ventana puntual** | A 2 m capta la ribera y sirve para estimar profundidad con la pendiente de orilla. Tiene huecos y artefactos sobre agua, y PGC lo publica como catálogo STAC estático (`arcticdem.json`), que el cliente STAC de SurtGIS todavía no recorre: hoy solo usa APIs con `/search`. Los mosaicos de 10 y 32 m podrían servir para cuencas; no se probaron. |

La intuición del brief se sostiene: GLO-30 para cuencas y ArcticDEM solo si hace falta la ribera. Agrego usar AW3D30 como segunda opinión, y FABDEM en lugar de GLO-30 en la franja boreal (715 de las 1.539 ubicaciones están entre 60 y 66,5°N), con GLO-30 como respaldo sobre 80°N.

---

## 5. `surtgis hydrology lake-catchment`: diseño y estado

**Entrada**
- `--lakes`: polígonos (GPKG, GeoParquet o Shapefile) **o** puntos con `--water-mask worldcover` como respaldo.
- `--dem`: un archivo local, o `--dem-stac pc:cop-dem-glo-30` (repetible, por ejemplo `--dem-stac pc:alos-dem` para el ensamble).
- `--window-km 16 --max-window-km 256`, `--jobs N`, `--retries 3`.
- `--runoff <raster>` opcional, por ejemplo P − ET anual de ERA5-Land, para el caudal.

**Salida**: GeoParquet o CSV con una fila por lago:
- `lake_id`, `lake_area_km2`, `lake_perimeter_km`, `lake_elev_m`;
- `catchment_area_km2`, `catchment_to_lake_ratio`, `catchment_mean_elev_m`;
- `window_km`, `dem`, `status` (`ok`, `edge`, `lake_split`, `error: …`);
- con dos DEM, `catchment_spread` = máx/mín entre DEM;
- con `--runoff`, `q_mean_m3s` y, si hay volumen, `residence_days`.

Los polígonos de cuenca van opcionalmente en un GPKG. Cada salida lleva la provenance con los ítems STAC leídos, desde 1.7.0.

**Piezas**

| Pieza | Estado | Esfuerzo |
|---|---|---|
| `hydrology::lake_catchment(dem, lake_mask)` en la librería: lago como salida, relleno, D8 y unión de cuencas | prototipo listo | 2–3 días con tests |
| Polígono → conjunto de salida sobre la grilla del DEM; detección desde máscara de agua con apertura y aviso de lago partido o unido a río | prototipo listo | 2–3 días |
| Orquestador: ventana STAC por lago, reproyección a UTM, ampliación adaptativa, ensamble de DEM, paralelismo, reintentos, tabla de salida | script en Python | 1 semana |
| Caudal: zonal de escorrentía sobre la cuenca | piezas existentes | 1–2 días |
| Volumen y residencia: profundidad desde la pendiente de ribera (Hollister y Milstead 2010) y volumen cónico; validar contra `Vol_total` de HydroLAKES en lagos grandes | no existe | 1 semana |
| Docs, ejemplo y CLI | — | 2–3 días |

**Estado (2026-11-05)**: construidos la función de librería, la extracción de lagos desde una máscara de agua, el orquestador (STAC y teselas locales, ventana adaptativa, varios DEM, paralelismo, reintentos, CSV y máscaras) y la cuenca local (`--sink-depth`). Faltan el caudal desde escorrentía y el volumen/residencia (unas 1,5 semanas) y la salida GeoParquet.

**Riesgos**
- La detección automática del lago (une o parte lagos) es el punto débil si no hay polígonos.
- El terreno plano de termokarst hace inestable la cuenca con cualquier DEM.
- Las teselas de WorldCover fallan a veces en Planetary Computer.

---

## 6. Hallazgos colaterales

- **SurtGIS no leía ALOS AW3D30 de Planetary Computer** (ninguna tesela): el lector LZW rechazaba streams que codifican más de lo que cabe en la tesela. Arreglado en el PR #151, verificado idéntico a GDAL en 78.554 celdas.
- **El cliente STAC no recorre catálogos estáticos** (ArcticDEM, REMA, EarthDEM de PGC): es un hueco para el diseño anterior.

---

## 7. Límites

- 20 lagos es una muestra pequeña; la estratificación por tamaño deja 4 lagos por clase.
- No hay una referencia de alta resolución independiente; la comparación entre dos DEM mide robustez, no exactitud.
- Las áreas se calculan en UTM a 30 m (error de escala menor a 0,2 %).
- La variante A depende de WorldCover 2021; la variante B, del polígono de HydroLAKES.
