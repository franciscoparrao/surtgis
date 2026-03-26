# METODOLOGÍA DE SEGURIDAD PARA PROCESOS COMPUTACIONALES INTENSIVOS

## 📋 Índice

1. [Introducción](#introducción)
2. [Problema](#problema)
3. [Solución: Sistema de Doble Protección](#solución-sistema-de-doble-protección)
4. [Componentes del Sistema](#componentes-del-sistema)
5. [Implementación Paso a Paso](#implementación-paso-a-paso)
6. [Ejemplos de Uso](#ejemplos-de-uso)
7. [Adaptación a Otros Proyectos](#adaptación-a-otros-proyectos)
8. [Troubleshooting](#troubleshooting)

---

## Introducción

Este documento describe una metodología robusta para ejecutar procesos computacionales intensivos (simulaciones, optimizaciones bayesianas, análisis de datos masivos) sin riesgo de congelar el sistema por consumo excesivo de memoria RAM.

**Casos de uso:**
- Optimizaciones bayesianas con múltiples iteraciones
- Simulaciones agent-based con muchos agentes
- Procesamiento de rasters grandes
- Machine learning con datasets extensos
- Análisis geoespaciales pesados

**Beneficio principal:** Evita que el computador se congele completamente, permitiendo detener procesos problemáticos de forma segura antes del colapso del sistema.

---

## Problema

### Síntomas Comunes:
1. **Sistema congelado:** Mouse/teclado no responden
2. **Swap exhausto:** Disco trabajando constantemente
3. **OOM Killer:** Linux mata procesos aleatoriamente
4. **Pérdida de trabajo:** Reinicio forzoso pierde todo el progreso
5. **Daño potencial:** Corrupción de archivos abiertos

### Causa Raíz:
Procesos que consumen RAM gradualmente hasta agotar toda la memoria disponible, causando que el sistema operativo entre en thrashing (intercambio excesivo con disco) y eventualmente se congele.

### Ejemplo Real (Caso Copiapó):
```
Proceso: Optimización bayesiana (50 iteraciones)
Problema: Después de ~10 iteraciones, sistema congelado
Causa: Acumulación de objetos en memoria sin liberar
Solución: Sistema de doble protección implementado
```

---

## Solución: Sistema de Doble Protección

### Arquitectura de 3 Capas:

```
┌─────────────────────────────────────────────────────┐
│  CAPA 1: Protección Interna (Python/R)             │
│  - Monitoreo continuo dentro del proceso           │
│  - Garbage collection forzado                      │
│  - Auto-detención preventiva                       │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  CAPA 2: Monitor Externo (Shell Script)            │
│  - Vigila el proceso cada N segundos               │
│  - Mata proceso si excede umbrales                 │
│  - Log de uso de memoria                           │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  CAPA 3: Verificación Pre-ejecución                │
│  - Comprueba memoria disponible antes de empezar   │
│  - Advierte al usuario si hay riesgo              │
│  - Permite cancelar si hay poca memoria            │
└─────────────────────────────────────────────────────┘
```

### Principios de Diseño:

1. **Defensa en Profundidad:** Múltiples capas de protección
2. **Fail-Safe:** Si una capa falla, las otras actúan
3. **Detección Temprana:** Actuar antes del colapso
4. **Limpieza Activa:** Liberar memoria proactivamente
5. **Logging:** Registrar todo para debugging

---

## Componentes del Sistema

### 1. Script de Monitoreo Externo (`monitor_proceso.sh`)

**Propósito:** Vigilar un proceso desde fuera y matarlo si consume demasiada memoria.

**Características:**
- Monitoreo cada N segundos (configurable)
- Umbrales de alerta y crítico
- Alertas consecutivas antes de matar
- Logging de uso de memoria
- Detención suave (SIGTERM) antes de forzada (SIGKILL)

**Parámetros configurables:**
```bash
MEMORY_THRESHOLD=85    # % RAM para alerta
MEMORY_CRITICAL=92     # % RAM para matar proceso
CHECK_INTERVAL=10      # Segundos entre verificaciones
MAX_ALERTS=3           # Alertas antes de matar
```

**Template:**
```bash
#!/bin/bash
# monitor_proceso.sh - Monitor de memoria para procesos intensivos

PID=$1

if [ -z "$PID" ]; then
    echo "❌ Error: Debes proporcionar el PID del proceso a monitorear"
    echo "Uso: $0 <PID>"
    exit 1
fi

# Configuración
MEMORY_THRESHOLD=85
MEMORY_CRITICAL=92
CHECK_INTERVAL=10
MAX_ALERTS=3

echo "🛡️  Monitor de memoria iniciado"
echo "   PID monitoreado: $PID"
echo "   Umbral alerta:   ${MEMORY_THRESHOLD}%"
echo "   Umbral crítico:  ${MEMORY_CRITICAL}%"
echo "   Intervalo:       ${CHECK_INTERVAL}s"
echo ""

ALERT_COUNT=0

while true; do
    # Verificar si el proceso sigue vivo
    if ! kill -0 $PID 2>/dev/null; then
        echo ""
        echo "✓ Proceso finalizado normalmente (PID: $PID)"
        exit 0
    fi

    # Obtener uso de memoria del sistema
    MEM_USED=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')

    # Obtener uso de memoria del proceso específico (en MB)
    PROC_MEM=$(ps -p $PID -o rss= 2>/dev/null | awk '{printf "%.0f", $1/1024}')

    # Obtener memoria disponible (en GB)
    MEM_AVAIL=$(free -g | grep Mem | awk '{print $7}')

    # Timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Mostrar estado
    echo "[$TIMESTAMP] Memoria sistema: ${MEM_USED}% | Proceso: ${PROC_MEM} MB | Disponible: ${MEM_AVAIL} GB"

    # Verificar umbral crítico
    if [ "$MEM_USED" -ge "$MEMORY_CRITICAL" ] || [ "$MEM_AVAIL" -le 0 ]; then
        echo ""
        echo "🚨 ¡MEMORIA CRÍTICA! (${MEM_USED}% usado)"
        echo "   Deteniendo proceso $PID de forma segura..."

        # Intentar detención suave primero
        kill -TERM $PID 2>/dev/null
        sleep 5

        # Si aún existe, forzar
        if kill -0 $PID 2>/dev/null; then
            echo "   Proceso no responde, forzando detención..."
            kill -9 $PID 2>/dev/null
        fi

        echo ""
        echo "⚠️  PROCESO DETENIDO POR USO EXCESIVO DE MEMORIA"
        exit 2

    # Verificar umbral de alerta
    elif [ "$MEM_USED" -ge "$MEMORY_THRESHOLD" ]; then
        ALERT_COUNT=$((ALERT_COUNT + 1))
        echo "   ⚠️  ALERTA: Uso de memoria elevado (${MEM_USED}%) - Alerta ${ALERT_COUNT}/${MAX_ALERTS}"

        if [ "$ALERT_COUNT" -ge "$MAX_ALERTS" ]; then
            echo ""
            echo "🚨 ¡ALERTAS CONSECUTIVAS MÁXIMAS ALCANZADAS!"
            echo "   Deteniendo proceso preventivamente..."
            kill -TERM $PID 2>/dev/null
            sleep 5
            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID 2>/dev/null
            fi
            echo ""
            echo "⚠️  PROCESO DETENIDO PREVENTIVAMENTE"
            exit 3
        fi
    else
        # Resetear contador si la memoria vuelve a niveles seguros
        if [ "$ALERT_COUNT" -gt 0 ]; then
            echo "   ✓ Memoria regresó a niveles normales"
        fi
        ALERT_COUNT=0
    fi

    sleep $CHECK_INTERVAL
done
```

**Códigos de salida:**
- `0`: Proceso terminó normalmente
- `2`: Proceso detenido por memoria crítica (>92%)
- `3`: Proceso detenido preventivamente (3+ alertas >85%)

---

### 2. Script Wrapper de Ejecución Segura (`run_proceso_seguro.sh`)

**Propósito:** Lanzar un proceso pesado con verificación previa y monitoreo automático.

**Características:**
- Verifica memoria disponible antes de empezar
- Lanza el proceso en background
- Inicia el monitor automáticamente
- Interpreta resultados y da recomendaciones

**Template:**
```bash
#!/bin/bash
# run_proceso_seguro.sh - Wrapper para ejecución segura de procesos intensivos

echo "========================================================================"
echo "🛡️  EJECUCIÓN SEGURA - [NOMBRE DEL PROCESO]"
echo "========================================================================"
echo ""

# ============================================================================
# CONFIGURACIÓN - PERSONALIZAR SEGÚN TU PROYECTO
# ============================================================================

SCRIPT_A_EJECUTAR="python3 mi_script_pesado.py"
LOG_OUTPUT="resultados/proceso.log"
MEMORIA_MINIMA_GB=4  # Memoria mínima recomendada

# ============================================================================
# VERIFICACIÓN PREVIA
# ============================================================================

# Verificar memoria disponible antes de empezar
MEM_TOTAL=$(free -g | grep Mem | awk '{print $2}')
MEM_AVAIL=$(free -g | grep Mem | awk '{print $7}')

echo "→ Verificando recursos del sistema..."
echo "  Memoria total:       ${MEM_TOTAL} GB"
echo "  Memoria disponible:  ${MEM_AVAIL} GB"
echo ""

# Advertir si hay poca memoria
if [ "$MEM_AVAIL" -lt "$MEMORIA_MINIMA_GB" ]; then
    echo "⚠️  ADVERTENCIA: Memoria disponible baja (${MEM_AVAIL} GB)"
    echo ""
    echo "   Recomendaciones:"
    echo "   1. Cierra navegadores web y aplicaciones pesadas"
    echo "   2. Considera reducir parámetros del proceso (ej. iteraciones)"
    echo "   3. Reinicia el sistema para liberar memoria"
    echo ""
    read -p "¿Deseas continuar de todos modos? (s/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Operación cancelada por el usuario"
        exit 1
    fi
fi

# ============================================================================
# EJECUCIÓN DEL PROCESO
# ============================================================================

echo "→ Iniciando proceso en segundo plano..."
echo ""

# Activar entorno virtual si es necesario
# source venv/bin/activate

# Ejecutar proceso en background
nohup $SCRIPT_A_EJECUTAR > $LOG_OUTPUT 2>&1 &
PROCESO_PID=$!

echo "✓ Proceso iniciado (PID: $PROCESO_PID)"
echo ""

# Dar tiempo al proceso para iniciar
sleep 3

# Verificar que el proceso se inició correctamente
if ! kill -0 $PROCESO_PID 2>/dev/null; then
    echo "❌ Error: El proceso no se inició correctamente"
    echo ""
    echo "Revisa el log en:"
    echo "  $LOG_OUTPUT"
    exit 1
fi

# ============================================================================
# INICIAR MONITOR DE MEMORIA
# ============================================================================

echo "→ Iniciando monitor de memoria..."
echo ""

./monitor_proceso.sh $PROCESO_PID

# Capturar código de salida del monitor
MONITOR_EXIT=$?

# ============================================================================
# INTERPRETACIÓN DE RESULTADOS
# ============================================================================

echo ""
echo "========================================================================"

if [ $MONITOR_EXIT -eq 0 ]; then
    echo "✅ PROCESO COMPLETADO EXITOSAMENTE"
    echo ""
    echo "📁 Revisa los resultados en:"
    echo "   [RUTA A TUS RESULTADOS]"
    echo ""
    echo "📊 Log completo:"
    echo "   $LOG_OUTPUT"

elif [ $MONITOR_EXIT -eq 2 ]; then
    echo "⚠️  PROCESO DETENIDO POR USO CRÍTICO DE MEMORIA"
    echo ""
    echo "El sistema estaba a punto de congelarse. El proceso fue detenido"
    echo "de forma segura para proteger tu computador."
    echo ""
    echo "💡 Soluciones:"
    echo "   1. Reduce parámetros del proceso (ej. menos iteraciones)"
    echo "   2. Cierra todas las aplicaciones no esenciales"
    echo "   3. Reinicia el sistema para liberar memoria"
    echo "   4. Considera usar una máquina con más RAM"
    echo ""
    echo "📝 Revisa el log parcial en:"
    echo "   $LOG_OUTPUT"

elif [ $MONITOR_EXIT -eq 3 ]; then
    echo "⚠️  PROCESO DETENIDO PREVENTIVAMENTE"
    echo ""
    echo "Se detectaron múltiples alertas de memoria alta."
    echo ""
    echo "💡 Soluciones similares al caso anterior"

else
    echo "❌ ERROR EN EL MONITOREO"
    echo ""
    echo "Revisa el log en:"
    echo "  $LOG_OUTPUT"
fi

echo "========================================================================"
echo ""
```

---

### 3. Protección Interna en Python

**Propósito:** Monitoreo y limpieza de memoria desde dentro del proceso.

**Librerías necesarias:**
```bash
pip install psutil
```

**Template de funciones reutilizables:**

```python
import gc
import psutil
import os
import ctypes
import time

# ============================================================================
# FUNCIONES DE GESTIÓN DE MEMORIA
# ============================================================================

def get_memory_usage():
    """
    Retorna uso de memoria del proceso y del sistema.

    Returns:
        tuple: (memoria_proceso_gb, memoria_sistema_percent)
    """
    # Memoria del proceso actual
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)

    # Memoria del sistema
    virtual_memory = psutil.virtual_memory()
    mem_percent = virtual_memory.percent

    return mem_gb, mem_percent


def force_cleanup():
    """
    Limpieza agresiva de memoria.

    - Fuerza garbage collection
    - Libera memoria no usada al SO (solo Linux)
    """
    # Forzar garbage collection de Python
    gc.collect()

    # Intentar liberar memoria no usada (solo Linux)
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass  # No disponible en Windows/Mac


def check_memory_safety(threshold_critical=90, threshold_alert=80):
    """
    Verifica si es seguro continuar con el proceso.

    Args:
        threshold_critical (int): % de RAM para detener (default: 90)
        threshold_alert (int): % de RAM para alertar (default: 80)

    Returns:
        bool: True si es seguro continuar, False si debe detenerse
    """
    mem_percent = psutil.virtual_memory().percent

    if mem_percent > threshold_critical:
        print(f"\n⚠️  MEMORIA CRÍTICA: {mem_percent:.1f}%")
        print("   Deteniendo proceso por seguridad")
        return False

    elif mem_percent > threshold_alert:
        print(f"\n⚠️  Memoria alta: {mem_percent:.1f}% - Realizando limpieza...")
        force_cleanup()
        time.sleep(2)  # Dar tiempo a que se libere memoria

    return True


def log_memory_status(iteration=None, prefix=""):
    """
    Muestra estado de memoria en consola.

    Args:
        iteration (int, optional): Número de iteración
        prefix (str, optional): Prefijo para el mensaje
    """
    mem_gb, mem_percent = get_memory_usage()

    iter_str = f"Iteración #{iteration} - " if iteration else ""
    print(f"\n💾 {prefix}{iter_str}Memoria proceso: {mem_gb:.2f} GB | Sistema: {mem_percent:.1f}%")


# ============================================================================
# EJEMPLO DE USO EN LOOP DE PROCESAMIENTO
# ============================================================================

def procesar_datos_pesados(n_iteraciones=100):
    """
    Ejemplo de función que procesa datos de forma intensiva.
    """

    print("Iniciando procesamiento con protección de memoria...")

    for i in range(n_iteraciones):
        # Verificar memoria antes de cada iteración
        if not check_memory_safety():
            print(f"\n❌ Procesamiento detenido en iteración {i} por memoria")
            break

        print(f"\n{'='*70}")
        print(f"Iteración {i+1}/{n_iteraciones}")
        print(f"{'='*70}")

        # === TU CÓDIGO PESADO AQUÍ ===
        # Ejemplo: cargar datos grandes, hacer cálculos, etc.
        resultado = hacer_algo_pesado()

        # === FIN DE TU CÓDIGO ===

        # Mostrar estado de memoria
        log_memory_status(iteration=i+1)

        # CRÍTICO: Liberar memoria explícitamente
        del resultado  # Eliminar objetos grandes
        force_cleanup()  # Forzar garbage collection

        # Verificación periódica adicional (opcional)
        if (i + 1) % 10 == 0:
            mem_gb_after, mem_percent_after = get_memory_usage()
            print(f"💾 Memoria después de limpieza: {mem_gb_after:.2f} GB | Sistema: {mem_percent_after:.1f}%")

    print("\n✅ Procesamiento completado")


# ============================================================================
# PATRÓN PARA OPTIMIZACIÓN BAYESIANA
# ============================================================================

def objective_function_con_proteccion(params):
    """
    Función objetivo para optimización bayesiana con protección de memoria.

    Args:
        params (dict): Parámetros a evaluar

    Returns:
        float: Valor del objetivo (o penalización si hay error)
    """

    # Verificar memoria antes de empezar
    if not check_memory_safety():
        return float('inf')  # Penalización máxima

    try:
        # === TU EVALUACIÓN AQUÍ ===
        resultado = evaluar_parametros(params)
        # === FIN ===

        # Liberar memoria
        force_cleanup()

        return resultado

    except Exception as e:
        print(f"\n❌ Error: {e}")
        force_cleanup()
        return float('inf')  # Penalización


# ============================================================================
# PATRÓN PARA MODELOS AGENT-BASED (Mesa)
# ============================================================================

def run_simulation_con_proteccion(model, n_steps=500, check_every=100):
    """
    Ejecuta simulación ABM con verificación periódica de memoria.

    Args:
        model: Modelo Mesa
        n_steps (int): Número de steps
        check_every (int): Cada cuántos steps verificar memoria

    Returns:
        bool: True si completó, False si se detuvo
    """

    for step in range(n_steps):
        model.step()

        # Verificar memoria periódicamente
        if step % check_every == 0 and step > 0:
            if not check_memory_safety():
                print(f"\n⚠️  Simulación detenida en step {step} por memoria")
                return False

    return True


# ============================================================================
# DECORADOR PARA FUNCIONES
# ============================================================================

def memory_safe(func):
    """
    Decorador para agregar protección de memoria a cualquier función.

    Usage:
        @memory_safe
        def mi_funcion_pesada(args):
            # código aquí
            pass
    """
    def wrapper(*args, **kwargs):
        # Verificar antes
        if not check_memory_safety():
            raise MemoryError("Memoria insuficiente para ejecutar función")

        # Ejecutar
        try:
            result = func(*args, **kwargs)
        finally:
            # Limpiar después
            force_cleanup()

        return result

    return wrapper
```

**Ejemplo de uso del decorador:**

```python
@memory_safe
def procesar_raster_gigante(raster_path):
    """Esta función está protegida automáticamente"""
    with rasterio.open(raster_path) as src:
        data = src.read()
        # procesamiento pesado...
    return resultado
```

---

### 4. Protección Interna en R

**Librerías necesarias:**
```r
# No requiere instalación, usa funciones base de R
```

**Template de funciones reutilizables:**

```r
# ============================================================================
# FUNCIONES DE GESTIÓN DE MEMORIA EN R
# ============================================================================

#' Obtener uso de memoria del proceso en GB
#'
#' @return Uso de memoria en GB
get_memory_usage <- function() {
  # Obtener memoria usada por objetos en el entorno
  mem_used_bytes <- sum(sapply(ls(envir = .GlobalEnv), function(x) {
    object.size(get(x, envir = .GlobalEnv))
  }))

  mem_used_gb <- mem_used_bytes / (1024^3)

  return(mem_used_gb)
}


#' Limpieza forzada de memoria
#'
#' @details Ejecuta garbage collection y libera memoria no usada
force_cleanup <- function() {
  # Garbage collection
  gc(verbose = FALSE, full = TRUE)

  # Intentar liberar más memoria (solo Linux)
  if (Sys.info()['sysname'] == "Linux") {
    try({
      system("sync; echo 3 > /proc/sys/vm/drop_caches", ignore.stderr = TRUE)
    }, silent = TRUE)
  }
}


#' Verificar si es seguro continuar
#'
#' @param threshold_gb Umbral de memoria en GB
#' @return TRUE si es seguro, FALSE si debe detenerse
check_memory_safety <- function(threshold_gb = 25) {
  mem_info <- system("free -g | grep Mem", intern = TRUE)
  mem_parts <- as.numeric(unlist(strsplit(trimws(mem_info), "\\s+")))

  # mem_parts: total, used, free, shared, buff/cache, available
  mem_available_gb <- mem_parts[7]

  if (mem_available_gb < 2) {
    cat("\n⚠️  MEMORIA CRÍTICA: Solo", mem_available_gb, "GB disponibles\n")
    cat("   Deteniendo proceso por seguridad\n")
    return(FALSE)
  }

  if (mem_available_gb < 5) {
    cat("\n⚠️  Memoria baja:", mem_available_gb, "GB disponibles - Limpiando...\n")
    force_cleanup()
    Sys.sleep(2)
  }

  return(TRUE)
}


#' Log de estado de memoria
#'
#' @param iteration Número de iteración (opcional)
#' @param prefix Prefijo del mensaje
log_memory_status <- function(iteration = NULL, prefix = "") {
  mem_gb <- get_memory_usage()

  mem_info <- system("free -g | grep Mem", intern = TRUE)
  mem_parts <- as.numeric(unlist(strsplit(trimws(mem_info), "\\s+")))
  mem_available_gb <- mem_parts[7]

  iter_str <- if (!is.null(iteration)) paste0("Iteración #", iteration, " - ") else ""

  cat(sprintf("\n💾 %s%sMemoria proceso: %.2f GB | Disponible: %d GB\n",
              prefix, iter_str, mem_gb, mem_available_gb))
}


# ============================================================================
# EJEMPLO DE USO EN LOOP DE PROCESAMIENTO
# ============================================================================

procesar_datos_pesados <- function(n_iteraciones = 100) {
  cat("Iniciando procesamiento con protección de memoria...\n")

  for (i in 1:n_iteraciones) {
    # Verificar memoria antes de cada iteración
    if (!check_memory_safety()) {
      cat(sprintf("\n❌ Procesamiento detenido en iteración %d por memoria\n", i))
      break
    }

    cat(sprintf("\n%s\nIteración %d/%d\n%s\n",
                paste(rep("=", 70), collapse = ""), i, n_iteraciones,
                paste(rep("=", 70), collapse = "")))

    # === TU CÓDIGO PESADO AQUÍ ===
    resultado <- hacer_algo_pesado()
    # === FIN ===

    # Mostrar estado de memoria
    log_memory_status(iteration = i)

    # CRÍTICO: Liberar memoria explícitamente
    rm(resultado)
    force_cleanup()

    # Verificación periódica adicional
    if (i %% 10 == 0) {
      cat(sprintf("💾 Verificación cada 10 iteraciones: %.2f GB usados\n",
                  get_memory_usage()))
    }
  }

  cat("\n✅ Procesamiento completado\n")
}


# ============================================================================
# PATRÓN PARA PROCESAMIENTO POR BLOQUES (RASTERS)
# ============================================================================

procesar_raster_por_bloques <- function(raster_path, output_path,
                                        block_size = 1000) {

  require(terra)

  r <- rast(raster_path)

  # Crear raster de salida
  r_out <- rast(r)

  # Obtener dimensiones
  nr <- nrow(r)
  nc <- ncol(r)

  # Calcular número de bloques
  n_blocks <- ceiling(nr / block_size)

  cat(sprintf("Procesando raster en %d bloques de %d filas\n", n_blocks, block_size))

  for (block in 1:n_blocks) {
    # Verificar memoria
    if (!check_memory_safety()) {
      cat(sprintf("\n❌ Procesamiento detenido en bloque %d por memoria\n", block))
      break
    }

    # Calcular filas del bloque
    row_start <- (block - 1) * block_size + 1
    row_end <- min(block * block_size, nr)

    cat(sprintf("Bloque %d/%d (filas %d-%d)\n", block, n_blocks, row_start, row_end))

    # Leer bloque
    bloque_data <- r[row_start:row_end, ]

    # === PROCESAR BLOQUE ===
    bloque_procesado <- procesar_bloque(bloque_data)
    # === FIN ===

    # Escribir resultado
    r_out[row_start:row_end, ] <- bloque_procesado

    # Liberar memoria
    rm(bloque_data, bloque_procesado)
    force_cleanup()

    log_memory_status(iteration = block, prefix = "Bloque ")
  }

  # Guardar resultado
  writeRaster(r_out, output_path, overwrite = TRUE)

  cat("\n✅ Raster procesado y guardado\n")
}
```

---

## Implementación Paso a Paso

### Para Python:

**1. Crear estructura de archivos:**
```bash
mi_proyecto/
├── monitor_proceso.sh           # Monitor externo
├── run_proceso_seguro.sh        # Wrapper de ejecución
├── mi_script_pesado.py          # Tu script principal
└── utils/
    └── memory_utils.py          # Funciones de memoria
```

**2. Copiar funciones de protección:**
```bash
# Copiar las funciones de memoria a utils/memory_utils.py
# Importar en tu script:
from utils.memory_utils import check_memory_safety, force_cleanup, log_memory_status
```

**3. Modificar tu código:**
```python
# Antes de cada operación pesada:
if not check_memory_safety():
    break

# Después de cada operación:
del objeto_grande
force_cleanup()
log_memory_status(iteration=i)
```

**4. Hacer scripts ejecutables:**
```bash
chmod +x monitor_proceso.sh
chmod +x run_proceso_seguro.sh
```

**5. Ejecutar:**
```bash
./run_proceso_seguro.sh
```

---

### Para R:

**1. Crear estructura:**
```bash
mi_proyecto/
├── monitor_proceso.sh           # Monitor externo
├── run_proceso_seguro.sh        # Wrapper de ejecución
├── mi_script_pesado.R           # Tu script principal
└── utils/
    └── memory_utils.R           # Funciones de memoria
```

**2. Cargar funciones en tu script:**
```r
source("utils/memory_utils.R")

# Usar en tu código:
if (!check_memory_safety()) {
  stop("Memoria insuficiente")
}

# Después de operaciones pesadas:
rm(objeto_grande)
force_cleanup()
log_memory_status(iteration = i)
```

**3. Modificar wrapper para R:**
```bash
# En run_proceso_seguro.sh, cambiar:
SCRIPT_A_EJECUTAR="Rscript mi_script_pesado.R"
```

---

## Ejemplos de Uso

### Ejemplo 1: Optimización Bayesiana en Python

```python
from skopt import gp_minimize
from utils.memory_utils import check_memory_safety, force_cleanup

iteration_count = 0

def objective(params):
    global iteration_count
    iteration_count += 1

    # Verificar memoria
    if not check_memory_safety():
        return float('inf')

    # Tu evaluación
    resultado = evaluar_modelo(params)

    # Limpiar
    force_cleanup()

    return -resultado  # Minimizar

# Ejecutar optimización
result = gp_minimize(objective, search_space, n_calls=50)
```

**Lanzar con:**
```bash
./run_proceso_seguro.sh
```

---

### Ejemplo 2: Procesamiento de Rasters en R

```r
source("utils/memory_utils.R")

library(terra)

# Procesar múltiples rasters
raster_files <- list.files("datos/", pattern = "*.tif", full.names = TRUE)

for (i in seq_along(raster_files)) {

  # Verificar memoria antes de cargar
  if (!check_memory_safety()) {
    cat(sprintf("Detenido en archivo %d por memoria\n", i))
    break
  }

  cat(sprintf("\nProcesando %d/%d: %s\n", i, length(raster_files),
              basename(raster_files[i])))

  # Cargar y procesar
  r <- rast(raster_files[i])
  resultado <- procesar_raster(r)

  # Guardar
  writeRaster(resultado, sprintf("salida/resultado_%d.tif", i))

  # CRÍTICO: Limpiar
  rm(r, resultado)
  force_cleanup()

  log_memory_status(iteration = i)
}
```

---

### Ejemplo 3: Simulación Agent-Based (Mesa)

```python
from mesa import Model, Agent
from utils.memory_utils import check_memory_safety, force_cleanup

class MiModelo(Model):
    def step(self):
        # Tu lógica de step
        for agent in self.agents:
            agent.step()

# Ejecutar simulación con protección
model = MiModelo(n_agents=10000)

for step in range(1000):
    # Verificar cada 100 steps
    if step % 100 == 0:
        if not check_memory_safety():
            print(f"Simulación detenida en step {step}")
            break

    model.step()

    # Limpiar periódicamente
    if step % 100 == 0:
        force_cleanup()

# Limpiar al final
del model
force_cleanup()
```

---

## Adaptación a Otros Proyectos

### Checklist de Adaptación:

- [ ] **1. Copiar archivos base:**
  - `monitor_proceso.sh`
  - `run_proceso_seguro.sh`
  - Funciones de memoria (Python o R)

- [ ] **2. Personalizar wrapper:**
  - Cambiar `SCRIPT_A_EJECUTAR` a tu comando
  - Ajustar `MEMORIA_MINIMA_GB` según tu caso
  - Modificar rutas de logs y resultados

- [ ] **3. Integrar en tu código:**
  - Agregar `check_memory_safety()` antes de operaciones pesadas
  - Agregar `force_cleanup()` después de liberar objetos
  - Agregar `log_memory_status()` para monitoreo

- [ ] **4. Ajustar umbrales:**
  - Modificar `MEMORY_THRESHOLD` y `MEMORY_CRITICAL` según tu RAM
  - Ajustar `CHECK_INTERVAL` según duración de iteraciones

- [ ] **5. Probar:**
  - Ejecutar con `./run_proceso_seguro.sh`
  - Verificar que el monitor funciona
  - Comprobar que los logs se generan correctamente

---

### Configuración por Tipo de Tarea:

| Tipo de Tarea | MEMORY_THRESHOLD | MEMORY_CRITICAL | CHECK_INTERVAL |
|---------------|------------------|-----------------|----------------|
| **Optimización rápida** (iters <5min) | 85% | 92% | 30s |
| **Optimización lenta** (iters >5min) | 80% | 90% | 10s |
| **Procesamiento batch** | 85% | 92% | 60s |
| **Simulación larga** | 80% | 90% | 10s |
| **Machine Learning** | 85% | 92% | 30s |

**Regla general:**
- **MEMORY_THRESHOLD:** 80-85% (alerta temprana)
- **MEMORY_CRITICAL:** 90-92% (detención inmediata)
- **CHECK_INTERVAL:** Menor para procesos que consumen memoria rápido

---

## Troubleshooting

### Problema 1: Script no ejecuta

**Síntomas:**
```bash
bash: ./run_proceso_seguro.sh: Permission denied
```

**Solución:**
```bash
chmod +x run_proceso_seguro.sh
chmod +x monitor_proceso.sh
```

---

### Problema 2: Caracteres extraños en script

**Síntomas:**
```bash
$'\r': orden no encontrada
error sintáctico cerca del elemento inesperado
```

**Causa:** Archivo creado en Windows con CRLF

**Solución:**
```bash
dos2unix run_proceso_seguro.sh
# o
sed -i 's/\r$//' run_proceso_seguro.sh
```

---

### Problema 3: Monitor no encuentra el proceso

**Síntomas:**
```bash
✓ Proceso finalizado normalmente (PID: XXXX)
```
Pero el proceso no había terminado.

**Causa:** PID incorrecto

**Solución:**
Verificar que `$PROCESO_PID` captura el PID correcto:
```bash
echo "PID capturado: $PROCESO_PID"
ps aux | grep $PROCESO_PID
```

---

### Problema 4: Limpieza de memoria no funciona (Linux)

**Síntomas:** `malloc_trim` no reduce memoria

**Causa:** Requiere permisos o no disponible en todos los sistemas

**Solución:**
Es normal, la función hace "best effort". El GC de Python/R es lo más importante.

---

### Problema 5: Proceso se detiene demasiado pronto

**Síntomas:** Monitor mata el proceso pero la memoria parece baja

**Causa:** Umbrales muy conservadores o memoria swap no visible

**Solución:**
```bash
# Ajustar umbrales en monitor_proceso.sh:
MEMORY_THRESHOLD=90  # Aumentar de 85 a 90
MEMORY_CRITICAL=95   # Aumentar de 92 a 95
```

---

### Problema 6: Python no libera memoria después de `del`

**Síntomas:** Memoria del proceso no baja después de `force_cleanup()`

**Causa:** Python no devuelve memoria al SO inmediatamente

**Solución:**
- Es comportamiento normal
- La memoria se reutilizará en iteraciones siguientes
- El monitor vigila la memoria **del sistema**, no solo del proceso
- Si el sistema está OK, el proceso puede seguir

---

## Mejores Prácticas

### ✅ DO:

1. **Siempre verifica memoria antes de operaciones grandes:**
   ```python
   if not check_memory_safety():
       break
   ```

2. **Limpia explícitamente objetos grandes:**
   ```python
   del modelo, datos, resultado
   force_cleanup()
   ```

3. **Usa procesamiento por bloques para datos gigantes:**
   ```python
   for bloque in bloques:
       procesar_bloque(bloque)
       force_cleanup()
   ```

4. **Monitorea progreso en logs:**
   ```python
   log_memory_status(iteration=i)
   ```

5. **Guarda resultados parciales frecuentemente:**
   ```python
   if i % 10 == 0:
       guardar_checkpoint(resultados)
   ```

---

### ❌ DON'T:

1. **No acumules todos los resultados en memoria:**
   ```python
   # ❌ MAL
   resultados = []
   for i in range(1000):
       resultados.append(calcular(i))  # Acumula en RAM

   # ✅ BIEN
   for i in range(1000):
       resultado = calcular(i)
       guardar_resultado(resultado, f"iter_{i}.pkl")
       del resultado
   ```

2. **No confíes solo en Python/R GC:**
   ```python
   # ❌ MAL
   for i in range(100):
       obj = crear_objeto_grande()
       # Confía en que Python lo limpiará...

   # ✅ BIEN
   for i in range(100):
       obj = crear_objeto_grande()
       procesar(obj)
       del obj
       force_cleanup()
   ```

3. **No ignores las advertencias del monitor:**
   ```bash
   ⚠️  Memoria alta: 87% - Realizando limpieza...
   # Esto significa que estás cerca del límite
   # Considera reducir parámetros
   ```

4. **No uses umbrales muy altos en sistemas compartidos:**
   ```bash
   # ❌ MAL en servidor compartido
   MEMORY_CRITICAL=98  # Demasiado alto

   # ✅ BIEN
   MEMORY_CRITICAL=85  # Conservador en entorno compartido
   ```

---

## Casos de Uso Documentados

### Caso 1: Calibración de Modelo en Copiapó

**Contexto:** Optimización bayesiana de 40 iteraciones, cada una ejecuta simulación ABM de 500 steps con 100 agentes.

**Problema:** Después de 10 iteraciones, sistema se congeló (necesitó reinicio forzoso).

**Solución Implementada:**
1. Monitor externo cada 10s (85%/92% umbrales)
2. Protección interna en Python con `psutil`
3. Garbage collection forzado después de cada iteración
4. Liberación explícita de objetos grandes

**Resultado:**
- ✅ Completó 40 iteraciones sin congelamiento
- ✅ Memoria máxima: 42% del sistema
- ✅ Proceso protegido todo el tiempo
- ✅ Guardó checkpoints cada iteración

**Código:** Ver `calibrate_copiapo_optimized.py`

---

## Conclusiones

Este sistema de doble protección ha demostrado ser efectivo en prevenir congelamientos del sistema durante procesos computacionales intensivos. La combinación de monitoreo externo (shell) e interno (Python/R) proporciona defensa en profundidad contra consumo excesivo de memoria.

**Beneficios principales:**
- ✅ **Seguridad:** No más sistemas congelados
- ✅ **Recuperabilidad:** Detención suave preserva trabajo parcial
- ✅ **Visibilidad:** Logs detallados de uso de memoria
- ✅ **Prevención:** Detección temprana antes del colapso
- ✅ **Reutilizable:** Templates adaptables a cualquier proyecto

**Mantenimiento:**
- Revisa los umbrales periódicamente según tu hardware
- Actualiza las funciones de limpieza si encuentras mejores técnicas
- Documenta casos de uso específicos de tu proyecto

---

## Referencias

- **psutil documentation:** https://psutil.readthedocs.io/
- **Python garbage collection:** https://docs.python.org/3/library/gc.html
- **Linux memory management:** https://www.kernel.org/doc/html/latest/admin-guide/mm/
- **R memory management:** https://adv-r.hadley.nz/names-values.html

---

## Historial de Versiones

- **v1.0 (2025-10-31):** Versión inicial basada en caso de optimización Copiapó
- Próximas versiones incluirán soporte para clusters y ejecución distribuida

---

**Última actualización:** 2025-10-31
**Autor:** Metodología desarrollada durante proyecto de calibración de modelo de remociones en masa, Copiapó, Chile
**Licencia:** Reutilizable para cualquier proyecto de investigación
