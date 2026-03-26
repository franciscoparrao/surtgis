#!/bin/bash
# run_safe.sh - Ejecucion segura del benchmark con monitor de memoria
#
# Uso:
#   ./benchmarks/run_safe.sh              # benchmark completo
#   ./benchmarks/run_safe.sh --quick      # validacion rapida
#   ./benchmarks/run_safe.sh --experiment 1  # solo experimento 1
#
# Implementa Capa 3 de la Metodologia de Seguridad:
#   - Verificacion pre-ejecucion de memoria
#   - Lanza benchmark + monitor en paralelo
#   - Interpreta resultados

echo "========================================================================"
echo "  BENCHMARK SURTGIS - EJECUCION SEGURA"
echo "========================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

MEMORIA_MINIMA_GB=6

# ============================================================================
# VERIFICACION PREVIA
# ============================================================================

MEM_TOTAL=$(free -g | grep Mem | awk '{print $2}')
MEM_AVAIL=$(free -g | grep Mem | awk '{print $7}')

echo "-> Verificando recursos del sistema..."
echo "   Memoria total:       ${MEM_TOTAL} GB"
echo "   Memoria disponible:  ${MEM_AVAIL} GB"
echo "   Minima requerida:    ${MEMORIA_MINIMA_GB} GB"
echo ""

if [ "$MEM_AVAIL" -lt "$MEMORIA_MINIMA_GB" ]; then
    echo "** ADVERTENCIA: Memoria disponible baja (${MEM_AVAIL} GB) **"
    echo ""
    echo "   Recomendaciones:"
    echo "   1. Cierra navegadores web y aplicaciones pesadas"
    echo "   2. Usa --quick para una prueba mas ligera"
    echo "   3. Reinicia el sistema para liberar memoria"
    echo ""
    read -p "Deseas continuar de todos modos? (s/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Operacion cancelada"
        exit 1
    fi
fi

# ============================================================================
# ACTIVAR ENTORNO VIRTUAL
# ============================================================================

if [ -d "$VENV_DIR" ]; then
    echo "-> Activando entorno virtual..."
    source "$VENV_DIR/bin/activate"
else
    echo "Error: No se encontro entorno virtual en $VENV_DIR"
    echo "Ejecuta: python3 -m venv .venv && pip install numpy psutil maturin whitebox"
    exit 1
fi

# Verificar dependencias
python3 -c "import psutil" 2>/dev/null || {
    echo "Error: psutil no instalado. Ejecuta: pip install psutil"
    exit 1
}

# ============================================================================
# LIMPIAR PROCESOS PREVIOS
# ============================================================================

# Matar cualquier benchmark anterior que haya quedado colgado
pkill -f "run_benchmarks.py" 2>/dev/null
pkill -f "monitor_benchmark.sh" 2>/dev/null
sleep 1

# ============================================================================
# EJECUTAR BENCHMARK CON MONITOR
# ============================================================================

echo ""
echo "-> Iniciando benchmark en segundo plano..."
echo "   Argumentos: $@"
echo ""

# Crear directorio de resultados
mkdir -p "$SCRIPT_DIR/results"

# Ejecutar benchmark
python3 "$SCRIPT_DIR/run_benchmarks.py" "$@" > "$SCRIPT_DIR/results/benchmark.log" 2>&1 &
BENCH_PID=$!

echo "   Benchmark PID: $BENCH_PID"

# Dar tiempo al proceso para iniciar
sleep 3

# Verificar que el proceso se inicio
if ! kill -0 $BENCH_PID 2>/dev/null; then
    echo "Error: El benchmark no se inicio correctamente"
    echo "Revisa: $SCRIPT_DIR/results/benchmark.log"
    tail -20 "$SCRIPT_DIR/results/benchmark.log"
    exit 1
fi

echo "-> Iniciando monitor de memoria..."
echo ""

# Iniciar monitor (bloquea hasta que el benchmark termine o sea matado)
bash "$SCRIPT_DIR/monitor_benchmark.sh" $BENCH_PID

MONITOR_EXIT=$?

# ============================================================================
# INTERPRETACION DE RESULTADOS
# ============================================================================

echo ""
echo "========================================================================"

if [ $MONITOR_EXIT -eq 0 ]; then
    echo "  BENCHMARK COMPLETADO EXITOSAMENTE"
    echo ""
    echo "  Resultados en:"
    echo "    $SCRIPT_DIR/results/experiment1_scalability.csv"
    echo "    $SCRIPT_DIR/results/experiment2_accuracy.csv"
    echo "    $SCRIPT_DIR/results/experiment3_crossplatform.csv"
    echo ""
    echo "  Log completo:"
    echo "    $SCRIPT_DIR/results/benchmark.log"

elif [ $MONITOR_EXIT -eq 2 ] || [ $MONITOR_EXIT -eq 3 ]; then
    echo "  BENCHMARK DETENIDO POR PROTECCION DE MEMORIA"
    echo ""
    echo "  Los RESULTADOS PARCIALES se guardaron (CSV incremental):"
    echo "    $SCRIPT_DIR/results/experiment1_scalability.csv"
    echo ""
    echo "  Soluciones:"
    echo "    1. Cierra aplicaciones y vuelve a ejecutar"
    echo "    2. Usa --experiment 1 --tools surtgis,gdal (sin GRASS/WBT pesados)"
    echo "    3. Usa --quick para prueba ligera"
    echo ""
    echo "  Log parcial:"
    echo "    $SCRIPT_DIR/results/benchmark.log"
    echo ""
    echo "  Monitor de memoria:"
    echo "    $SCRIPT_DIR/results/memory_monitor.log"

else
    echo "  ERROR EN EL MONITOREO"
    echo "  Revisa: $SCRIPT_DIR/results/benchmark.log"
fi

echo "========================================================================"
echo ""
