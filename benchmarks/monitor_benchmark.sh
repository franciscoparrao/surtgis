#!/bin/bash
# monitor_benchmark.sh - Monitor de memoria para benchmarks SurtGIS
# Capa 2 de protección: vigila el proceso desde fuera y lo mata si es necesario.
#
# Uso:
#   ./benchmarks/monitor_benchmark.sh <PID>
#   ./benchmarks/monitor_benchmark.sh  (auto-detecta el proceso Python)

PID=$1

# Auto-detectar si no se pasa PID
if [ -z "$PID" ]; then
    PID=$(pgrep -f "run_benchmarks.py" | head -1)
    if [ -z "$PID" ]; then
        echo "Error: No se encontro proceso de benchmark activo"
        echo "Uso: $0 <PID>"
        exit 1
    fi
    echo "Auto-detectado PID: $PID"
fi

# Configuracion
MEMORY_THRESHOLD=88    # % RAM para alerta (la proteccion Python actua al 80%)
MEMORY_CRITICAL=93     # % RAM para matar proceso (Python actua al 90%)
CHECK_INTERVAL=10      # Segundos entre verificaciones
MAX_ALERTS=6           # Alertas consecutivas antes de matar (60s sostenido)
LOG_FILE="benchmarks/results/memory_monitor.log"

echo "Monitor de memoria para benchmarks"
echo "  PID monitoreado: $PID"
echo "  Umbral alerta:   ${MEMORY_THRESHOLD}%"
echo "  Umbral critico:  ${MEMORY_CRITICAL}%"
echo "  Intervalo:       ${CHECK_INTERVAL}s"
echo "  Log:             ${LOG_FILE}"
echo ""

mkdir -p benchmarks/results

ALERT_COUNT=0

while true; do
    # Verificar si el proceso sigue vivo
    if ! kill -0 $PID 2>/dev/null; then
        echo ""
        echo "OK: Proceso finalizado normalmente (PID: $PID)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISHED normally" >> "$LOG_FILE"
        exit 0
    fi

    # Obtener uso de memoria del sistema
    MEM_USED=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')

    # Obtener uso de memoria del proceso especifico (en MB)
    PROC_MEM=$(ps -p $PID -o rss= 2>/dev/null | awk '{printf "%.0f", $1/1024}')

    # Obtener memoria disponible (en GB)
    MEM_AVAIL=$(free -g | grep Mem | awk '{print $7}')

    # Timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Mostrar y loguear estado
    STATUS="[$TIMESTAMP] Sistema: ${MEM_USED}% | Proceso: ${PROC_MEM}MB | Disponible: ${MEM_AVAIL}GB"
    echo "$STATUS"
    echo "$STATUS" >> "$LOG_FILE"

    # Verificar umbral critico
    if [ "$MEM_USED" -ge "$MEMORY_CRITICAL" ] || [ "$MEM_AVAIL" -le 1 ]; then
        echo ""
        echo "*** MEMORIA CRITICA! (${MEM_USED}% usado, ${MEM_AVAIL}GB libre) ***"
        echo "    Deteniendo proceso $PID..."
        echo "[$TIMESTAMP] CRITICAL: ${MEM_USED}% - KILLING $PID" >> "$LOG_FILE"

        # Intentar detencion suave primero (SIGTERM)
        kill -TERM $PID 2>/dev/null
        sleep 5

        # Si aun existe, forzar
        if kill -0 $PID 2>/dev/null; then
            echo "    Proceso no responde, forzando..."
            kill -9 $PID 2>/dev/null
        fi

        echo ""
        echo "PROCESO DETENIDO POR USO EXCESIVO DE MEMORIA"
        echo "Los resultados parciales estan en benchmarks/results/ (CSV incremental)"
        exit 2

    # Verificar umbral de alerta
    elif [ "$MEM_USED" -ge "$MEMORY_THRESHOLD" ]; then
        ALERT_COUNT=$((ALERT_COUNT + 1))
        echo "  ** ALERTA: Memoria elevada (${MEM_USED}%) - ${ALERT_COUNT}/${MAX_ALERTS} **"
        echo "[$TIMESTAMP] ALERT #${ALERT_COUNT}: ${MEM_USED}%" >> "$LOG_FILE"

        if [ "$ALERT_COUNT" -ge "$MAX_ALERTS" ]; then
            echo ""
            echo "*** ALERTAS MAXIMAS ALCANZADAS ***"
            echo "    Deteniendo proceso preventivamente..."
            echo "[$TIMESTAMP] MAX_ALERTS reached - KILLING $PID" >> "$LOG_FILE"

            kill -TERM $PID 2>/dev/null
            sleep 5
            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID 2>/dev/null
            fi
            echo ""
            echo "PROCESO DETENIDO PREVENTIVAMENTE"
            echo "Los resultados parciales estan en benchmarks/results/ (CSV incremental)"
            exit 3
        fi
    else
        # Resetear contador si la memoria vuelve a niveles seguros
        if [ "$ALERT_COUNT" -gt 0 ]; then
            echo "  OK: Memoria regreso a niveles normales"
            echo "[$TIMESTAMP] Memory back to normal" >> "$LOG_FILE"
        fi
        ALERT_COUNT=0
    fi

    sleep $CHECK_INTERVAL
done
