#!/bin/bash
# ─── SurtGIS - Demo ──────────────────────────────────────
# Levanta la demo web en tu navegador.
#
# Requisitos:
#   - Python 3
#   - ngrok (instalado)
#
# Uso:
#   ./demo.sh
#   Abre: https://surtgis-demo.ngrok-free.dev
# ──────────────────────────────────────────────────────────

set -e

cd "$(dirname "$0")/surtgis-demo"

echo "=== SurtGIS Web Demo ==="
echo ""

# Iniciar HTTP server en background
echo "▶ Iniciando servidor HTTP en puerto 9999..."
python3 -m http.server 9999 > /tmp/surtgis_demo_http.log 2>&1 &
HTTP_PID=$!
echo $HTTP_PID > /tmp/surtgis_demo_http.pid

sleep 2

# Iniciar ngrok con dominio personalizado
echo "▶ Iniciando ngrok (dominio: surtgis-demo.ngrok-free.dev)..."
ngrok http 9999 --domain=surtgis-demo.ngrok-free.dev --log=stdout > /tmp/surtgis_demo_ngrok.log 2>&1 &
NGROK_PID=$!
echo $NGROK_PID > /tmp/surtgis_demo_ngrok.pid

sleep 3

# Verificar que ngrok está activo
if ! ps -p $NGROK_PID > /dev/null; then
    echo ""
    echo "✗ Error iniciando ngrok. Revisa:"
    echo "  - ¿Está instalado ngrok? (ngrok --version)"
    echo "  - ¿Tienes authtoken configurado? (ngrok config check)"
    echo "  - ¿El puerto 9999 está libre?"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  ✅ SurtGIS Demo LISTA"
echo "════════════════════════════════════════════════════"
echo ""
echo "  🌐 URL PÚBLICA:"
echo "     https://surtgis-demo.ngrok-free.dev"
echo ""
echo "  📱 LOCAL (no public):"
echo "     http://localhost:9999"
echo ""
echo "  🧪 Quick Test (30 segundos):"
echo "     1. Click '📂 Load Demo DEM'"
echo "     2. Select 'Slope'"
echo "     3. Click 'Compute'"
echo "     4. Ver resultado en mapa"
echo ""
echo "  ⏹️  Para detener:"
echo "     kill $(cat /tmp/surtgis_demo_http.pid)"
echo "     kill $(cat /tmp/surtgis_demo_ngrok.pid)"
echo ""
echo "════════════════════════════════════════════════════"
echo ""

# Mantener proceso en foreground
wait
