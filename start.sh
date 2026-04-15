#!/bin/bash
# start.sh — Lanza Xvfb (display virtual headless) y luego uvicorn.
# Necesario para que mss/pyautogui funcionen dentro del container Docker.

set -e

echo "[start.sh] Limpiando posibles lock files de Xvfb anteriores..."
# Eliminar lock file si existe (contenedor anterior crasheó)
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

# Matar cualquier Xvfb existente en :99 (si quedó zombie)
pkill -f "Xvfb :99" 2>/dev/null || true

echo "[start.sh] Iniciando Xvfb en :99 (1920x1080x24)..."
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Esperar a que Xvfb esté listo
sleep 1

# Verificar que Xvfb levantó correctamente
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "[start.sh] ERROR: Xvfb no pudo iniciar. Abortando."
    exit 1
fi

echo "[start.sh] Xvfb corriendo en PID $XVFB_PID"
echo "[start.sh] DISPLAY=$DISPLAY"
echo "[start.sh] Iniciando uvicorn..."

# Ejecutar uvicorn en foreground (reemplaza este proceso)
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
