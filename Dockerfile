FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Sistema base + GUI headless (Computer Use Agent) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    # Xvfb: display virtual para correr Chromium sin pantalla real
    xvfb x11-utils \
    # Chromium: browser para el Computer Use Agent
    chromium chromium-driver \
    # Dependencias gráficas de pyautogui / mss
    python3-xlib libxtst6 libxrandr2 \
    # tkinter requerido por pyautogui/mouseinfo
    python3-tk python3-dev \
    # xauth para autenticación con Xvfb
    xauth \
    # xdotool: click/type/key via X11 (usado por computer_use_tool.py como método primario)
    xdotool \
    # wmctrl: gestión de ventanas (focus, raise)
    wmctrl \
    # xclip: clipboard support para pegar texto largo sin xdotool type lento
    xclip \
    # Utilidades de red
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Apuntar el display virtual al que lanzaremos con Xvfb
ENV DISPLAY=:99
# Indicar a Chromium que no tiene sandbox real (necesario en Docker)
ENV CHROMIUM_FLAGS="--no-sandbox --disable-dev-shm-usage"

# Copy dependency files
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project

COPY app ./app
RUN uv sync

# Script de arranque: lanza Xvfb en background y luego uvicorn
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000

CMD ["/start.sh"]
