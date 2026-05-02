FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Sistema base + GUI headless (Computer Use Agent) + CUDA runtime ---
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
    # CUDA runtime libs + OpenGL para PyTorch GPU / OpenCV (ultralytics)
    libgl1 libglib2.0-0 \
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
# GPU: permitir que PyTorch vea la GPU host cuando se usa --gpus all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- PyTorch CUDA index (for uv/pip) ---
# Si tu host tiene CUDA 12.x, PyTorch descarga wheels con soporte GPU automáticamente.
# Para forzar CUDA 12.4 wheels, descomenta la siguiente línea:
# ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# Copy dependency files
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project

COPY app ./app
RUN uv sync

# Install Playwright browser dependencies
RUN uv run playwright install chromium --with-deps

# --- Pre-download OmniParser V2 weights (optional build-time cache) ---
# Esto evita la descarga en runtime (ahorra ~2-5 min en primer arranque).
# Requiere HF_TOKEN como build-arg. Si no se pasa, simplemente omite el step.
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
    mkdir -p /omniparser/weights && \
    HF_TOKEN=$HF_TOKEN uv run python -c \
    "from huggingface_hub import snapshot_download; \
     snapshot_download(repo_id='microsoft/OmniParser-v2.0', local_dir='/omniparser/weights', local_dir_use_symlinks=False)"; \
    fi

# Script de arranque: lanza Xvfb en background y luego uvicorn
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000

CMD ["/start.sh"]
