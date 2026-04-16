"""
Computer Use Tools — Digital Optimus Local
==========================================
Herramientas LangChain para el loop Observe-Think-Act del Computer Use Agent.

DEMO_MODE=True (config.computer_use_demo_mode):
  - take_screenshot(): carga imágenes pre-grabadas de /static/demo/screens/
    o genera un screenshot HTML simulado via Playwright headless.
  - execute_action(): loguea la acción pero NO la ejecuta realmente.
  - Funciona sin SAP GUI ni pantalla real — ideal para desarrollo y demos.

DEMO_MODE=False (producción):
  - take_screenshot(): captura pantalla real con mss (async via run_in_executor).
  - execute_action(): ejecuta con pyautogui (async via asyncio.to_thread).
  - Requiere que SAP GUI esté abierto y visible en la pantalla del edge node.
"""

import asyncio
import base64
import io
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.config import settings


# ── Screenshot capture ────────────────────────────────────────────────────────

# Screenshot is downscaled by this factor to save VL context tokens.
# execute_action MUST multiply coordinates by the same factor before calling pyautogui
# so that image-space coords map back to actual 1920×1080 screen-space coords.
SCREENSHOT_SCALE = 2  # image sent to model = actual_resolution / SCREENSHOT_SCALE


def _capture_screen_sync() -> str:
    """Captura pantalla real con mss. Síncrona — wrappear con run_in_executor."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # pantalla principal (monitor 0 = all screens)
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        # Reducir resolución para ahorrar tokens en el contexto VL.
        # IMPORTANT: execute_action re-escalará las coordenadas × SCREENSHOT_SCALE
        # para que los coords del modelo (imagen reducida) coincidan con la pantalla real.
        img = img.resize(
            (img.width // SCREENSHOT_SCALE, img.height // SCREENSHOT_SCALE),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_demo_screenshot() -> str:
    """
    Retorna un screenshot simulado para el modo DEMO.
    Busca en /static/demo/screens/ un PNG apropiado.
    Si no encuentra ninguno, genera una imagen placeholder.
    """
    demo_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "static", "demo", "screens"
    )
    demo_dir = os.path.abspath(demo_dir)

    if os.path.isdir(demo_dir):
        screens = sorted(f for f in os.listdir(demo_dir) if f.endswith(".png"))
        if screens:
            screen_path = os.path.join(demo_dir, screens[0])
            with open(screen_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    # Fallback: imagen placeholder 800x600 gris con texto "DEMO"
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (800, 600), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)
        draw.text((300, 260), "DEMO MODE - SAP GUI Placeholder", fill=(180, 180, 180))
        draw.text((320, 300), "Computer Use Agent Active", fill=(100, 200, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except ImportError:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _add_coordinate_grid(b64_png: str) -> str:
    """
    Mejora A (Digital Optimus / OmniParser): Pinta una cuadricula de coordenadas
    semi-transparente sobre el screenshot. El VL model puede leer los numeros
    directamente en la imagen en lugar de adivinar las coordenadas.

    La cuadricula dibuja marcas cada 100px con el numero de coordenada.
    Para una imagen 960x540, las marcas son: 0, 100, 200, ..., 900 (x) y 0, 100, ..., 500 (y).

    Si PIL no esta disponible, retorna el screenshot sin modificar.
    """
    try:
        from PIL import Image, ImageDraw

        img_bytes = base64.b64decode(b64_png)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        w, h = img.size
        grid_step = 100
        line_color = (255, 255, 0, 60)   # amarillo semi-transparente
        text_color = (255, 255, 0, 200)  # amarillo mas opaco para numeros
        dot_color  = (255, 100, 100, 180) # punto rojo en intersecciones

        # Lineas verticales y etiquetas X
        for x in range(0, w, grid_step):
            draw.line([(x, 0), (x, h)], fill=line_color, width=1)
            if x > 0:
                draw.text((x + 2, 2), str(x), fill=text_color)

        # Lineas horizontales y etiquetas Y
        for y in range(0, h, grid_step):
            draw.line([(0, y), (w, y)], fill=line_color, width=1)
            if y > 0:
                draw.text((2, y + 2), str(y), fill=text_color)

        # Puntos en intersecciones para referencia visual
        for x in range(0, w, grid_step):
            for y in range(0, h, grid_step):
                draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=dot_color)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        logger.debug(f"[ComputerUse] _add_coordinate_grid skipped: {e}")
        return b64_png


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
async def take_screenshot(config: RunnableConfig) -> str:
    """
    Captura la pantalla actual y la devuelve como imagen base64 PNG.
    
    En DEMO_MODE: retorna un screenshot simulado de una interfaz SAP.
    En modo producción: captura la pantalla real del edge node con mss.
    
    Returns:
        String base64 de la imagen PNG. Incluye prefijo data:image/png;base64,...
        para compatibilidad directa con el modelo VL.
    """
    demo_mode = settings.computer_use_demo_mode

    if demo_mode:
        logger.debug("[ComputerUse] DEMO MODE: retornando screenshot simulado.")
        b64 = _get_demo_screenshot()
    else:
        logger.debug("[ComputerUse] Capturando pantalla real con mss...")
        try:
            loop = asyncio.get_running_loop()
            b64 = await loop.run_in_executor(None, _capture_screen_sync)
        except Exception as e:
            logger.error(f"[ComputerUse] Error capturando pantalla: {e}")
            b64 = _get_demo_screenshot()  # fallback a demo si falla

    # Mejora A: Pintar cuadricula de coordenadas para que el VL model
    # lea las coordenadas directamente en la imagen (OmniParser concept)
    b64 = _add_coordinate_grid(b64)

    return f"data:image/png;base64,{b64}"


@tool
async def execute_action(config: RunnableConfig, action_json: str) -> str:
    """
    Ejecuta una acción en la interfaz gráfica de la pantalla.
    
    Formatos de acción soportados:
      click:      {"type": "click", "x": 1450, "y": 280}
      type:       {"type": "type", "text": "MB51"}
      press:      {"type": "press", "key": "enter"}
      move:       {"type": "move", "x": 200, "y": 50}
      scroll:     {"type": "scroll", "x": 800, "y": 400, "amount": 3}
      double_click: {"type": "double_click", "x": 1450, "y": 280}
    
    En DEMO_MODE: loguea la acción pero NO la ejecuta realmente.
    En producción: usa pyautogui de forma async (asyncio.to_thread).
    
    Returns:
        Confirmación de la acción ejecutada o descripción del error.
    """
    try:
        action = json.loads(action_json)
    except json.JSONDecodeError as e:
        return f"ERROR: action_json inválido — {e}. Asegúrate de enviar JSON válido."

    action_type = action.get("type", "unknown")
    demo_mode = settings.computer_use_demo_mode

    if demo_mode:
        logger.info(f"[ComputerUse] DEMO ACTION [{action_type}]: {action_json}")
        return f"[DEMO] Acción {action_type} registrada: {action_json}"

    # Producción — ejecutar con pyautogui
    def _execute_sync():
        try:
            try:
                import pyautogui
            except (SystemExit, ImportError) as _pyautogui_err:
                return f"ERROR: pyautogui no disponible ({_pyautogui_err}). Instala python3-tk en el contenedor."
            pyautogui.FAILSAFE = True  # mover mouse a esquina sup-izq cancela
            pyautogui.PAUSE = 0.1     # pausa entre acciones para estabilidad

            # The model outputs coordinates in the downscaled image space (960×540).
            # Multiply by SCREENSHOT_SCALE to map back to actual screen space (1920×1080).
            sx = SCREENSHOT_SCALE
            x = action.get("x", 0) * sx
            y = action.get("y", 0) * sx

            if action_type == "click":
                pyautogui.click(x, y)
                return f"Click ejecutado en ({x}, {y}) [imagen: {action['x']},{action['y']}]"

            elif action_type == "double_click":
                pyautogui.doubleClick(x, y)
                return f"Double-click en ({x}, {y}) [imagen: {action['x']},{action['y']}]"




            elif action_type == "type":
                text = action.get("text", "")
                import subprocess
                safe_text = text.replace("'", "'\''")
                try:
                    subprocess.run(
                        f"DISPLAY=:99 xdotool type --clearmodifiers --delay 50 '{safe_text}'",
                        shell=True, check=True, timeout=15
                    )
                except Exception as e:
                    pyautogui.typewrite(text, interval=0.04)
                return f"Texto escrito: '{text[:50]}{'...' if len(text) > 50 else ''}'"

            elif action_type == "press":
                key = action.get("key", "")
                pyautogui.press(key)
                return f"Tecla presionada: {key}"

            elif action_type == "move":
                pyautogui.moveTo(x, y, duration=0.2)
                return f"Mouse movido a ({x}, {y}) [imagen: {action['x']},{action['y']}]"

            elif action_type == "scroll":
                pyautogui.scroll(
                    action.get("amount", 3),
                    x=x,
                    y=y,
                )
                return f"Scroll {action.get('amount', 3)} en ({x}, {y})"

            else:
                return f"Tipo de acción desconocido: {action_type}"

        except Exception as e:
            logger.error(f"[ComputerUse] Error ejecutando acción {action_type}: {e}")
            return f"ERROR ejecutando {action_type}: {e}"

    result = await asyncio.to_thread(_execute_sync)
    logger.info(f"[ComputerUse] Acción ejecutada: {result}")
    return result


@tool
async def run_shell_command(config: RunnableConfig, command: str) -> str:
    """
    Executes a shell command on the server. Use this to launch applications
    that cannot be opened via pyautogui alone (e.g., opening a browser).

    Examples:
      - Open Chromium on YouTube: command="chromium --no-sandbox --disable-dev-shm-usage https://youtube.com &"
      - Open a terminal: command="xterm &"
      - Take a specific window to focus: command="wmctrl -a Chromium"

    IMPORTANT: Add '&' at the end to run non-blocking (background) processes.
    The command runs on the server with DISPLAY=:99 already set.

    Args:
        command: Shell command string to execute.

    Returns:
        stdout/stderr output or confirmation of execution.
    """
    demo_mode = settings.computer_use_demo_mode

    if demo_mode:
        logger.info(f"[ComputerUse] DEMO SHELL: {command}")
        return f"[DEMO] Shell command logged: {command}"

    # Auto-inject mandatory Chromium flags when the command starts with chromium/chromium-browser.
    # Without --no-sandbox the process crashes immediately inside Docker (no user namespace).
    _cmd = command.strip()
    if _cmd.startswith("chromium") or _cmd.startswith("chromium-browser"):
        _required = "--no-sandbox --disable-dev-shm-usage"
        if "--no-sandbox" not in _cmd:
            # Insert flags right after the binary name
            _binary, _, _rest = _cmd.partition(" ")
            _cmd = f"{_binary} {_required} {_rest}".strip()
            command = _cmd
            logger.debug(f"[ComputerUse] Chromium flags auto-injected: {command}")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "DISPLAY": ":99"},
        )
        # For background commands (&), don't wait forever
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            output = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()
            result = output or err or "Command executed (no output)."
        except asyncio.TimeoutError:
            result = f"Command launched in background (pid={proc.pid}). Use take_screenshot to verify."
        logger.info(f"[ComputerUse] Shell executed: {command!r} → {result[:100]}")
        return result
    except Exception as e:
        logger.error(f"[ComputerUse] Shell error: {e}")
        return f"ERROR running command: {e}"


@tool
async def task_complete(config: RunnableConfig, summary: str) -> str:
    """
    Señala que la tarea asignada fue completada exitosamente.
    
    El agente DEBE llamar a esta herramienta cuando haya completado
    todas las acciones que le indicó el Orchestrator.
    
    Args:
        summary: Descripción breve de lo que se completó.
                 Ej: "SAP actualizado (stock=12800). Email enviado a ops.manager@company.com"
    
    Returns:
        Confirmación de que la tarea fue marcada como completada.
    """
    logger.info(f"[ComputerUse] ✅ TAREA COMPLETADA: {summary}")
    return f"TASK_COMPLETE: {summary}"


# Export de las herramientas para el subagente
COMPUTER_USE_TOOLS = [take_screenshot, execute_action, run_shell_command, task_complete]
