"""
Computer Use Tools — Digital Optimus Local (Hybrid Vision Pipeline)
===================================================================
Herramientas LangChain para el loop Observe-Think-Act del Computer Use Agent.

Modos de operación (resueltos automáticamente por prioridad):

  1. DEMO_MODE (config.computer_use_demo_mode):
     - take_screenshot() → imágenes pre-grabadas / placeholder
     - execute_action() → loguea pero NO ejecuta

  2. PLAYWRIGHT MODE (config.playwright_enabled + browser activo):
     - take_screenshot() → Playwright page.screenshot() (solo viewport, limpio)
     - get_page_context() → Accessibility Tree + URL + título
     - execute_action() → acciones semánticas pw_click, pw_fill, pw_goto
     - Ideal para tareas web (Gmail, Google, SAP Fiori)

  3. NATIVE MODE (fallback — mss + xdotool):
     - take_screenshot() → captura pantalla completa Xvfb con mss
     - execute_action() → xdotool / pyautogui
     - Para SAP GUI desktop, terminales, apps nativas
"""

import asyncio
import base64
import io
import json
import os
import shlex
import subprocess

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.config import settings


# ── Clean screenshot store (for Live Screen Viewer — no coordinate grid) ──────
# Thread-safe dict keyed by thread_id to avoid race conditions between concurrent
# Computer Use sessions. Set by take_screenshot(); read by observe node for SSE display.
from typing import Dict
_clean_b64_map: Dict[str, str] = {}


def get_clean_b64(thread_id: str = "") -> str:
    """Returns the last clean screenshot (no coordinate grid) for the live viewer."""
    return _clean_b64_map.get(thread_id, "")


def _xdotool(*args: str) -> None:
    """Run xdotool safely without shell=True (prevents command injection)."""
    subprocess.run(
        ["xdotool"] + list(args),
        env={**os.environ, "DISPLAY": ":99"},
        check=True,
        timeout=5,
    )


# ── Screenshot capture ────────────────────────────────────────────────────────

# Screenshot no sufre downscaling para exprimir capacidades VL nativas (resolución 1:1)
# Si es necesario reducir contexto, realizar recortes espaciales (cropping).
SCREENSHOT_SCALE = 1  # image sent to model = actual_resolution / SCREENSHOT_SCALE


def _capture_screen_sync() -> str:
    """Captura pantalla real con mss. Síncrona — wrappear con run_in_executor."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # pantalla principal (monitor 0 = all screens)
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        # Mantenemos resolución 100% nativa. El VLM procesa en alta resolución.
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

        # Lineas verticales y etiquetas X — labels en screen space (imagen × SCREENSHOT_SCALE)
        for x in range(0, w, grid_step):
            draw.line([(x, 0), (x, h)], fill=line_color, width=1)
            if x > 0:
                draw.text((x + 2, 2), str(x * SCREENSHOT_SCALE), fill=text_color)

        # Lineas horizontales y etiquetas Y — labels en screen space (imagen × SCREENSHOT_SCALE)
        for y in range(0, h, grid_step):
            draw.line([(0, y), (w, y)], fill=line_color, width=1)
            if y > 0:
                draw.text((2, y + 2), str(y * SCREENSHOT_SCALE), fill=text_color)

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

    Prioridad de captura:
      1. DEMO_MODE → screenshot simulado
      2. PLAYWRIGHT_ENABLED + browser activo → viewport limpio via Playwright
      3. Fallback → pantalla completa Xvfb via mss

    Returns:
        String base64 de la imagen PNG. Incluye prefijo data:image/png;base64,...
        para compatibilidad directa con el modelo VL.
    """
    demo_mode = settings.computer_use_demo_mode

    if demo_mode:
        logger.debug("[ComputerUse] DEMO MODE: retornando screenshot simulado.")
        b64 = _get_demo_screenshot()
    elif settings.playwright_enabled:
        # ── Playwright hybrid: clean viewport-only screenshot ─────────────
        try:
            from app.domain.shared.agent.tools.browser_manager import get_browser_manager
            mgr = get_browser_manager()
            if mgr.is_ready:
                logger.debug("[ComputerUse] Playwright: capturando viewport del browser...")
                b64 = await mgr.screenshot_b64()
            else:
                logger.debug("[ComputerUse] Playwright no ready — fallback a mss.")
                loop = asyncio.get_running_loop()
                b64 = await loop.run_in_executor(None, _capture_screen_sync)
        except Exception as e:
            logger.warning(f"[ComputerUse] Playwright screenshot failed ({e}), fallback a mss.")
            loop = asyncio.get_running_loop()
            b64 = await loop.run_in_executor(None, _capture_screen_sync)
    else:
        logger.debug("[ComputerUse] Capturando pantalla real con mss...")
        try:
            loop = asyncio.get_running_loop()
            b64 = await loop.run_in_executor(None, _capture_screen_sync)
        except Exception as e:
            logger.error(f"[ComputerUse] Error capturando pantalla: {e}")
            b64 = _get_demo_screenshot()  # fallback a demo si falla

    # Save clean screenshot BEFORE grid overlay — keyed by thread_id for thread safety
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    _clean_b64_map[thread_id] = b64

    # Mejora A: Pintar cuadricula de coordenadas para que el VL model
    # lea las coordenadas directamente en la imagen (OmniParser concept)
    b64 = _add_coordinate_grid(b64)

    return f"data:image/png;base64,{b64}"


@tool
async def get_page_context(config: RunnableConfig) -> str:
    """
    Returns the Accessibility Tree and page metadata for the current browser page.

    This provides a structured, semantic view of all interactive elements
    visible on the page (buttons, inputs, links, headings) — far more
    reliable than guessing from pixel coordinates.

    Use this BEFORE deciding where to click. When the accessibility tree
    lists an element (e.g., button "Compose"), prefer pw_click over
    coordinate-based click.

    Only available when playwright_enabled=True and a browser is active.
    Returns an error message if Playwright is not available.

    Returns:
        Formatted string with URL, title, and accessibility tree YAML.
    """
    if not settings.playwright_enabled:
        return "(get_page_context unavailable: playwright_enabled=False. Use take_screenshot + coordinate-based actions.)"

    try:
        from app.domain.shared.agent.tools.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        if not mgr.is_ready:
            return "(Browser not started. Use run_shell_command or pw_goto to open a page first.)"

        info = await mgr.page_info()
        tree = await mgr.accessibility_snapshot()

        return (
            f"Current URL: {info['url']}\n"
            f"Page Title: {info['title']}\n"
            f"Viewport: {info['viewport']}\n\n"
            f"Accessibility Tree:\n{tree}"
        )
    except Exception as e:
        logger.warning(f"[ComputerUse] get_page_context error: {e}")
        return f"(Accessibility tree error: {e})"


@tool
async def execute_action(config: RunnableConfig, action_json: str) -> str:
    """
    Ejecuta una acción en la interfaz gráfica de la pantalla.
    
    Formatos de acción soportados:
      click:        {"type": "click", "x": 1450, "y": 280}
      double_click: {"type": "double_click", "x": 1450, "y": 280}
      type:         {"type": "type", "text": "MB51"}  ← usa clipboard para texto >80 chars
      press:        {"type": "press", "key": "enter"}  ← soporta ctrl+t, ctrl+w, ctrl+l, ctrl+Tab
      move:         {"type": "move", "x": 200, "y": 50}
      scroll:       {"type": "scroll", "x": 800, "y": 400, "amount": 3}  ← positivo=abajo
      new_tab:      {"type": "new_tab"}  ← abre nueva pestaña del browser
      close_tab:    {"type": "close_tab"}  ← cierra pestaña actual
      focus_address_bar: {"type": "focus_address_bar"}  ← Ctrl+L, listo para escribir URL
      navigate:     {"type": "navigate", "url": "https://..."}  ← abre URL en pestaña actual
    
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

    # ── Playwright semantic actions (pw_* prefix) ────────────────────────────
    if action_type.startswith("pw_") and settings.playwright_enabled:
        from app.domain.shared.agent.tools.browser_manager import get_browser_manager
        mgr = get_browser_manager()
        if not mgr.is_ready:
            return f"ERROR: Playwright browser not ready. Call pw_goto first to start the browser."

        try:
            if action_type == "pw_goto":
                url = action.get("url", "")
                return await mgr.goto(url)

            elif action_type == "pw_click":
                role = action.get("role", "")
                name = action.get("name", "")
                if role and name:
                    return await mgr.click_by_role(role, name)
                elif name:
                    return await mgr.click_by_text(name)
                elif "x" in action and "y" in action:
                    return await mgr.click_coordinates(int(action["x"]), int(action["y"]))
                else:
                    return "ERROR: pw_click requires (role+name), (name), or (x+y)"

            elif action_type == "pw_fill":
                role = action.get("role", "textbox")
                name = action.get("name", "")
                value = action.get("value", "")
                return await mgr.fill_field(role, name, value)

            elif action_type == "pw_type":
                text = action.get("text", "")
                return await mgr.type_text(text)

            elif action_type == "pw_press":
                key = action.get("key", "")
                return await mgr.press_key(key)

            elif action_type == "pw_scroll":
                direction = action.get("direction", "down")
                amount = int(action.get("amount", 3))
                return await mgr.scroll(direction, amount)

            elif action_type == "pw_wait":
                return await mgr.wait_for_load(timeout=int(action.get("timeout", 5000)))

            else:
                return f"ERROR: Unknown Playwright action type: {action_type}"

        except Exception as e:
            logger.error(f"[ComputerUse] Playwright action {action_type} failed: {e}")
            return f"ERROR in {action_type}: {e}"

    # ── Native mode — ejecutar con xdotool / pyautogui ───────────────────────
    def _execute_sync():
        try:
            try:
                import pyautogui
            except (SystemExit, ImportError) as _pyautogui_err:
                return f"ERROR: pyautogui no disponible ({_pyautogui_err}). Instala python3-tk en el contenedor."
            pyautogui.FAILSAFE = True  # mover mouse a esquina sup-izq cancela
            pyautogui.PAUSE = 0.1     # pausa entre acciones para estabilidad

            # The model outputs coordinates in screen space (1920×1080), read directly
            # from the grid labels on the screenshot. Use them as-is — no scaling needed.
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))

            if action_type == "click":
                try:
                    _xdotool("mousemove", str(x), str(y), "click", "1")
                    return f"Click ejecutado en ({x}, {y}) [imagen: {action['x']},{action['y']}]"
                except Exception as e:
                    logger.debug(f"[ComputerUse] xdotool click failed ({e}), fallback pyautogui")
                    pyautogui.click(x, y)
                    return f"Click ejecutado en ({x}, {y}) [fallback pyautogui]"

            elif action_type == "double_click":
                try:
                    _xdotool("mousemove", str(x), str(y), "click", "--repeat", "2", "--delay", "100", "1")
                    return f"Double-click en ({x}, {y}) [imagen: {action['x']},{action['y']}]"
                except Exception as e:
                    logger.debug(f"[ComputerUse] xdotool double_click failed ({e}), fallback pyautogui")
                    pyautogui.doubleClick(x, y)
                    return f"Double-click en ({x}, {y}) [fallback pyautogui]"




            elif action_type == "type":
                text = action.get("text", "")
                # For long text (>80 chars): use clipboard paste — faster and handles special chars
                if len(text) > 80:
                    try:
                        subprocess.run(
                            ["xclip", "-selection", "clipboard"],
                            input=text.encode(),
                            check=True, timeout=5,
                        )
                        _xdotool("key", "ctrl+v")
                        return f"Texto pegado via clipboard: '{text[:50]}...'"
                    except Exception as e:
                        logger.debug(f"[ComputerUse] clipboard paste failed ({e}), fallback to xdotool type")
                        pass  # fall through to xdotool type
                try:
                    _xdotool("type", "--clearmodifiers", "--delay", "30", text)
                except Exception as e:
                    logger.debug(f"[ComputerUse] xdotool type failed ({e}), fallback pyautogui")
                    pyautogui.typewrite(text, interval=0.04)
                return f"Texto escrito: '{text[:50]}{'...' if len(text) > 50 else ''}'"

            elif action_type == "press":
                key = action.get("key", "")
                xdotool_key_map = {
                    "Return": "Return", "enter": "Return", "Tab": "Tab", "tab": "Tab",
                    "Escape": "Escape", "esc": "Escape", "ctrl+a": "ctrl+a",
                    "ctrl+c": "ctrl+c", "ctrl+v": "ctrl+v", "ctrl+z": "ctrl+z",
                    "ctrl+f": "ctrl+f", "ctrl+l": "ctrl+l", "ctrl+t": "ctrl+t",
                    "ctrl+w": "ctrl+w", "ctrl+r": "ctrl+r", "ctrl+Tab": "ctrl+Tab",
                    "ctrl+shift+Tab": "ctrl+shift+Tab", "alt+Left": "alt+Left",
                    "alt+Right": "alt+Right", "BackSpace": "BackSpace", "Delete": "Delete",
                    "F5": "F5", "F11": "F11", "space": "space", "Home": "Home", "End": "End",
                    "Page_Up": "Page_Up", "Page_Down": "Page_Down",
                }
                xdotool_key = xdotool_key_map.get(key, key)
                try:
                    _xdotool("key", xdotool_key)
                except Exception as e:
                    logger.debug(f"[ComputerUse] xdotool key failed ({e}), fallback pyautogui")
                    pyautogui.press(key)
                return f"Tecla presionada: {key}"

            elif action_type == "move":
                try:
                    _xdotool("mousemove", str(x), str(y))
                except Exception as e:
                    logger.debug(f"[ComputerUse] xdotool move failed ({e}), fallback pyautogui")
                    pyautogui.moveTo(x, y, duration=0.2)
                return f"Mouse movido a ({x}, {y}) [imagen: {action['x']},{action['y']}]"

            elif action_type == "scroll":
                amount = action.get("amount", 3)
                # xdotool scroll: button 4 = up, button 5 = down
                btn = "5" if amount > 0 else "4"
                clicks = abs(amount)
                try:
                    _xdotool("mousemove", str(x), str(y), "click", "--repeat", str(clicks), btn)
                except Exception as e:
                    logger.debug(f"[ComputerUse] xdotool scroll failed ({e}), fallback pyautogui")
                    pyautogui.scroll(-amount, x=x, y=y)  # pyautogui: negative=down
                return f"Scroll {'abajo' if amount > 0 else 'arriba'} ×{abs(amount)} en ({x},{y})"

            elif action_type == "new_tab":
                _xdotool("key", "ctrl+t")
                return "Nueva pestaña abierta (Ctrl+T)"

            elif action_type == "close_tab":
                _xdotool("key", "ctrl+w")
                return "Pestaña cerrada (Ctrl+W)"

            elif action_type == "focus_address_bar":
                _xdotool("key", "ctrl+l")
                return "Barra de dirección enfocada (Ctrl+L) — ya puedes escribir la URL"

            elif action_type == "navigate":
                url = action.get("url", "")
                # Focus address bar, clear it, type URL and navigate
                _xdotool("key", "ctrl+l")
                import time as _time; _time.sleep(0.3)
                _xdotool("type", "--clearmodifiers", url)
                _xdotool("key", "Return")
                return f"Navegando a: {url}"

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
        _profile = "--user-data-dir=/tmp/chromium-profile --profile-directory=Default"
        if "--no-sandbox" not in _cmd:
            _binary, _, _rest = _cmd.partition(" ")
            _cmd = f"{_binary} {_required} {_profile} {_rest}".strip()
        elif "--user-data-dir" not in _cmd:
            _binary, _, _rest = _cmd.partition(" ")
            _cmd = f"{_binary} {_profile} {_rest}".strip()

        # Elimina el candado residual de la sesión anterior para evitar crasheos de perfil bloqueado
        try:
            subprocess.run(
                ["rm", "-f", "/tmp/chromium-profile/SingletonLock", "/tmp/chromium-profile/SingletonCookie"],
                check=False, timeout=2,
            )
        except Exception as e:
            logger.debug(f"[ComputerUse] SingletonLock cleanup failed: {e}")
        logger.debug(f"[ComputerUse] Chromium flags auto-injected & unlocked: {_cmd}")
        command = _cmd

    cmd_str = command.strip()
    background = cmd_str.endswith(" &")
    if background:
        cmd_str = cmd_str[:-2].strip()

    # Block dangerous patterns (defense-in-depth)
    _blocked = ["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:", "curl.*|.*bash", "wget.*|.*bash", "> /dev/sd"]
    import re
    for pat in _blocked:
        if re.search(pat, cmd_str, re.IGNORECASE):
            logger.warning(f"[ComputerUse] BLOCKED dangerous command: {cmd_str[:100]}")
            return "ERROR: Command blocked for security reasons."

    cmd_parts = shlex.split(cmd_str)
    if not cmd_parts:
        return "ERROR: Empty command."

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
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
        logger.info(f"[ComputerUse] Shell executed: {cmd_str!r} → {result[:100]}")
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
COMPUTER_USE_TOOLS = [take_screenshot, get_page_context, execute_action, run_shell_command, task_complete]
