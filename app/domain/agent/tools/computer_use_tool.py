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

def _capture_screen_sync() -> str:
    """Captura pantalla real con mss. Síncrona — wrappear con run_in_executor."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # pantalla principal (monitor 0 = all screens)
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        # Reducir resolución para ahorrar tokens en el contexto VL
        img = img.resize(
            (img.width // 2, img.height // 2),
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
            # Retorna el primer screenshot disponible (se puede rotar con un state)
            screen_path = os.path.join(demo_dir, screens[0])
            with open(screen_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    # Fallback: imagen placeholder 800x600 gris con texto "DEMO"
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (800, 600), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)
        draw.text((300, 260), "DEMO MODE - SAP GUI Placeholder", fill=(180, 180, 180))
        draw.text((320, 300), "Computer Use Agent Active", fill=(100, 200, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except ImportError:
        # Ultra-fallback: 1x1 pixel transparente
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


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
            loop = asyncio.get_event_loop()
            b64 = await loop.run_in_executor(None, _capture_screen_sync)
        except Exception as e:
            logger.error(f"[ComputerUse] Error capturando pantalla: {e}")
            b64 = _get_demo_screenshot()  # fallback a demo si falla

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
            import pyautogui
            pyautogui.FAILSAFE = True  # mover mouse a esquina sup-izq cancela
            pyautogui.PAUSE = 0.1     # pausa entre acciones para estabilidad

            if action_type == "click":
                pyautogui.click(action["x"], action["y"])
                return f"Click ejecutado en ({action['x']}, {action['y']})"

            elif action_type == "double_click":
                pyautogui.doubleClick(action["x"], action["y"])
                return f"Double-click en ({action['x']}, {action['y']})"

            elif action_type == "type":
                text = action.get("text", "")
                pyautogui.typewrite(text, interval=0.04)
                return f"Texto escrito: '{text[:50]}{'...' if len(text) > 50 else ''}'"

            elif action_type == "press":
                key = action.get("key", "")
                pyautogui.press(key)
                return f"Tecla presionada: {key}"

            elif action_type == "move":
                pyautogui.moveTo(action["x"], action["y"], duration=0.2)
                return f"Mouse movido a ({action['x']}, {action['y']})"

            elif action_type == "scroll":
                pyautogui.scroll(
                    action.get("amount", 3),
                    x=action.get("x"),
                    y=action.get("y"),
                )
                return f"Scroll {action.get('amount', 3)} en ({action.get('x')}, {action.get('y')})"

            else:
                return f"Tipo de acción desconocido: {action_type}"

        except Exception as e:
            logger.error(f"[ComputerUse] Error ejecutando acción {action_type}: {e}")
            return f"ERROR ejecutando {action_type}: {e}"

    result = await asyncio.to_thread(_execute_sync)
    logger.info(f"[ComputerUse] Acción ejecutada: {result}")
    return result


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
COMPUTER_USE_TOOLS = [take_screenshot, execute_action, task_complete]
