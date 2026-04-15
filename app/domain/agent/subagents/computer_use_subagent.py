"""
Computer Use Subagent — Sistema 1 / Digital Optimus Local
==========================================================
Implementa el loop Observe-Think-Act con Qwen3.5 nativo multimodal.

Rol en la arquitectura Macrohard:
  - Recibe instrucción de alto nivel del Orchestrator (System 2)
  - Ejecuta la instrucción viendo la pantalla en tiempo real
  - Guarda cada step de la trayectoria en el VL Replay Buffer
  - Termina cuando llama a task_complete() o supera max_steps

Flujo por step:
  1. take_screenshot() → imagen base64
  2. Modelo Qwen3.5 (nativamente VLM) recibe imagen + instrucción
  3. Modelo devuelve JSON de acción: {"type": "click", "x": N, "y": N}
  4. execute_action(action_json) → ejecuta la acción
  5. VL Replay Buffer guarda (screenshot, instrucción, acción)
  6. Loop hasta task_complete() o max_steps
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict, List, Annotated
import operator
from loguru import logger

from app.core.config import settings
from app.domain.agent.tools.computer_use_tool import (
    take_screenshot,
    execute_action,
    run_shell_command,
    task_complete,
    COMPUTER_USE_TOOLS,
)
from app.persistence.vl_replay_buffer import VLReplayBuffer

# ── System Prompt ─────────────────────────────────────────────────────────────

COMPUTER_USE_SYSTEM_PROMPT = """\
<role>Aura Sistema 1 — Computer Use Executor (Digital Optimus)</role>

<mission>
You are an autonomous computer operator. You receive a high-level instruction,
observe the screen in real time, decide the best single action to take,
and execute it — repeating until the task is complete.
You interact with the actual computer screen. Every action has real effects.
Think carefully before acting. One wrong click can navigate away from the current state.
</mission>

<environment>
- OS: Ubuntu Linux (headless, Xvfb virtual display at :99, actual resolution 1920×1080)
- Browser: Chromium (launched via run_shell_command — flags injected automatically)
- Screenshots delivered at HALF resolution: 960×540 pixels
- Coordinates: output x,y in IMAGE space (0–960 width, 0–540 height) based on what you SEE.
  The system scales them ×2 to actual 1920×1080 before executing — do NOT pre-scale yourself.
</environment>

<core_loop>
Every turn follows exactly this sequence:
  OBSERVE → THINK → ACT (one action only)

Step 1 — OBSERVE: Call take_screenshot to see the current screen state.
Step 2 — THINK (internally in <thinking> tags):
  a. Where am I? What application/page is currently visible?
  b. What is the next concrete step toward completing the instruction?
  c. What UI element do I need to interact with?
  d. What are its exact coordinates in the 960×540 image?
  e. Have I already tried this exact action and failed? If so, try an alternative.
Step 3 — ACT: Output EXACTLY ONE tool call. No text before or after.
</core_loop>

<tools>
take_screenshot()
  → Returns the current screen as a base64 PNG image.
  → Call this FIRST every turn before any other action.

execute_action(action_json: str)
  → Executes a single GUI interaction. JSON format:
  {"type": "click",        "x": int, "y": int}
  {"type": "double_click", "x": int, "y": int}
  {"type": "type",         "text": "string to type"}
  {"type": "press",        "key": "Return|Tab|Escape|ctrl+a|ctrl+c|..."}
  {"type": "scroll",       "x": int, "y": int, "amount": int}  ← positive=down
  {"type": "move",         "x": int, "y": int}
  → Execute EXACTLY ONE action per turn.

run_shell_command(command: str)
  → Runs a shell command. Use to launch applications.
  → Chromium flags (--no-sandbox, --disable-gpu, etc.) are injected automatically.
  → Examples: "chromium https://youtube.com &", "xdg-open file.pdf &"

task_complete(result: str)
  → Call when the task is FULLY done OR after 3 consecutive failed attempts at the same step.
  → Provide a clear result summary: what was accomplished, what (if anything) failed.
</tools>

<browser_workflow>
To navigate to a website:
1. run_shell_command: "chromium https://example.com &"
2. take_screenshot — wait for browser to appear
3. take_screenshot again if browser is still loading (WAIT 2–3 steps before clicking)
4. Once loaded: click address bar (y≈25, x≈380 in 960×540), type URL, press Enter

Common element positions in 960×540 image space (approximate — always verify with screenshot):
  - Chromium address bar: y≈25, x≈350–450
  - YouTube search bar:   y≈40, x≈480
  - SAP Fiori search:     y≈55, x≈480
</browser_workflow>

<error_recovery>
If an action has no visible effect after 1 attempt:
  1. Take another screenshot to verify current state
  2. Try a slightly different approach (different coordinates, different action type)
  3. If still failing after 3 attempts at the SAME step: call task_complete with failure description
     (Why: infinite retries of a broken action loop is worse than stopping and reporting)
</error_recovery>

<rules>
- ALWAYS call take_screenshot before any execute_action — never act blind.
  (Why: the screen state changes constantly; acting without seeing causes errors.)
- Output ONLY ONE tool call per turn — never chain multiple calls.
- Use <thinking> tags for internal reasoning — never output reasoning as plain text.
- NO conversational filler — your output is tool calls only.
- After opening a browser or application, take 2–3 screenshots before clicking (let it load).
- Coordinates must be in 960×540 IMAGE space — the system handles scaling.
</rules>

<examples>
<example>
<instruction>Open YouTube and describe what is on the homepage.</instruction>
<step_1>take_screenshot() → see: empty desktop</step_1>
<thinking>Browser not open yet. Launch Chromium with YouTube URL.</thinking>
<step_2>run_shell_command("chromium https://youtube.com &")</step_2>
<step_3>take_screenshot() → see: browser loading</step_3>
<thinking>Browser still loading. Wait one more step.</thinking>
<step_4>take_screenshot() → see: YouTube homepage fully loaded</step_4>
<thinking>Page loaded. I can now describe the homepage content visible on screen.</thinking>
<step_5>task_complete("YouTube homepage loaded. Visible content: trending videos section with thumbnails, search bar at top, YouTube logo top-left, sidebar with Home/Shorts/Subscriptions navigation.")</step_5>
</example>

<example>
<instruction>Navigate to SAP Fiori and open transaction MB51.</instruction>
<step_1>take_screenshot() → see: empty desktop</step_1>
<thinking>Need to open SAP Fiori in browser. Launch Chromium.</thinking>
<step_2>run_shell_command("chromium https://sap-fiori.local &")</step_2>
<step_3>take_screenshot() → see: SAP Fiori login screen</step_3>
<thinking>Login screen visible. Need to enter credentials. Click username field (y≈200, x≈480).</thinking>
<step_4>execute_action({"type": "click", "x": 480, "y": 200})</step_4>
<step_5>execute_action({"type": "type", "text": "AURA_USER"})</step_5>
</example>
</examples>
"""


# ── LangGraph State ───────────────────────────────────────────────────────────

class ComputerUseState(TypedDict):
    instruction: str
    messages: Annotated[List, operator.add]
    steps_taken: int
    last_screenshot_b64: Optional[str]
    trajectory: Annotated[List, operator.add]  # lista de steps para VL Buffer
    is_complete: bool
    result_summary: str


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def _build_init_node():
    """
    Entry node: extracts the instruction from the messages list injected by
    deepagents and initialises all required ComputerUseState fields.

    deepagents passes subagent inputs as {"messages": [HumanMessage(...)]}.
    The raw StateGraph expects ComputerUseState fields (instruction, steps_taken,
    etc.) — this node bridges that impedance mismatch.
    """

    async def init(state: ComputerUseState) -> dict:
        instruction = state.get("instruction", "")

        if not instruction:
            for msg in reversed(state.get("messages", [])):
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    instruction = content.strip()
                    break
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "").strip()
                            if text:
                                instruction = text
                                break
                    if instruction:
                        break

        return {
            "instruction": instruction,
            "steps_taken": state.get("steps_taken", 0),
            "last_screenshot_b64": state.get("last_screenshot_b64"),
            "trajectory": state.get("trajectory", []),
            "is_complete": state.get("is_complete", False),
            "result_summary": state.get("result_summary", ""),
        }

    return init


def _build_observe_node(llm: BaseChatModel):
    """Nodo: captura screenshot y lo inyecta en el state."""

    async def observe(state: ComputerUseState, config: RunnableConfig) -> dict:
        logger.info(f"[ComputerUse] Step {state['steps_taken'] + 1} — Observando pantalla...")

        # Captura pantalla
        screenshot_result = await take_screenshot.ainvoke({}, config=config)

        # El resultado es "data:image/png;base64,<data>"
        b64_data = screenshot_result.split(",", 1)[-1] if "," in screenshot_result else screenshot_result

        return {"last_screenshot_b64": b64_data}

    return observe


def _build_think_act_node(llm: BaseChatModel, vl_replay_buffer: Optional[VLReplayBuffer]):
    """Nodo: razona sobre el screenshot y ejecuta UNA acción."""

    async def think_act(state: ComputerUseState, config: RunnableConfig) -> dict:
        steps = state["steps_taken"]
        instruction = state["instruction"]
        screenshot_b64 = state["last_screenshot_b64"]
        max_steps = settings.computer_use_max_steps

        if steps >= max_steps:
            logger.warning(f"[ComputerUse] Límite de {max_steps} steps alcanzado. Finalizando.")
            return {
                "is_complete": True,
                "result_summary": f"Límite de {max_steps} pasos alcanzado. Tarea parcialmente completada.",
                "steps_taken": steps + 1,
            }

        # Construir mensaje multimodal para el VL model
        user_content = [
            {
                "type": "text",
                "text": f"Instruction: {instruction}\nStep {steps + 1}: Decide next action.",
            }
        ]

        if screenshot_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            })

        messages_for_llm = [
            SystemMessage(content=COMPUTER_USE_SYSTEM_PROMPT),
        ] + state["messages"] + [
            HumanMessage(content=user_content),
        ]

        # Llamar al VL model con herramientas
        llm_with_tools = llm.bind_tools(COMPUTER_USE_TOOLS)

        try:
            response = await llm_with_tools.ainvoke(messages_for_llm, config=config)
        except Exception as e:
            logger.error(f"[ComputerUse] Error invocando VL model: {e}")
            return {
                "is_complete": True,
                "result_summary": f"Error en el modelo VL: {e}",
                "steps_taken": steps + 1,
                "messages": [HumanMessage(content=user_content)],
            }

        # Parsear tool calls del response
        tool_calls = getattr(response, "tool_calls", [])
        
        # --- Fallback robusto para Qwen3.5-VL ---
        # A veces Qwen emite el XML correcto pero LangChain falla en parsearlo
        if not tool_calls and isinstance(response.content, str) and "<tool_call>" in response.content:
            try:
                import xml.etree.ElementTree as ET
                xml_str = response.content[response.content.find("<tool_call>"):response.content.rfind("</tool_call>") + 12]
                root = ET.fromstring(xml_str)
                tool_name = root.find("name").text if root.find("name") is not None else ""
                args_str = root.find("arguments").text if root.find("arguments") is not None else "{}"
                tool_calls = [{"name": tool_name.strip(), "args": json.loads(args_str)}]
                logger.info(f"[ComputerUse] Fallback XML parser exitoso: extraídos {len(tool_calls)} tool calls.")
            except Exception as xml_e:
                logger.warning(f"[ComputerUse] Error en fallback XML parser: {xml_e}")

        new_trajectory = []
        is_complete = False
        result_summary = ""

        if not tool_calls:
            # El modelo respondió texto en vez de tool call — ignorar y continuar
            logger.warning(f"[ComputerUse] El modelo no retornó tool_call. Respuesta: {response.content[:100]}")
            return {
                "steps_taken": steps + 1,
                "messages": [HumanMessage(content=user_content), response],
            }

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            if tool_name == "take_screenshot":
                # El modelo quiere ver la pantalla de nuevo (ya fue capturada en observe)
                pass

            elif tool_name == "execute_action":
                action_json = tool_args.get("action_json", "{}")
                action_result = await execute_action.ainvoke(
                    {"action_json": action_json}, config=config
                )
                logger.info(f"[ComputerUse] Acción ejecutada: {action_result}")

                # Guardar step en el VL Replay Buffer
                if vl_replay_buffer and screenshot_b64:
                    try:
                        await vl_replay_buffer.append_trajectory_step(
                            instruction=instruction,
                            screenshot_b64=screenshot_b64,
                            action_json=action_json,
                            tool_name="computer_use",
                        )
                        new_trajectory.append({
                            "instruction": instruction,
                            "screenshot_b64": screenshot_b64,
                            "action_json": action_json,
                        })
                    except Exception as e:
                        logger.warning(f"[ComputerUse] Error guardando en VL buffer: {e}")

            elif tool_name == "run_shell_command":
                command = tool_args.get("command", "")
                action_result = await run_shell_command.ainvoke(
                    {"command": command}, config=config
                )
                logger.info(f"[ComputerUse] Shell command disparado: {action_result}")
                
                # Para comandos de shell, guardamos el action_json simulado para el replay buffer
                if vl_replay_buffer and screenshot_b64:
                    try:
                        await vl_replay_buffer.append_trajectory_step(
                            instruction=instruction,
                            screenshot_b64=screenshot_b64,
                            action_json=json.dumps({"type": "shell", "command": command}),
                            tool_name="run_shell_command",
                        )
                        new_trajectory.append({
                            "instruction": instruction,
                            "screenshot_b64": screenshot_b64,
                            "action_json": json.dumps({"type": "shell", "command": command}),
                        })
                    except Exception as e:
                        logger.warning(f"[ComputerUse] Error guardando en VL buffer: {e}")

            elif tool_name == "task_complete":
                summary = tool_args.get("summary", "Tarea completada.")
                task_result = await task_complete.ainvoke(
                    {"summary": summary}, config=config
                )
                is_complete = True
                result_summary = summary
                logger.info(f"[ComputerUse] ✅ Tarea completada: {summary}")

        return {
            "steps_taken": steps + 1,
            "is_complete": is_complete,
            "result_summary": result_summary,
            "trajectory": new_trajectory,
            "messages": [HumanMessage(content=user_content), response],
        }

    return think_act


def _should_continue(state: ComputerUseState) -> str:
    """Router: continúa el loop o termina."""
    if state.get("is_complete", False):
        return END
    if state["steps_taken"] >= settings.computer_use_max_steps:
        return END
    return "observe"


# ── Factory ───────────────────────────────────────────────────────────────────

def create_computer_use_agent(
    vision_llm: BaseChatModel,
    vl_replay_buffer: Optional[VLReplayBuffer] = None,
) -> CompiledStateGraph:
    """
    Construye el grafo LangGraph del Computer Use Agent (Sistema 1).

    Args:
        vision_llm: Instancia de ChatOpenAI configurada con el modelo VL
                    (Qwen2.5-VL local cargado en vLLM).
        vl_replay_buffer: Buffer donde se guardan los steps para re-training.
                           Si es None, no se guardan los steps.

    Returns:
        CompiledStateGraph listo para ser invocado con:
        {
          "instruction": "Open MB51 in SAP, enter CRUDE-100, update stock...",
          "messages": [],
          "steps_taken": 0,
          "last_screenshot_b64": None,
          "trajectory": [],
          "is_complete": False,
          "result_summary": "",
        }
    """
    init_node = _build_init_node()
    observe_node = _build_observe_node(vision_llm)
    think_act_node = _build_think_act_node(vision_llm, vl_replay_buffer)

    graph = StateGraph(ComputerUseState)
    graph.add_node("init", init_node)
    graph.add_node("observe", observe_node)
    graph.add_node("think_act", think_act_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "observe")
    graph.add_edge("observe", "think_act")
    graph.add_conditional_edges("think_act", _should_continue, {END: END, "observe": "observe"})

    return graph.compile()


async def run_computer_use_task(
    instruction: str,
    vision_llm: BaseChatModel,
    vl_replay_buffer: Optional[VLReplayBuffer] = None,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Helper de alto nivel para ejecutar una tarea de computer use completa.

    Args:
        instruction: Instrucción de alto nivel del Orchestrator.
        vision_llm: Modelo VL inicializado.
        vl_replay_buffer: Buffer para guardar el dataset de training.
        config: RunnableConfig de LangGraph (thread_id, user_id, etc.)

    Returns:
        Resumen del resultado de la tarea.
    """
    agent = create_computer_use_agent(vision_llm, vl_replay_buffer)

    initial_state = {
        "instruction": instruction,
        "messages": [],
        "steps_taken": 0,
        "last_screenshot_b64": None,
        "trajectory": [],
        "is_complete": False,
        "result_summary": "",
    }

    try:
        final_state = await agent.ainvoke(initial_state, config=config)
        result = final_state.get("result_summary", "Tarea completada sin resumen.")
        steps = final_state.get("steps_taken", 0)
        trajectory_count = len(final_state.get("trajectory", []))
        logger.info(
            f"[ComputerUse] Tarea finalizada en {steps} steps. "
            f"{trajectory_count} steps guardados en VL buffer."
        )
        return result
    except Exception as e:
        logger.error(f"[ComputerUse] Error ejecutando tarea: {e}")
        return f"Error en computer use agent: {e}"
