"""
Computer Use Subagent — Sistema 1 / Digital Optimus Local
==========================================================
Implementa el loop Observe-Think-Act con Qwen2.5-VL local.

Rol en la arquitectura Macrohard:
  - Recibe instrucción de alto nivel del Orchestrator (System 2)
  - Ejecuta la instrucción viendo la pantalla en tiempo real
  - Guarda cada step de la trayectoria en el VL Replay Buffer
  - Termina cuando llama a task_complete() o supera max_steps

Flujo por step:
  1. take_screenshot() → imagen base64
  2. Modelo VL (ChatOllama con Qwen2.5-VL) recibe imagen + instrucción
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
    task_complete,
    COMPUTER_USE_TOOLS,
)
from app.persistence.vl_replay_buffer import VLReplayBuffer

# ── System Prompt ─────────────────────────────────────────────────────────────

COMPUTER_USE_SYSTEM_PROMPT = """\
<role>Aura Sistema 1: Computer Use Executor (Digital Optimus Local)</role>

<workflow>Observe -> Act</workflow>

<action_format>
- `take_screenshot`: Call this FIRST to observe.
- `execute_action`: Execute EXACTLY ONE action. JSON format:
   {"type": "click|type|press|move|double_click|scroll", "x": int, "y": int, "text": "str", "key": "str", "amount": int}
- `task_complete`: Call ONLY when the instruction is fully complete or permanently stuck.
</action_format>

<rules>
- ALWAYS call `take_screenshot` before deciding an action.
- Output ONLY ONE action per turn. Never chain tool calls.
- If stuck after 3 attempts at the same step, call `task_complete` with error.
- NEVER explain your reasoning to the user. Use <think> tags internally if needed, but the output must be valid tool calls.
- NO CONVERSATIONAL FILLER. Output only the requested actions.
</rules>
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
                        vl_replay_buffer.append_trajectory_step(
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
    observe_node = _build_observe_node(vision_llm)
    think_act_node = _build_think_act_node(vision_llm, vl_replay_buffer)

    graph = StateGraph(ComputerUseState)
    graph.add_node("observe", observe_node)
    graph.add_node("think_act", think_act_node)

    graph.set_entry_point("observe")
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
