"""
Generalist Orchestrator Factory.

Routes queries to the appropriate specialist via a flat 2-level hierarchy:

  Orchestrator (generalist_model — Sistema 2)
    ¦
    +-- SISTEMA 1 (fine-tuned specialists)
    ¦     +-- sistema1-historico  (expert_model_instance — Text LoRA) ? historical data
    ¦     +-- sistema1-vl        (vision_model — VL LoRA)            ? Observe-Think-Act loop
    ¦                                                                   (browser, web, SAP GUI)
    +-- SISTEMA 2 tools (access to external world)
          +-- industrial-expert   (expert_model factory — Text LoRA) ? RAG + MCP live data

Design principles:
  - sistema1-historico has ZERO tools — knowledge baked into fine-tuned weights.
  - sistema1-vl runs the Observe-Think-Act computer use loop (browser, GUI, email, SAP).
  - industrial-expert accesses real-time sensors and internal documents via RAG + MCP.
  - The orchestrator LLM decides which subagents to invoke based on the user query.
  - All Sistema 1 subagents degrade gracefully if their model is unavailable.

IMPORTANT:
  - `generalist_model` MUST be a resolved LLM instance (not a callable).
  - `expert_model` can be an async callable (lazy-resolved — used by industrial-expert).
  - `expert_model_instance` MUST be a resolved LLM instance (used by sistema1-historico).
  - `vision_model` MUST be a resolved LLM instance (or None for graceful degradation).
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.proactiva.agent.factory import create_industrial_agent
from app.domain.proactiva.agent.subagents.system1_subagent import (
    create_system1_historico_agent,
    create_system1_vl_agent,
)
from app.domain.proactiva.agent.prompts.generalist import build_generalist_prompt
from app.core.config import settings


def create_generalist_orchestrator(
    generalist_model,
    expert_model,
    expert_model_instance=None,
    vision_model=None,
    worker_model=None,
    checkpointer=None,
    store=None,
    mcp_tools_context: str = "No dynamic tools currently registered.",
    enable_knowledge: bool = True,
    enable_mcp: bool = True,
    enable_system1: bool = True,
    enable_computer_use: bool = True,
    vl_replay_buffer=None,
) -> object:
    """
    Creates the Generalist Orchestrator — the top-level router (Sistema 2).

    The orchestrator exposes ALL subagents as callable tools to the LLM.
    It invokes whichever are needed based on the user's query — one, several,
    or all simultaneously — like parallel tool/function calls.

    Args:
        generalist_model: Resolved LLM for orchestration (Sistema 2 brain — Gemma 4).
        expert_model: Async factory for IndustrialExpert (lazy, Sistema 2 tool).
        expert_model_instance: Resolved LLM instance for sistema1-historico (Sistema 1).
                               If None, the historical subagent is skipped gracefully.
        vision_model: Resolved multimodal LLM for Sistema 1 VL + Computer Use.
                      If None, all VL subagents are skipped gracefully.
        worker_model: Model for IndustrialExpert sub-subagents. Defaults to generalist_model.
        checkpointer: LangGraph checkpointer for thread-scoped memory.
        store: LangGraph store for user-scoped long-term memory.
        mcp_tools_context: Formatted MCP tools string for IndustrialExpert.
        enable_knowledge: Whether IndustrialExpert can use the RAG knowledge base.
        enable_mcp: Whether IndustrialExpert can call real-time MCP tools.
        enable_system1: Toggle for sistema1-historico (Text LoRA, historical data).
        enable_computer_use: Toggle for sistema1-vl (VL LoRA, Observe-Think-Act loop).
        vl_replay_buffer: VLReplayBuffer instance to store training trajectories.

    Returns:
        A compiled LangGraph graph ready for ainvoke() / astream_events().
    """
    _has_vision = vision_model is not None
    _has_expert = expert_model_instance is not None

    logger.info(
        f"[Orchestrator] Assembling. "
        f"sistema1_historico={enable_system1 and _has_expert}, "
        f"sistema1_vl={enable_computer_use and _has_vision and settings.computer_use_enabled}, "
        f"knowledge={enable_knowledge}, mcp={enable_mcp}"
    )

    all_subagents = []

    # -- Sistema 1: Histórico (Text LoRA — ZERO tools — historical data from weights) --
    if enable_system1:
        s1_hist = create_system1_historico_agent(
            expert_model=expert_model_instance,
            checkpointer=checkpointer,
            store=store,
        )
        if s1_hist is not None:
            all_subagents.append(s1_hist)
            logger.info("[Orchestrator] Sistema 1 Histórico registrado.")
        else:
            logger.warning(
                "[Orchestrator] Sistema 1 Histórico skipped (expert_model_instance unavailable). "
                "Historical queries will fall through to industrial-expert."
            )

    # -- Sistema 1: VL (VL LoRA — Observe-Think-Act computer use loop) -------------
    if enable_computer_use and _has_vision and settings.computer_use_enabled:
        from app.persistence.proactiva.vl_replay_buffer import vl_replay_buffer as default_vl_buffer

        _active_buffer = vl_replay_buffer or default_vl_buffer

        s1_vl = create_system1_vl_agent(
            vision_model=vision_model,
            vl_replay_buffer=_active_buffer,
        )
        if s1_vl is not None:
            all_subagents.append(s1_vl)
            logger.info(
                f"[Orchestrator] Sistema 1 VL (Computer Use) registrado "
                f"(demo_mode={settings.computer_use_demo_mode}, "
                f"max_steps={settings.computer_use_max_steps})."
            )
        else:
            logger.warning(
                "[Orchestrator] Sistema 1 VL skipped (vision_model unavailable). "
                "GUI/web queries will not be handled."
            )
    elif enable_computer_use and not _has_vision:
        logger.warning(
            "[Orchestrator] Sistema 1 VL skipped (vision_model unavailable). "
            "Deploy VL model via OTA to activate."
        )

    # -- Sistema 2: Industrial Expert (Text LoRA — RAG + MCP — live data) ------------
    industrial_expert_graph = create_industrial_agent(
        model=expert_model,
        worker_model=worker_model or generalist_model,
        checkpointer=checkpointer,
        store=store,
        mcp_tools_context=mcp_tools_context,
        enable_knowledge=enable_knowledge,
        enable_mcp=enable_mcp,
    )

    industrial_expert = CompiledSubAgent(
        name="industrial-expert",
        description=(
            "USE for real-time and near-real-time industrial data: "
            "current SCADA/PLC sensor readings, live KPIs, equipment status NOW, "
            "searching internal company manuals (ISO, OSHA, NOM regulations), "
            "and incident report lookup. "
            "DO NOT use for historical data older than 6 months — "
            "use sistema1-historico for that. "
            "DO NOT use for browser, GUI, or web tasks — use sistema1-vl for that."
        ),
        runnable=industrial_expert_graph,
    )
    all_subagents.append(industrial_expert)

    logger.info(f"[Orchestrator] {len(all_subagents)} subagente(s) registrado(s) como tools del orquestador.")

    # -- Assemble Orchestrator ---------------------------------------------------------
    registered_names = [s['name'] if isinstance(s, dict) else s.name for s in all_subagents]
    dynamic_prompt = build_generalist_prompt(registered_names)
    logger.info(f"[Orchestrator] Available subagents injected into prompt: {registered_names}")

    from app.domain.proactiva.agent.memory import create_composite_backend
    return create_deep_agent(
        model=generalist_model,
        system_prompt=dynamic_prompt,
        tools=[],
        subagents=all_subagents,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        checkpointer=checkpointer,
        store=store,
    )

