"""
Generalist Orchestrator Factory.

Routes queries to the appropriate specialist via a flat 2-level hierarchy:

  Orchestrator (generalist_model — Sistema 2)
    │
    ├── SISTEMA 1 (fine-tuned, ZERO tools — respond from weights)
    │     ├── sistema1-historico  (expert_model_instance — Text LoRA) ← historical data
    │     └── sistema1-vl        (vision_model — VL LoRA)            ← visual analysis
    │
    └── SISTEMA 2 tools (access to external world)
          ├── computer-use-agent  (vision_model — VL)                ← Observe-Think-Act loop
          └── industrial-expert   (expert_model factory — Text LoRA) ← RAG + MCP live data

Design principles:
  - Sistema 1 has ZERO tools — knowledge is baked into fine-tuned weights.
  - Sistema 2 (industrial-expert, computer-use-agent) accesses external data/GUIs.
  - The orchestrator LLM decides which subagents to invoke (one, several, or all)
    based on the user query — like parallel tool calls. No forced ordering.
  - All Sistema 1 subagents degrade gracefully if their model is unavailable.

IMPORTANT:
  - `generalist_model` MUST be a resolved LLM instance (not a callable).
  - `expert_model` can be an async callable (lazy-resolved — used by industrial-expert).
  - `expert_model_instance` MUST be a resolved LLM instance (used by sistema1-historico).
  - `vision_model` MUST be a resolved LLM instance (or None for graceful degradation).
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.agent.factory import create_industrial_agent
from app.domain.agent.subagents.system1_subagent import (
    create_system1_historico_agent,
    create_system1_vl_agent,
)
from app.domain.agent.prompts.generalist import GENERALIST_SYSTEM_PROMPT
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
        generalist_model: Resolved LLM for orchestration (Sistema 2 brain).
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
        enable_system1: Toggle for Sistema 1 subagents (historical + VL).
        enable_computer_use: Toggle for Computer Use Agent (Digital Optimus Local).
        vl_replay_buffer: VLReplayBuffer instance to store training trajectories.

    Returns:
        A compiled LangGraph graph ready for ainvoke() / astream_events().
    """
    _has_vision = vision_model is not None
    _has_expert = expert_model_instance is not None

    logger.info(
        f"[Orchestrator] Assembling. "
        f"sistema1_historico={enable_system1 and _has_expert}, "
        f"sistema1_vl={enable_system1 and _has_vision}, "
        f"computer_use={enable_computer_use and _has_vision and settings.computer_use_enabled}, "
        f"knowledge={enable_knowledge}, mcp={enable_mcp}"
    )

    all_subagents = []

    # ── Sistema 1: Histórico (Text LoRA — ZERO tools — historical data from weights) ──
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

    # ── Sistema 1: VL (VL LoRA — ZERO tools — visual analysis from weights) ──────────
    if enable_system1:
        s1_vl = create_system1_vl_agent(
            vision_model=vision_model,
            checkpointer=checkpointer,
            store=store,
        )
        if s1_vl is not None:
            all_subagents.append(s1_vl)
            logger.info("[Orchestrator] Sistema 1 VL registrado.")
        else:
            logger.warning(
                "[Orchestrator] Sistema 1 VL skipped (vision_model unavailable). "
                "Visual queries will fall through to industrial-expert."
            )

    # ── Sistema 2: Computer Use Agent (GUI Observe→Think→Act) ────────────────────────
    if enable_computer_use and _has_vision and settings.computer_use_enabled:
        from app.domain.agent.subagents.computer_use_subagent import create_computer_use_agent
        from app.persistence.vl_replay_buffer import vl_replay_buffer as default_vl_buffer

        _active_buffer = vl_replay_buffer or default_vl_buffer

        computer_use_graph = create_computer_use_agent(
            vision_llm=vision_model,
            vl_replay_buffer=_active_buffer,
        )

        computer_use_subagent = CompiledSubAgent(
            name="computer-use-agent",
            description=(
                "USE when the user asks to PERFORM AN ACTION on a computer interface: "
                "navigating SAP GUI transactions (MB51, ME21N, VL02N, etc.), "
                "clicking buttons, filling forms, updating records in ERP/database screens, "
                "or sending emails via an email client. "
                "This agent SEES the screen and ACTS on it step by step. "
                "Provide a clear, single instruction describing what to accomplish. "
                "DO NOT use for answering questions — use industrial-expert for that. "
                "DO NOT use for visual analysis only — use sistema1-vl for that."
            ),
            runnable=computer_use_graph,
        )
        all_subagents.append(computer_use_subagent)
        logger.info(
            f"[Orchestrator] Computer Use Agent registrado "
            f"(demo_mode={settings.computer_use_demo_mode}, "
            f"max_steps={settings.computer_use_max_steps})."
        )
    elif enable_computer_use and not _has_vision:
        logger.warning(
            "[Orchestrator] Computer Use Agent skipped (vision_model unavailable). "
            "Deploy VL model via OTA to activate."
        )

    # ── Sistema 2: Industrial Expert (Text LoRA — RAG + MCP — live data) ────────────
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
            "DO NOT use for visual screenshot analysis — use sistema1-vl for that. "
            "DO NOT use for GUI actions — use computer-use-agent for that."
        ),
        runnable=industrial_expert_graph,
    )
    all_subagents.append(industrial_expert)

    logger.info(f"[Orchestrator] {len(all_subagents)} subagente(s) registrado(s) como tools del orquestador.")

    # ── Assemble Orchestrator ─────────────────────────────────────────────────────────
    from app.domain.agent.memory import create_composite_backend
    return create_deep_agent(
        model=generalist_model,
        system_prompt=GENERALIST_SYSTEM_PROMPT,
        tools=[],
        subagents=all_subagents,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        checkpointer=checkpointer,
        store=store,
    )

