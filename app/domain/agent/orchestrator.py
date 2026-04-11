"""
Generalist Orchestrator Factory.

Routes queries to the appropriate specialist via a flat 2-level hierarchy:

  Orchestrator (generalist_model — Qwen 32b)
    ├── Sistema1Subagent   (vision_model — VL fine-tuned)   ← historical + vision
    ├── ComputerUseAgent   (vision_model — VL fine-tuned)   ← Observe-Think-Act loop
    └── IndustrialExpert   (expert_model — Aura fine-tuned) ← RAG + MCP real-time

Design principles:
  - Sistema 1 handles knowledge BAKED INTO ITS WEIGHTS (historical, fine-tuned patterns).
  - ComputerUseAgent handles GUI interaction tasks (Macrohard Digital Optimus Local).
  - IndustrialExpert handles LIVE DATA (RAG search + MCP sensors).
  - vision_model can be None: orchestrator degrades gracefully without VL subagents.

IMPORTANT:
  - `generalist_model` MUST be a resolved LLM instance (not a callable).
  - `expert_model` can be an async callable (lazy-resolved on first use).
  - `vision_model` MUST be a resolved LLM instance (or None for graceful degradation).
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.agent.factory import create_industrial_agent
from app.domain.agent.subagents.system1_subagent import create_system1_agent
from app.domain.agent.prompts.generalist import GENERALIST_SYSTEM_PROMPT
from app.core.config import settings


def create_generalist_orchestrator(
    generalist_model,
    expert_model,
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
    Creates the Generalist Orchestrator — the top-level router.

    Args:
        generalist_model: Resolved LLM for orchestration. MUST be a real LLM instance.
        expert_model: Resolved LLM OR async factory for the IndustrialExpert.
        vision_model: Resolved multimodal LLM for Sistema 1 + Computer Use (Qwen2.5-VL).
                      If None, VL subagents are skipped gracefully.
        worker_model: Model for IndustrialExpert sub-subagents. Defaults to generalist_model.
        checkpointer: LangGraph checkpointer for thread-scoped memory.
        store: LangGraph store for user-scoped long-term memory.
        mcp_tools_context: Formatted MCP tools string for IndustrialExpert.
        enable_knowledge: Whether IndustrialExpert can use the RAG knowledge base.
        enable_mcp: Whether IndustrialExpert can call real-time MCP tools.
        enable_system1: Toggle for Sistema 1 historical/VL subagent.
        enable_computer_use: Toggle for Computer Use Agent (Digital Optimus Local).
        vl_replay_buffer: VLReplayBuffer instance to store training trajectories.

    Returns:
        A compiled LangGraph graph ready for ainvoke() / astream_events().
    """
    _has_vision = vision_model is not None
    logger.info(
        f"[Orchestrator] Assembling. "
        f"enable_system1={enable_system1 and _has_vision}, "
        f"enable_computer_use={enable_computer_use and _has_vision and settings.computer_use_enabled}, "
        f"enable_knowledge={enable_knowledge}, enable_mcp={enable_mcp}"
    )

    all_subagents = []

    # ── 1. Sistema 1 (VL Fine-tuned — Historical + Vision) ────────────────
    if enable_system1:
        sistema1 = create_system1_agent(
            vision_model=vision_model,
            checkpointer=checkpointer,
            store=store,
        )
        if sistema1 is not None:
            all_subagents.append(sistema1)
            logger.info("[Orchestrator] Sistema 1 VL subagent registered.")
        else:
            logger.warning(
                "[Orchestrator] Sistema 1 skipped (vision_model unavailable). "
                "Historical queries will fall through to IndustrialExpert."
            )

    # ── 2. Computer Use Agent (Digital Optimus Local — Macrohard) ─────────
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
                "DO NOT use for answering questions — use industrial-expert for that."
            ),
            graph=computer_use_graph,
        )
        all_subagents.append(computer_use_subagent)
        logger.info(
            f"[Orchestrator] Computer Use Agent registered "
            f"(demo_mode={settings.computer_use_demo_mode}, "
            f"max_steps={settings.computer_use_max_steps})."
        )
    elif enable_computer_use and not _has_vision:
        logger.warning(
            "[Orchestrator] Computer Use Agent skipped (vision_model unavailable). "
            "Deploy VL model via OTA to activate."
        )

    # ── 3. Industrial Expert (Aura Fine-tuned — RAG + MCP) ────────────────
    async def _load_industrial_expert():
        """Async factory: resolve expert_model and assemble IndustrialExpert."""
        resolved_expert = await expert_model() if callable(expert_model) else expert_model
        return create_industrial_agent(
            model=resolved_expert,
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
            "use sistema1-experto for that. "
            "DO NOT use for GUI actions — use computer-use-agent for that."
        ),
        graph=_load_industrial_expert,
    )
    all_subagents.append(industrial_expert)

    logger.info(f"[Orchestrator] {len(all_subagents)} subagent(s) registered.")

    # ── 4. Assemble Orchestrator ───────────────────────────────────────────
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
