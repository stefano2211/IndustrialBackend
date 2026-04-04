"""
Generalist Orchestrator Factory.

Routes queries to the appropriate specialist via a flat 2-level hierarchy:

  Orchestrator (generalist_model — Qwen 32b)
    ├── Sistema1Subagent (vision_model — VL fine-tuned)   ← historical + vision
    └── IndustrialExpert (expert_model — Aura fine-tuned) ← RAG + MCP real-time

Design principles:
  - Sistema 1 handles knowledge BAKED INTO ITS WEIGHTS (historical, fine-tuned patterns).
  - IndustrialExpert handles LIVE DATA (RAG search + MCP sensors).
  - Both specialize; the orchestrator only routes — it does NOT reason about domain data.
  - vision_model can be None: the orchestrator degrades gracefully without Sistema 1.

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
) -> object:
    """
    Creates the Generalist Orchestrator — the top-level router.

    Args:
        generalist_model: Resolved LLM for orchestration. MUST be a real LLM instance.
        expert_model: Resolved LLM OR async factory for the IndustrialExpert.
                      Resolved lazily inside the graph to avoid loading into VRAM upfront.
        vision_model: Resolved multimodal LLM for Sistema 1 (e.g., qwen2.5-vl:7b).
                      If None, Sistema 1 is skipped — system operates without it.
        worker_model: Model used by the IndustrialExpert's sub-subagents (RAG, MCP).
                      Defaults to generalist_model (sub-agents don't need the fine-tuned model).
        checkpointer: LangGraph checkpointer for thread-scoped conversation memory.
        store: LangGraph store for user-scoped long-term memory (cross-thread).
        mcp_tools_context: Formatted MCP tools string passed down to IndustrialExpert.
        enable_knowledge: Whether IndustrialExpert can use the RAG knowledge base.
        enable_mcp: Whether IndustrialExpert can call real-time MCP tools.
        enable_system1: Master toggle for Sistema 1 (also requires vision_model != None).

    Returns:
        A compiled LangGraph graph ready for ainvoke() / astream_events().
    """
    logger.info(
        f"[Orchestrator] Assembling. "
        f"enable_system1={enable_system1 and vision_model is not None}, "
        f"enable_knowledge={enable_knowledge}, enable_mcp={enable_mcp}"
    )

    all_subagents = []

    # ── 1. Sistema 1 (VL Fine-tuned — Historical + Vision) ────────────────
    # Registered FIRST so the orchestrator considers it before the IndustrialExpert.
    # This way, historical queries are answered from the fine-tuned weights
    # without turning on the heavier RAG/MCP pipeline.
    if enable_system1:
        sistema1 = create_system1_agent(
            vision_model=vision_model,   # None → skipped gracefully with a warning
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

    # ── 2. Industrial Expert (Aura Fine-tuned — RAG + MCP) ────────────────
    # Lazy-loaded to avoid instantiating the fine-tuned model into VRAM until needed.
    # Sub-subagents (knowledge-researcher, mcp-orchestrator) use worker_model,
    # which defaults to generalist_model — they don't need the fine-tuned expert.
    async def _load_industrial_expert():
        """Async factory: resolve expert_model and assemble IndustrialExpert."""
        resolved_expert = await expert_model() if callable(expert_model) else expert_model
        return create_industrial_agent(
            model=resolved_expert,
            worker_model=worker_model or generalist_model,   # sub-agents use generalist
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
            "use sistema1-experto for that."
        ),
        graph=_load_industrial_expert,
    )
    all_subagents.append(industrial_expert)

    logger.info(f"[Orchestrator] {len(all_subagents)} subagent(s) registered.")

    # ── 3. Assemble Orchestrator ───────────────────────────────────────────
    return create_deep_agent(
        model=generalist_model,
        system_prompt=GENERALIST_SYSTEM_PROMPT,
        tools=all_subagents,
        subagents=[],
        checkpointer=checkpointer,
        store=store,
    )
