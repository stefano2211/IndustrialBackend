"""
Sistema 1 Subagents — Fine-tuned experts with ZERO tools.

Sistema 1 = the two fine-tuned models that receive OTA updates from ApiLLMOps.
They answer exclusively from their trained weights — no RAG, no MCP, no external calls.

┌─────────────────────────────────────────────────────────────────────┐
│  SISTEMA 1                                                          │
│  ├── sistema1-historico  Text LoRA (aura_tenant_01-v2)             │
│  │     → Historical SCADA/SAP data > 6 months from weights         │
│  └── sistema1-vl         VL LoRA   (aura_tenant_01-vl)             │
│        → Visual analysis of industrial app screenshots from weights │
└─────────────────────────────────────────────────────────────────────┘

The Generalist Orchestrator (Sistema 2) exposes both as callable tools
and invokes whichever are needed based on the user's query — one, both, or neither.
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.agent.prompts.system1 import SISTEMA1_SYSTEM_PROMPT
from app.domain.agent.prompts.system1_historico import SISTEMA1_HISTORICO_PROMPT
from app.domain.agent.memory import create_composite_backend


def create_system1_historico_agent(
    expert_model,
    checkpointer=None,
    store=None,
) -> CompiledSubAgent | None:
    """
    Creates the Sistema 1 Histórico subagent.

    Uses the fine-tuned TEXT LoRA (aura_tenant_01-v2) to answer historical
    industrial queries directly from its trained weights.

    Args:
        expert_model: A resolved BaseChatModel instance pointing to the text LoRA
                      (e.g., vLLM with model_name='aura_tenant_01-v2').
                      If None, the subagent is skipped gracefully.
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for user-scoped long-term memory.

    Returns:
        CompiledSubAgent ready to be registered in the orchestrator,
        or None if expert_model is not available.
    """
    if expert_model is None:
        logger.warning(
            "[Sistema1-Histórico] expert_model is None — skipping. "
            "Deploy text LoRA via OTA to activate historical subagent."
        )
        return None

    logger.info("[Sistema1-Histórico] Assembling (no tools — fine-tuned text weights).")

    graph = create_deep_agent(
        model=expert_model,
        tools=[],                            # ← ZERO tools by design
        system_prompt=SISTEMA1_HISTORICO_PROMPT,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        subagents=[],
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="sistema1-historico",
        description=(
            "USE for queries about industrial data OLDER THAN 6 MONTHS: "
            "historical SCADA sensor trends, past equipment failures, yearly KPIs, "
            "SAP historical transactions (MB51, ME21N, etc.), operational patterns, "
            "and any data that was collected more than 6 months ago. "
            "This model has this knowledge BAKED INTO its fine-tuned weights — NO external tools. "
            "DO NOT use for real-time or current data — use industrial-expert instead. "
            "DO NOT use for visual screenshot analysis — use sistema1-vl instead."
        ),
        runnable=graph,
    )


def create_system1_vl_agent(
    vision_model,
    checkpointer=None,
    store=None,
) -> CompiledSubAgent | None:
    """
    Creates the Sistema 1 VL subagent.

    Uses the fine-tuned VL LoRA (aura_tenant_01-vl) to analyze screenshots
    and visual content of industrial applications from its trained weights.

    Args:
        vision_model: A resolved multimodal BaseChatModel instance pointing to the VL LoRA
                      (e.g., vLLM with model_name='aura_tenant_01-vl').
                      If None, the subagent is skipped gracefully.
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for user-scoped long-term memory.

    Returns:
        CompiledSubAgent ready to be registered in the orchestrator,
        or None if vision_model is not available.
    """
    if vision_model is None:
        logger.warning(
            "[Sistema1-VL] vision_model is None — skipping. "
            "Deploy VL LoRA via OTA to activate visual analysis subagent."
        )
        return None

    logger.info("[Sistema1-VL] Assembling (no tools — fine-tuned VL weights).")

    graph = create_deep_agent(
        model=vision_model,
        tools=[],                            # ← ZERO tools by design
        system_prompt=SISTEMA1_SYSTEM_PROMPT,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        subagents=[],
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="sistema1-vl",
        description=(
            "USE when the query includes a SCREENSHOT or IMAGE of an industrial application "
            "(SAP GUI, SCADA HMI, PLC panel, or any industrial UI) that requires visual analysis: "
            "reading screen values, describing interface state, identifying UI elements, "
            "or interpreting what is displayed. "
            "This model analyzes visuals from its fine-tuned VL weights — NO external tools. "
            "DO NOT use for real-time sensor data — use industrial-expert instead. "
            "DO NOT use for performing GUI ACTIONS (clicking, typing) — use computer-use-agent instead."
        ),
        runnable=graph,
    )


# ---------------------------------------------------------------------------
# Backward-compat alias — will be removed in a future cleanup
# ---------------------------------------------------------------------------
def create_system1_agent(
    vision_model,
    checkpointer=None,
    store=None,
) -> CompiledSubAgent | None:
    """Deprecated: use create_system1_vl_agent() instead."""
    logger.warning(
        "[Sistema1] create_system1_agent() is deprecated. "
        "Use create_system1_vl_agent() or create_system1_historico_agent()."
    )
    return create_system1_vl_agent(vision_model, checkpointer, store)
