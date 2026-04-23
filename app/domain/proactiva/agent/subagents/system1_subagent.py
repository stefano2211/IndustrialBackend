"""
Sistema 1 Subagents.

Sistema 1 = the two fine-tuned models that receive OTA updates from ApiLLMOps.

┌─────────────────────────────────────────────────────────────────────┐
│  SISTEMA 1                                                          │
│  ├── sistema1-historico  Text LoRA (aura_tenant_01-v2)             │
│  │     → Historical SCADA/SAP data > 6 months (from weights)       │
│  └── sistema1-vl         VL LoRA   (aura_tenant_01-vl)             │
│        → Observe-Think-Act loop: browser, web, SAP GUI, email      │
└─────────────────────────────────────────────────────────────────────┘

The Generalist Orchestrator exposes both as callable tools and invokes
whichever are needed based on the user's query.
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.proactiva.agent.prompts.system1_historico import SISTEMA1_HISTORICO_PROMPT
from app.domain.proactiva.agent.memory import create_composite_backend


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
    vl_replay_buffer=None,
) -> CompiledSubAgent | None:
    """
    Creates the Sistema 1 VL subagent.

    Implements the Observe-Think-Act computer use loop using the VL LoRA
    (aura_tenant_01-vl). This is the primary agent for any task requiring
    browser navigation, GUI interaction, SAP/ERP transactions, email, or
    any website visit.

    Args:
        vision_model: A resolved multimodal BaseChatModel instance pointing to the VL LoRA
                      (e.g., vLLM with model_name='aura_tenant_01-vl').
                      If None, the subagent is skipped gracefully.
        vl_replay_buffer: Optional VLReplayBuffer to store training trajectories.

    Returns:
        CompiledSubAgent ready to be registered in the orchestrator,
        or None if vision_model is not available.
    """
    if vision_model is None:
        logger.warning(
            "[Sistema1-VL] vision_model is None — skipping. "
            "Deploy VL LoRA via OTA to activate computer use subagent."
        )
        return None

    logger.info("[Sistema1-VL] Assembling (Observe-Think-Act computer use loop).")

    from app.domain.proactiva.agent.subagents.computer_use_subagent import create_computer_use_agent

    graph = create_computer_use_agent(
        vision_llm=vision_model,
        vl_replay_buffer=vl_replay_buffer,
    )

    return CompiledSubAgent(
        name="sistema1-vl",
        description=(
            "USE for ANY task requiring a real browser, website visit, or screen interaction. "
            "Capabilities: "
            "(1) LIVE WEB ACCESS — searches (Google, Bing, news), current prices, weather, "
            "any live page content, web forms, online services. Opens a real browser and SEES "
            "exactly what is on screen. "
            "(2) EMAIL — compose, send, and read emails via Gmail or any web email client. "
            "(3) SAP/ERP GUI — navigate transactions (MB51, ME21N, VL02N, etc.), click buttons, "
            "fill forms, read and update records in any ERP or industrial web interface. "
            "(4) ANY WEBSITE — if the task requires visiting a URL, this is the correct agent. "
            "Pass a clear, self-contained instruction: target site + action + what to report back. "
            "Do NOT answer live web-content questions from memory — always use this agent."
        ),
        runnable=graph,
    )


# ---------------------------------------------------------------------------
# Backward-compat alias — will be removed in a future cleanup
# ---------------------------------------------------------------------------
def create_system1_agent(
    vision_model,
    vl_replay_buffer=None,
) -> CompiledSubAgent | None:
    """Deprecated: use create_system1_vl_agent() instead."""
    logger.warning(
        "[Sistema1] create_system1_agent() is deprecated. "
        "Use create_system1_vl_agent() or create_system1_historico_agent()."
    )
    return create_system1_vl_agent(vision_model, vl_replay_buffer)
