"""
Reactive Sistema 1 Subagents.

These are the reactive-domain wrappers for Sistema 1. They use the shared
LoRA-backed agent constructors (infraestructura común) but with:
  - Reactive-specific system prompts for the historical subagent
  - Reactive-specific descriptions and naming context

This guarantees that `domain.reactiva.agent.*` NEVER imports from
`domain.proactiva.agent.*` — the only cross-domain imports are from
`domain.shared.agent.subagents.system1` (LoRA infrastructure) and
`domain.shared.agent.memory_backends` (LangGraph persistence).
"""

from deepagents import CompiledSubAgent
from loguru import logger

from app.domain.shared.agent.subagents.system1 import (
    create_system1_historico_agent,
    create_system1_vl_agent,
)
from app.domain.reactiva.agent.prompts.reactive_system1_historico import (
    REACTIVE_SISTEMA1_HISTORICO_PROMPT,
)


def create_reactive_system1_historico_agent(
    expert_model,
    checkpointer=None,
    store=None,
) -> CompiledSubAgent | None:
    """
    Creates the REACTIVE Sistema 1 Histórico subagent.

    Uses the same text LoRA (aura_tenant_01-v2) as the proactive counterpart,
    but with a system prompt tuned for event-driven historical diagnosis:
    pattern matching against past incidents, root-cause correlation, and
    corrective-action suggestions from historical weights.

    Args:
        expert_model: Resolved BaseChatModel pointing to the text LoRA.
        checkpointer: LangGraph checkpointer.
        store: LangGraph store for long-term memory.

    Returns:
        CompiledSubAgent or None if model unavailable.
    """
    logger.info("[ReactiveSystem1] Assembling reactive sistema1-historico (event diagnosis).")

    agent = create_system1_historico_agent(
        expert_model=expert_model,
        checkpointer=checkpointer,
        store=store,
        system_prompt=REACTIVE_SISTEMA1_HISTORICO_PROMPT,
    )

    if agent is None:
        logger.warning(
            "[ReactiveSystem1] Reactive sistema1-historico skipped — expert_model unavailable."
        )
        return None

    # Override description to reflect reactive event-diagnosis role
    return CompiledSubAgent(
        name="sistema1-historico",
        description=(
            "USE for HISTORICAL PATTERN MATCHING on industrial EVENTS older than 6 months: "
            "past equipment failure patterns, historical root-cause signatures, "
            "seasonal anomaly baselines, previous corrective-action outcomes. "
            "This model has knowledge BAKED INTO fine-tuned weights — NO external tools. "
            "DO NOT use for real-time sensor readings — use industrial-expert instead. "
            "DO NOT use for GUI actions — use sistema1-vl instead."
        ),
        runnable=agent.runnable,
    )


def create_reactive_system1_vl_agent(
    vision_model,
    vl_replay_buffer=None,
) -> CompiledSubAgent | None:
    """
    Creates the REACTIVE Sistema 1 VL subagent.

    Uses the same VL LoRA (aura_tenant_01-vl) and Observe-Think-Act loop as the
    proactive counterpart. The computer-use agent is domain-agnostic by design;
    the reactive independence is achieved by having a separate instantiation path
    and reactive-specific caller context.

    Args:
        vision_model: Resolved multimodal BaseChatModel pointing to the VL LoRA.
        vl_replay_buffer: Optional VLReplayBuffer.

    Returns:
        CompiledSubAgent or None if model unavailable.
    """
    logger.info("[ReactiveSystem1] Assembling reactive sistema1-vl (Observe-Think-Act for events).")

    agent = create_system1_vl_agent(
        vision_model=vision_model,
        vl_replay_buffer=vl_replay_buffer,
    )

    if agent is None:
        logger.warning(
            "[ReactiveSystem1] Reactive sistema1-vl skipped — vision_model unavailable."
        )
        return None

    # Description tuned for reactive remediation context
    return CompiledSubAgent(
        name="sistema1-vl",
        description=(
            "USE for AUTONOMOUS GUI EXECUTION during reactive event remediation: "
            "open SCADA HMI to acknowledge alarms, navigate SAP/ERP to trigger work orders, "
            "send emergency emails, update setpoints on industrial dashboards, fill web forms. "
            "Operates a real browser and SEES the screen. "
            "Pass a SINGLE precise instruction: target app/URL + exact field values + expected outcome. "
            "Do NOT use for answering historical questions — use sistema1-historico instead. "
            "Do NOT use for real-time data queries — use industrial-expert instead."
        ),
        runnable=agent.runnable,
    )
