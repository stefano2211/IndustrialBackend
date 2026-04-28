"""
Reactive Orchestrator Factory.

Builds the top-level Reactive Generalist Orchestrator that triages events
and generates remediation plans by delegating to subagents.
"""

from loguru import logger
from deepagents import create_deep_agent

from app.domain.reactiva.agent.prompts.reactive_orchestrator import build_reactive_orchestrator_prompt
from app.domain.reactiva.agent.reactive_factory import create_reactive_industrial_agent
from app.domain.proactiva.agent.subagents.system1_subagent import (
    create_system1_historico_agent,
    create_system1_vl_agent,
)
from app.domain.proactiva.agent.memory import create_composite_backend
from app.core.config import settings


def create_reactive_orchestrator(
    generalist_model,
    expert_model,
    expert_model_instance=None,
    checkpointer=None,
    store=None,
    mcp_tools_context: str = "No dynamic tools registered.",
    enable_knowledge: bool = True,
    enable_mcp: bool = True,
    enable_system1: bool = True,
) -> object:
    """
    Creates the Reactive Orchestrator graph.
    
    Subagents:
      - industrial-expert (Reactive RAG + MCP)
      - sistema1-historico (Shared fine-tuned Text LoRA)
      - (Optional) sistema1-vl (Shared fine-tuned VL LoRA — disabled by default in config)
    """
    logger.info("[ReactiveOrchestrator] Assembling Reactive Orchestrator.")

    subagents = []
    registered_subagent_names = []

    # 1. Reactive Industrial Expert
    industrial_expert = create_reactive_industrial_agent(
        model=expert_model,
        checkpointer=checkpointer,
        store=store,
        mcp_tools_context=mcp_tools_context,
        enable_knowledge=enable_knowledge,
        enable_mcp=enable_mcp,
    )
    subagents.append(industrial_expert)
    registered_subagent_names.append("industrial-expert")

    # 2. Sistema 1 Histórico (Shared from proactive domain)
    if enable_system1 and expert_model_instance is not None:
        historico_model = expert_model_instance.get("aura_tenant_01-v2")
        if historico_model:
            sistema1_historico = create_system1_historico_agent(
                expert_model=historico_model,
                checkpointer=checkpointer,
                store=store,
            )
            if sistema1_historico:
                subagents.append(sistema1_historico)
                registered_subagent_names.append("sistema1-historico")

    # 3. Sistema 1 VL (Shared from proactive domain - Guarded by config)
    if enable_system1 and settings.reactive_computer_use_enabled and expert_model_instance is not None:
        vl_model = expert_model_instance.get("aura_tenant_01-vl")
        if vl_model:
            logger.warning("[ReactiveOrchestrator] DANGER: Computer Use (VL) is enabled for reactive mode.")
            sistema1_vl = create_system1_vl_agent(
                vision_model=vl_model,
                vl_replay_buffer=None,
            )
            if sistema1_vl:
                subagents.append(sistema1_vl)
                registered_subagent_names.append("sistema1-vl")

    # Build prompt with only registered subagents
    system_prompt = build_reactive_orchestrator_prompt(registered_subagent_names)

    # Compile Top-Level Agent
    graph = create_deep_agent(
        model=generalist_model,
        tools=[],  # Orchestrator uses no tools directly, only subagents
        system_prompt=system_prompt,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        subagents=subagents,
        checkpointer=checkpointer,
        store=store,
    )

    return graph
