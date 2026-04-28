"""
Reactive Factory for Subagents.

Builds the reactive-specific subagents, primarily the Reactive Industrial Agent
which uses the reactive MCP and RAG tools.
"""

from loguru import logger
from deepagents import create_deep_agent, CompiledSubAgent

from app.domain.reactiva.agent.prompts.reactive_industrial import (
    REACTIVE_INDUSTRIAL_PROMPT,
)
from app.domain.reactiva.agent.tools.reactive_knowledge_tool import ask_reactive_knowledge
from app.domain.reactiva.agent.tools.reactive_mcp_tool import call_reactive_mcp
from app.domain.proactiva.agent.memory import create_composite_backend


def create_reactive_industrial_agent(
    model,
    checkpointer=None,
    store=None,
    mcp_tools_context: str = "No dynamic tools registered.",
    enable_knowledge: bool = True,
    enable_mcp: bool = True,
) -> CompiledSubAgent:
    """
    Creates the Reactive Industrial Expert subagent.

    This subagent is responsible for data extraction in the reactive domain.
    It uses reactive-specific tools to fetch SOPs and read real-time sensors.
    """
    logger.info("[ReactiveFactory] Assembling Reactive Industrial Expert.")

    active_tools = []
    if enable_knowledge:
        logger.info("[ReactiveFactory] Reactive Knowledge Base Tool included.")
        active_tools.append(ask_reactive_knowledge)
    else:
        logger.info("[ReactiveFactory] Knowledge Base Tool DISABLED.")

    if enable_mcp:
        logger.info("[ReactiveFactory] Reactive MCP Tool included.")
        active_tools.append(call_reactive_mcp)
    else:
        logger.info("[ReactiveFactory] MCP Tool DISABLED.")

    # Inject dynamic MCP tools context into the system prompt
    formatted_system_prompt = REACTIVE_INDUSTRIAL_PROMPT.format(
        dynamic_tools_context=mcp_tools_context
    )

    graph = create_deep_agent(
        model=model,
        tools=active_tools,
        system_prompt=formatted_system_prompt,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        subagents=[],  # Flat hierarchy: Expert directly uses tools
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="industrial-expert",
        description=(
            "USE for ANY industrial data extraction: SCADA, PLC sensors, KPIs, "
            "SOPs, emergency procedures, maintenance manuals, safety guidelines. "
            "Provides CURRENT, REAL-TIME data and DOCUMENT context."
        ),
        runnable=graph,
    )
