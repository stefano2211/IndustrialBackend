"""
Industrial Agent Factory.

Single Responsibility: Assemble the IndustrialAgent from its components.

The factory takes pre-configured LLM instances and infrastructure backends,
and wires them together into a compiled DeepAgent graph. It does NOT create
LLM instances — that is the responsibility of AgentService.
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.proactiva.agent.prompts import INDUSTRIAL_SYSTEM_PROMPT, AGENTS_MD_CONTENT
from app.domain.proactiva.agent.memory import create_composite_backend
from app.domain.proactiva.agent.tools.knowledge_tool import ask_knowledge_agent
from app.domain.proactiva.agent.tools.mcp_tool import call_dynamic_mcp


def create_industrial_agent(
    model=None,
    worker_model=None,
    checkpointer=None,
    store=None,
    custom_system_prompt: str = None,
    mcp_tools_context: str = "No dynamic tools currently registered.",
    enable_knowledge: bool = True,
    enable_mcp: bool = True,
) -> object:
    """
    Creates the IndustrialAgent — the domain expert for data extraction.

    Args:
        model: Pre-configured BaseChatModel.
        worker_model: Ignored in this simplified architecture.
        checkpointer: LangGraph checkpointer for thread-scoped conversation persistence.
        store: LangGraph store for long-term, user-scoped memory (cross-thread).
        custom_system_prompt: Extra instructions appended to the base prompt.
        mcp_tools_context: Formatted string listing all registered MCP tools.
        enable_knowledge: Whether to enable RAG tool.
        enable_mcp: Whether to enable MCP tools.

    Returns:
        A compiled LangGraph graph ready for ainvoke() / astream_events().
    """
    # ── 1. Build System Prompt ─────────────────────────────────────────────
    full_prompt = INDUSTRIAL_SYSTEM_PROMPT.format(
        dynamic_tools_context=mcp_tools_context
    )
    if custom_system_prompt:
        full_prompt += f"\n\n## ADDITIONAL USER INSTRUCTIONS:\n{custom_system_prompt}"

    logger.debug(
        f"[IndustrialAgent] Building agent. "
        f"enable_knowledge={enable_knowledge}, enable_mcp={enable_mcp}"
    )

    # ── 2. Register Active Tools ───────────────────────────────────────────
    active_tools = []
    if enable_knowledge:
        active_tools.append(ask_knowledge_agent)
    if enable_mcp:
        active_tools.append(call_dynamic_mcp)

    logger.info(f"[IndustrialAgent] Assembled with {len(active_tools)} tools.")

    # ── 3. Assemble and Compile Agent ──────────────────────────────────────
    return create_deep_agent(
        model=model,
        tools=active_tools,
        system_prompt=full_prompt,
        subagents=[],
        backend=create_composite_backend,
        checkpointer=checkpointer,
        store=store,
    )
