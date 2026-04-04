"""
Industrial Agent Factory.

Single Responsibility: Assemble the IndustrialAgent from its components.

The factory takes pre-configured LLM instances and infrastructure backends,
and wires them together into a compiled DeepAgent graph. It does NOT create
LLM instances — that is the responsibility of AgentService.
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.agent.prompts import INDUSTRIAL_SYSTEM_PROMPT, AGENTS_MD_CONTENT
from app.domain.agent.subagents.definitions import get_all_subagents
from app.domain.agent.memory import create_composite_backend
from app.domain.agent.tools.knowledge_tool import ask_knowledge_agent
from app.domain.agent.tools.mcp_tool import call_dynamic_mcp


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
    Creates the IndustrialAgent — the domain expert for safety & compliance.

    Args:
        model: Pre-configured BaseChatModel for the main orchestration turn.
        worker_model: Model for subagent execution. Defaults to `model` if not provided.
        checkpointer: LangGraph checkpointer for thread-scoped conversation persistence.
        store: LangGraph store for long-term, user-scoped memory (cross-thread).
        custom_system_prompt: Extra instructions appended to the base prompt.
        mcp_tools_context: Formatted string listing all registered MCP tools.
        enable_knowledge: Whether to include the knowledge-researcher subagent (RAG).
        enable_mcp: Whether to include the mcp-orchestrator subagent (real-time data).

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

    # ── 2. Build Subagents ─────────────────────────────────────────────────
    # worker_model defaults to the main model if not explicitly provided.
    effective_worker = worker_model or model
    subagents = []

    for sa_def in get_all_subagents():
        name = sa_def["name"]

        # --- knowledge-researcher: RAG over user documents ---
        if name == "knowledge-researcher":
            if not enable_knowledge:
                logger.debug("[IndustrialAgent] Skipping knowledge-researcher (disabled).")
                continue
            graph = create_deep_agent(
                model=effective_worker,
                tools=[ask_knowledge_agent],
                system_prompt=sa_def["system_prompt"],
                subagents=[],
            )
            subagents.append(
                CompiledSubAgent(
                    name=sa_def["name"],
                    description=sa_def["description"],
                    graph=graph,
                )
            )

        # --- mcp-orchestrator: real-time sensor / API data ---
        elif name == "mcp-orchestrator":
            if not enable_mcp:
                logger.debug("[IndustrialAgent] Skipping mcp-orchestrator (disabled).")
                continue
            # Inject the dynamic tools context into the MCP subagent's prompt
            mcp_prompt = sa_def["system_prompt"].format(
                dynamic_tools_context=mcp_tools_context
            )
            graph = create_deep_agent(
                model=effective_worker,
                tools=[call_dynamic_mcp],
                system_prompt=mcp_prompt,
                subagents=[],
            )
            subagents.append(
                CompiledSubAgent(
                    name=sa_def["name"],
                    description=sa_def["description"],
                    graph=graph,
                )
            )

        # --- general-assistant: fallback, no tools needed ---
        # Passed as a plain dict — deepagents accepts this format for simple subagents.
        else:
            subagents.append(sa_def)

    logger.info(f"[IndustrialAgent] Assembled {len(subagents)} subagent(s).")

    # ── 3. Assemble and Compile Agent ──────────────────────────────────────
    return create_deep_agent(
        model=model,
        tools=[],           # All tools are encapsulated inside CompiledSubAgents
        system_prompt=full_prompt,
        subagents=subagents,
        backend=create_composite_backend,
        memory=["/AGENTS.md"],
        skills=["/skills/industrial_safety/"],
        checkpointer=checkpointer,
        store=store,
    )
