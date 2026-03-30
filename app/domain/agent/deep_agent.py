"""
Deep Agent Factory for Industrial Safety & Compliance AI.

Assembles the agent from dedicated modules:
  - prompts/       → System prompt & domain memory
  - memory/        → Storage backends (persistent + ephemeral)
  - tools/         → LangChain tools (knowledge search)
  - subagents.py   → Sub-agent definitions
  - middleware.py  → Tool call middleware

The factory has ONE responsibility: assembling the agent (SRP).
"""

from deepagents import create_deep_agent, CompiledSubAgent

from app.domain.agent.prompts import INDUSTRIAL_SYSTEM_PROMPT, AGENTS_MD_CONTENT
from app.domain.agent.subagents import get_all_subagents
from app.domain.agent.middleware import get_all_middleware
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
    enable_mcp: bool = True
):
    """
    Creates a Deep Agent configured for Industrial Safety & Compliance.

    Args:
        model: A pre-configured BaseChatModel instance.
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for long-term memory.
        custom_system_prompt: Optional custom instructions to append to the base prompt.
        mcp_tools_context: A string-formatted list of dynamic tools and their descriptions.

    Returns:
        A compiled LangGraph graph (Deep Agent).
    """
    full_prompt = INDUSTRIAL_SYSTEM_PROMPT.format(
        dynamic_tools_context=mcp_tools_context
    )
    if custom_system_prompt:
        full_prompt += f"\n\n## ADICIONAL USER INSTRUCTIONS:\n{custom_system_prompt}"

    # Prepare sub-agents with dynamic context
    subagents = []
    
    for sa in get_all_subagents():
        sa_copy = sa.copy()
        
        sa_tools = []
        # Filter Knowledge Subagent
        if sa_copy["name"] == "knowledge-researcher":
            if not enable_knowledge:
                continue
            sa_tools.append(ask_knowledge_agent)
            
        # Filter MCP Subagent
        elif sa_copy["name"] == "mcp-orchestrator":
            if not enable_mcp:
                continue
            sa_copy["system_prompt"] = sa_copy["system_prompt"].format(
                dynamic_tools_context=mcp_tools_context
            )
            sa_tools.append(call_dynamic_mcp)
            
        if sa_tools and worker_model:
            sa_graph = create_deep_agent(
                model=worker_model,
                tools=sa_tools,
                system_prompt=sa_copy["system_prompt"],
                subagents=[]
            )
            compiled_sa = CompiledSubAgent(
                name=sa_copy["name"],
                description=sa_copy["description"],
                graph=sa_graph
            )
            compiled_sa["system_prompt"] = sa_copy.get("system_prompt", "")
            subagents.append(compiled_sa)
        else:
            if "tools" in sa_copy:
                sa_copy["tools"] = sa_tools
            subagents.append(sa_copy)

    return create_deep_agent(
        model=model,
        tools=[],  # Tools are now encapsulated in the CompiledSubAgents
        system_prompt=full_prompt,
        subagents=subagents,
        backend=create_composite_backend,
        middleware=get_all_middleware(),
        memory=["/AGENTS.md"],
        skills=["/skills/industrial_safety/"], # Virtual path pointing to our new SKILL.md
        checkpointer=checkpointer,
        store=store,
    )
