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

from deepagents import create_deep_agent

from app.domain.agent.prompts import INDUSTRIAL_SYSTEM_PROMPT, AGENTS_MD_CONTENT
from app.domain.agent.subagents import get_all_subagents
from app.domain.agent.middleware import get_all_middleware
from app.domain.agent.memory import create_composite_backend
from app.domain.agent.tools.knowledge_tool import ask_knowledge_agent


def create_industrial_agent(model=None, checkpointer=None, store=None, custom_system_prompt: str = None):
    """
    Creates a Deep Agent configured for Industrial Safety & Compliance.

    Args:
        model: A pre-configured BaseChatModel instance.
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for long-term memory.
        custom_system_prompt: Optional custom instructions to append to the base prompt.

    Returns:
        A compiled LangGraph graph (Deep Agent).
    """
    full_prompt = INDUSTRIAL_SYSTEM_PROMPT
    if custom_system_prompt:
        full_prompt += f"\n\n## ADICIONAL USER INSTRUCTIONS:\n{custom_system_prompt}"

    return create_deep_agent(
        model=model,
        tools=[ask_knowledge_agent],
        system_prompt=full_prompt,
        subagents=get_all_subagents(),
        backend=create_composite_backend,
        middleware=get_all_middleware(),
        memory=["/AGENTS.md"],
        checkpointer=checkpointer,
        store=store,
    )
