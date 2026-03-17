"""
Sub-agent definitions for the Industrial Safety Deep Agent.

Each sub-agent is a dict with:
  - name: identifier
  - description: what the sub-agent handles (used for routing)
  - system_prompt: instructions for the sub-agent
  - tools: list of LangChain tools available to the sub-agent

Add new sub-agents here without modifying the agent factory (Open-Closed).
"""

from app.domain.agent.tools.knowledge_tool import ask_knowledge_agent
from app.domain.agent.tools.mcp_tool import call_dynamic_mcp


KNOWLEDGE_SUBAGENT = {
    "name": "knowledge-researcher",
    "description": (
        "Specialized in searching and analyzing documents from the user's "
        "Knowledge Base. Use for document retrieval, regulation lookup, "
        "compliance checks, and incident report analysis."
    ),
    "system_prompt": (
        "You are an Industrial Safety document specialist. "
        "Your job is to search through the user's Knowledge Base using the "
        "ask_knowledge_agent tool, extract relevant information, and return "
        "a clear, concise summary of your findings. "
        "Always cite the source document name. "
        "If no relevant documents are found, say so clearly."
    ),
    "tools": [ask_knowledge_agent],
}

GENERAL_SUBAGENT = {
    "name": "general-assistant",
    "description": (
        "Handle general questions, features currently in development, "
        "or topics outside the scope of industrial safety documents."
    ),
    "system_prompt": (
        "You are a helpful general assistant. Answer the user's question "
        "to the best of your ability. If the question requires specific "
        "documents or data you don't have access to, explain what additional "
        "information would be needed."
    ),
    "tools": [],
}

MCP_SUBAGENT = {
    "name": "mcp-orchestrator",
    "description": (
        "Specialized in real-time data retrieval from industrial sensors, "
        "PLCs, Scada systems, and external APIs. Use this for ANY request "
        "asking for current metrics, history of points, or device status."
    ),
    "system_prompt": (
        "You are an Industrial Data Orchestrator. Your mission is to "
        "connect to various MCP sources to retrieve real-time figures. "
        "When a user asks for 'metrics', 'status', or 'values', identify "
        "the correct tool name from your descriptions and call `call_dynamic_mcp`. "
        "\n\n**AVAILABLE DYNAMIC TOOLS:**\n"
        "{dynamic_tools_context}"
        "\n\nAlways present data normalized: Key Figures for numbers, "
        "Key Values for states. If multiple sources are needed, aggregate "
        "the results before responding."
    ),
    "tools": [call_dynamic_mcp],
}


def get_all_subagents() -> list[dict]:
    """Returns the list of all configured sub-agents."""
    return [KNOWLEDGE_SUBAGENT, MCP_SUBAGENT, GENERAL_SUBAGENT]
