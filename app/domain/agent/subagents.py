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
        "Use `ask_knowledge_agent` to search the knowledge base. "
        "Extract the most relevant excerpts, cite the document and section, "
        "and return a clear and concise summary. "
        "If you find no results, explicitly state so. "
        "ALWAYS reply in the language the user uses."
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
        "information would be needed. "
        "ALWAYS reply in the language the user uses."
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
        "in your call to `call_dynamic_mcp`. Only include filters if the user asks for them; "
        "if they ask for everything, pass an empty arguments dict.\n\n"

        "Filtering Rules:\n"
        "- To filter by categorical values (text): include the key 'key_values' in arguments. "
        "  Its value is a dict where each key is the field name and the value is a list of allowed strings.\n"
        "- CRITICAL: If the user provides a direct reference like `Category.Value` (e.g., `Status.Running`, `TagName.BombaA`), "
        "  you MUST interpret it as an exact filter where `Category` equals `Value`. "
        "  Map it to arguments as `{{'key_values': {{'Category': ['Value']}}}}`.\n"
        "- To filter by numeric ranges: include the key 'key_figures' in arguments. "
        "  Its value is a list of dicts, each with 'field' (numeric field name), "
        "  and optionally 'min' and/or 'max'.\n"
        "- You can combine both filters in the same call.\n"
        "- Use EXACTLY the field names listed in 'Filterable fields' of each tool.\n"
        "- The values for key_values must match exactly what is listed.\n\n"

        "## CALL EFFICIENCY\n"
        "- Try to answer the user's question with a SINGLE well-filtered call.\n"
        "- Avoid calling again without filters 'just to see what else there is' if the first filtered call already answered the request.\n"
        "- Only make multiple calls if strictly necessary to complete a complex task.\n\n"

        "**AVAILABLE TOOLS:**\n"
        "{dynamic_tools_context}"
    ),

    "tools": [call_dynamic_mcp],
}



def get_all_subagents() -> list[dict]:
    """Returns the list of all configured sub-agents."""
    return [KNOWLEDGE_SUBAGENT, MCP_SUBAGENT, GENERAL_SUBAGENT]
