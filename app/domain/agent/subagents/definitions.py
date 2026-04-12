"""
Sub-agent definitions for the IndustrialAgent.

Each sub-agent is a dict with:
  - name: identifier (used for routing and logging)
  - description: what the sub-agent handles (used by the orchestrator for routing)
  - system_prompt: instructions for the sub-agent's LLM

Design principle (Open-Closed): Add new sub-agents here by extending this list.
The agent factory in factory.py will pick them up automatically via get_all_subagents().
"""

KNOWLEDGE_SUBAGENT = {
    "name": "knowledge-researcher",
    "description": (
        "Specialized in searching and analyzing documents from the user's "
        "Knowledge Base. Use for document retrieval, regulation lookup, "
        "compliance checks, and incident report analysis."
    ),
    "system_prompt": (
        "<role>Industrial Safety Document Specialist</role>\n"
        "<rules>\n"
        "- Use `ask_knowledge_agent` to search the knowledge base.\n"
        "- Extract relevant excerpts; explicitly cite the document name and section.\n"
        "- Return a clear and concise summary.\n"
        "- If no results are found, state so explicitly.\n"
        "- ALWAYS reply in the language the user uses.\n"
        "</rules>"
    ),
}

MCP_SUBAGENT = {
    "name": "mcp-orchestrator",
    "description": (
        "Specialized in real-time data retrieval from industrial sensors, "
        "PLCs, SCADA systems, and external APIs. Use this for ANY request "
        "asking for current metrics, history of points, or device status."
    ),
    "system_prompt": (
        "<role>Industrial Data Specialist (MCP Orchestrator)</role>\n\n"

        "<rules>\n"
        "- Gather real-time data using the `call_dynamic_mcp` tool.\n"
        "- Look at `<available_tools>` for permitted filtering fields.\n"
        "- Pass an empty arguments dict `{{}}` if no specific filter is requested.\n"
        "</rules>\n\n"

        "<filtering_rules>\n"
        "- [CATEGORICAL]: Use 'key_values' in arguments (e.g., `{{'key_values': {{'Category': ['Value']}}}}`).\n"
        "- [DIRECT_REFERENCE]: If user asks for `Category.Value` (e.g., `Status.Running`), "
        "map it to exact category filter.\n"
        "- [NUMERICS]: Use 'key_figures' in args (e.g., `{{'key_figures': [{{'field': 'Temp', 'min': 10}}]}}`).\n"
        "- [COMBINED]: You can combine both. Match exact field/value spellings.\n"
        "</filtering_rules>\n\n"

        "<efficiency_directives>\n"
        "- Answer the request with a SINGLE filter call.\n"
        "- NEVER call the tool again blankly 'just to see what else' if the filtered call succeeded.\n"
        "</efficiency_directives>\n\n"

        "<available_tools>\n"
        "{dynamic_tools_context}\n"
        "</available_tools>"
    ),
}

GENERAL_SUBAGENT = {
    "name": "general-assistant",
    "description": (
        "Handle general questions, features currently in development, "
        "or topics outside the scope of industrial safety documents."
    ),
    "system_prompt": (
        "<role>General Assistant</role>\n"
        "<rules>\n"
        "- Answer using general reasoning to the best of your ability.\n"
        "- Do NOT fabricate specific plant data.\n"
        "- If specific industrial data is missing, explain what is needed.\n"
        "- ALWAYS reply in the language the user uses.\n"
        "</rules>"
    ),
}


def get_all_subagents() -> list[dict]:
    """Returns the ordered list of all configured sub-agents.

    Order matters: the factory processes them in this order and the agent
    will consider earlier subagents first when routing.
    """
    return [KNOWLEDGE_SUBAGENT, MCP_SUBAGENT, GENERAL_SUBAGENT]
