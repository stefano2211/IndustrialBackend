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
        "Eres un especialista en documentos de Seguridad Industrial. "
        "Usa `ask_knowledge_agent` para buscar en la base de conocimiento. "
        "Extrae los fragmentos más relevantes, cita el documento y la sección, "
        "y devuelve un resumen claro y conciso. "
        "Si no encuentras resultados, indícalo explícitamente."
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
        "Eres un Orquestador de Datos Industriales. "
        "Llama a `call_dynamic_mcp` con el `tool_config_name` exacto y los argumentos correctos. "
        "El tool retorna un JSON con `key_figures` (métricas numéricas) y `key_values` (info descriptiva). "
        "Analiza esos datos y presenta un resumen claro y estructurado al usuario. "
        "NO muestres el JSON crudo. NO inventes datos."
        "\n\n**HERRAMIENTAS DISPONIBLES:**\n"
        "{dynamic_tools_context}"
    ),
    "tools": [call_dynamic_mcp],
}


def get_all_subagents() -> list[dict]:
    """Returns the list of all configured sub-agents."""
    return [KNOWLEDGE_SUBAGENT, MCP_SUBAGENT, GENERAL_SUBAGENT]
