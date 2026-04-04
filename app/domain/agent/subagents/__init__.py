"""
Subagents package — all subagent definitions and satellite agents.

Public exports:
  - Subagent definitions (KNOWLEDGE, MCP, GENERAL) and get_all_subagents()
  - Satellite agent factories (SAP, Google, Office)
  - Sistema 1 VL fine-tuned expert factory
"""

from app.domain.agent.subagents.definitions import (
    KNOWLEDGE_SUBAGENT,
    MCP_SUBAGENT,
    GENERAL_SUBAGENT,
    get_all_subagents,
)
from app.domain.agent.subagents.satellite import (
    create_sap_agent,
    create_google_agent,
    create_office_agent,
)
from app.domain.agent.subagents.system1_subagent import create_system1_agent

__all__ = [
    "KNOWLEDGE_SUBAGENT",
    "MCP_SUBAGENT",
    "GENERAL_SUBAGENT",
    "get_all_subagents",
    "create_sap_agent",
    "create_google_agent",
    "create_office_agent",
    "create_system1_agent",
]
