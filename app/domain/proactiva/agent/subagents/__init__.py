"""
Subagents package — all subagent definitions and satellite agents.

Public exports:
  - Satellite agent factories (SAP, Google, Office)
  - Sistema 1 VL fine-tuned expert factory
"""

from app.domain.proactiva.agent.subagents.satellite import (
    create_sap_agent,
    create_google_agent,
    create_office_agent,
)
from app.domain.proactiva.agent.subagents.system1_subagent import create_system1_agent

__all__ = [
    "create_sap_agent",
    "create_google_agent",
    "create_office_agent",
    "create_system1_agent",
]
