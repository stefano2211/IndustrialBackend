"""Re-export wrapper --- implementation moved to app.domain.shared.agent.subagents.system1."""
from app.domain.shared.agent.subagents.system1 import (
    create_system1_agent,
    create_system1_historico_agent,
    create_system1_vl_agent,
)

__all__ = [
    "create_system1_agent",
    "create_system1_historico_agent",
    "create_system1_vl_agent",
]
