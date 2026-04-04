"""
Industrial AI Agent System — Public API.

Import from here, not from internal modules:
    from app.domain.agent import create_industrial_agent, create_generalist_orchestrator
"""

from app.domain.agent.factory import create_industrial_agent
from app.domain.agent.orchestrator import create_generalist_orchestrator

__all__ = ["create_industrial_agent", "create_generalist_orchestrator"]
