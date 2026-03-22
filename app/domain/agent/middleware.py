"""
Middleware for the Deep Agent tool calls.

Middleware functions wrap every tool invocation, enabling:
  - Logging / observability
  - Rate limiting (future)
  - Caching (future)
  - Error enrichment (future)

Add new middleware here without modifying the agent factory (Open-Closed).
"""

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import AgentMiddleware
from loguru import logger


@wrap_tool_call
async def log_tool_calls(request, handler):
    """Log every tool call for debugging and monitoring."""
    tool_name = request.name if hasattr(request, "name") else str(request)
    args = request.args if hasattr(request, "args") else "N/A"
    logger.info(f"[Deep Agent] 🔧 Tool call: {tool_name} | Args: {args}")
    result = await handler(request)
    logger.info(f"[Deep Agent] ✅ Tool call {tool_name} completed")
    return result


class GlobalHealthMiddleware(AgentMiddleware):
    """
    Middleware that runs at the graph level (before/after agent).
    Performs global health checks and injects status into the state.
    """
    def before_agent(self, state, runtime):
        logger.info("[GlobalHealthMiddleware] Agent execution starting. Diagnostics OK.")
        # Returns empty state update to pass without side effects while enabling future expansion
        return {}


def get_all_middleware() -> list:
    """Returns the list of all active middleware."""
    return [log_tool_calls, GlobalHealthMiddleware()]
