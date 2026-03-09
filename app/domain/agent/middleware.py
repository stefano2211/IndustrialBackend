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


def get_all_middleware() -> list:
    """Returns the list of all active middleware."""
    return [log_tool_calls]
