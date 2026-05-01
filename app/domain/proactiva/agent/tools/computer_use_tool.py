"""Re-export wrapper --- implementation moved to app.domain.shared.agent.tools.computer_use_tool."""
from app.domain.shared.agent.tools.computer_use_tool import (
    take_screenshot,
    get_page_context,
    execute_action,
    run_shell_command,
    task_complete,
    COMPUTER_USE_TOOLS,
    get_clean_b64,
)

__all__ = [
    "take_screenshot",
    "get_page_context",
    "execute_action",
    "run_shell_command",
    "task_complete",
    "COMPUTER_USE_TOOLS",
    "get_clean_b64",
]
