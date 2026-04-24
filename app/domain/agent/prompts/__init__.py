from app.domain.agent.prompts.industrial import INDUSTRIAL_SYSTEM_PROMPT, AGENTS_MD_CONTENT, TEMPORAL_ROUTER_PROMPT
from app.domain.agent.prompts.system1 import SISTEMA1_SYSTEM_PROMPT
from app.domain.agent.prompts.system1_historico import SISTEMA1_HISTORICO_PROMPT
from app.domain.agent.prompts.reactive_industrial import REACTIVE_INDUSTRIAL_PROMPT, REACTIVE_AGENTS_MD_CONTENT
from app.domain.agent.prompts.reactive_orchestrator import build_reactive_orchestrator_prompt

__all__ = [
    "INDUSTRIAL_SYSTEM_PROMPT",
    "AGENTS_MD_CONTENT",
    "TEMPORAL_ROUTER_PROMPT",
    "SISTEMA1_SYSTEM_PROMPT",
    "SISTEMA1_HISTORICO_PROMPT",
    "REACTIVE_INDUSTRIAL_PROMPT",
    "REACTIVE_AGENTS_MD_CONTENT",
    "build_reactive_orchestrator_prompt",
]
