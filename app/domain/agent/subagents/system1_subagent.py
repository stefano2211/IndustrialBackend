"""
Sistema 1 Subagent — Fine-tuned Vision-Language Expert.

This is the second specialist in the orchestrator hierarchy (alongside IndustrialExpert).

Key characteristics:
  - Uses a fine-tuned VL model (e.g., Qwen 3.2 VL 7b trained via ApiLLMOps OTA pipeline).
  - Has ZERO tools — all knowledge comes from the model's fine-tuned weights.
  - Handles two types of queries:
      1. HISTORICAL: data older than ~6 months (baked into training, no RAG needed).
      2. VISUAL (future): receives screenshots of industrial apps for computer use.

When to route here (for the orchestrator's description):
  - User asks about historical trends, incidents, or data from months/years ago.
  - User shares a screenshot of SAP, SCADA, HMI, or another industrial application.
  - Queries about past fine-tuning events or learned operational patterns.
"""

from deepagents import create_deep_agent, CompiledSubAgent
from loguru import logger

from app.domain.agent.prompts.system1 import SISTEMA1_SYSTEM_PROMPT
from app.domain.agent.memory import create_composite_backend


def create_system1_agent(
    vision_model,
    checkpointer=None,
    store=None,
) -> CompiledSubAgent | None:
    """
    Creates the Sistema 1 VL subagent.

    Args:
        vision_model: A pre-configured multimodal BaseChatModel instance
                      (e.g., Ollama with qwen2.5-vl:7b or fine-tuned variant).
                      If None, the subagent is skipped gracefully.
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for user-scoped long-term memory.

    Returns:
        CompiledSubAgent ready to be registered in the orchestrator,
        or None if vision_model is not available.
    """
    if vision_model is None:
        logger.warning(
            "[Sistema1] vision_model is None — Sistema 1 subagent will be skipped. "
            "Set system1_enabled=True and configure system1_model in settings."
        )
        return None

    logger.info("[Sistema1] Assembling Sistema 1 VL subagent (no tools — fine-tuned weights).")

    # Sistema 1 has NO tools. Its knowledge is baked into the fine-tuned model weights.
    # The absence of tools is intentional — it forces the model to answer from training.
    graph = create_deep_agent(
        model=vision_model,
        tools=[],                           # ← ZERO tools by design
        system_prompt=SISTEMA1_SYSTEM_PROMPT,
        backend=create_composite_backend,   # shared memory backend for user context
        memory=["/AGENTS.md"],             # injects domain memory from VFS
        subagents=[],
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="sistema1-experto",
        description=(
            "USE for: (1) historical queries about industrial data older than 6 months "
            "(trends, past incidents, yearly KPIs, equipment failure history) — "
            "this model has this data baked into its fine-tuned weights. "
            "(2) Queries that include a screenshot of an industrial application "
            "(SAP, SCADA, HMI panel) where visual analysis is needed. "
            "DO NOT use for real-time sensor data or current readings — use "
            "industrial-expert instead."
        ),
        graph=graph,
    )
