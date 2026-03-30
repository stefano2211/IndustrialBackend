"""
Creates a two-level hierarchy matching the Aura 2-model architecture:
  - Orchestrator/Worker: User-selected model (Director & Tool Executor)
  - Industrial Expert: Aura fine-tuned model (Domain Specialist)

Both models run on the same local Ollama container or via external providers.
"""

from deepagents import create_deep_agent
from deepagents import CompiledSubAgent
from loguru import logger

from app.domain.agent.deep_agent import create_industrial_agent
from app.domain.agent.satellite_agents import create_sap_agent, create_google_agent, create_office_agent
from app.domain.agent.prompts.generalist import GENERALIST_SYSTEM_PROMPT


def create_generalist_orchestrator(
    generalist_model,
    expert_model,
    worker_model=None,
    checkpointer=None,
    store=None,
    mcp_tools_context: str = "No dynamic tools currently registered.",
    enable_knowledge: bool = True,
    enable_mcp: bool = True,
    enable_satellite: bool = True,
):
    """
    Creates a Generalist → Specialized Tools orchestration system.
    """
    logger.info("[GeneralistOrchestrator] Assembling hierarchical tool-based architecture...")

    # ── Step 1: Build Specialized Agent Tools ──────────────────────────────────
    
    async def get_industrial_expert_graph():
        """Lazy loader for the industrial expert graph."""
        # If expert_model is a factory (callable), call it and wait for the LLM instance
        if callable(expert_model):
            resolved_expert = await expert_model()
        else:
            resolved_expert = expert_model
        
        return create_industrial_agent(
            model=resolved_expert,
            worker_model=worker_model,
            checkpointer=checkpointer,
            store=store,
            mcp_tools_context=mcp_tools_context,
            enable_knowledge=enable_knowledge,
            enable_mcp=enable_mcp,
        )

    industrial_tool = CompiledSubAgent(
        name="industrial-expert",
        description=(
            "USE ONLY for proprietary industrial data (SCADA, sensors, safety ISO, internal manuals). "
            "NEVER use for greetings, general questions, or simple chat."
        ),
        graph=get_industrial_expert_graph, # Pass the lazy loader
    )

    # Satellite Systems (SAP, Google, Office)
    satellite_tools = []
    if enable_satellite:
        satellite_tools = [
            create_sap_agent(expert_model, checkpointer, store),
            create_google_agent(expert_model, checkpointer, store),
            create_office_agent(expert_model, checkpointer, store),
        ]

    all_tools = [industrial_tool] + satellite_tools

    logger.info(f"[GeneralistOrchestrator] Registered {len(all_tools)} specialized tools.")

    # ── Step 2: Assemble the Generalist Orchestrator ───────────────────────────
    # The generalist chooses the tool based on the query.
    orchestrator = create_deep_agent(
        model=generalist_model,
        system_prompt=GENERALIST_SYSTEM_PROMPT,
        tools=all_tools,  # Refactored: experts are now tools
        subagents=[],     # Empty subagents: moving to native tool-calling
        checkpointer=checkpointer,
        store=store,
    )

    logger.info("[GeneralistOrchestrator] Tool-based Orchestrator assembled successfully.")
    return orchestrator
