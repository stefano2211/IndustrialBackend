"""
Generalist Orchestrator Factory — Magentic-One Pattern.

Creates a two-level hierarchy:
  - Top: Generalist agent (llama3.1:8b on Ollama) — Director/planner
  - Sub: Industrial Expert (qwen3.5:9b fine-tuned) — Domain specialist

The Expert agent is wrapped as a CompiledSubAgent so it runs in its own
isolated context window, keeping the generalist's context clean.

Both models run on the same local Ollama container (zero external dependencies).
"""

from deepagents import create_deep_agent
from deepagents.subagents import CompiledSubAgent
from loguru import logger

from app.domain.agent.deep_agent import create_industrial_agent
from app.domain.agent.prompts.generalist import GENERALIST_SYSTEM_PROMPT


def create_generalist_orchestrator(
    generalist_model,
    expert_model,
    checkpointer=None,
    store=None,
    mcp_tools_context: str = "No dynamic tools currently registered.",
    enable_knowledge: bool = True,
    enable_mcp: bool = True,
):
    """
    Creates a two-level Generalist → Expert orchestration system.

    Args:
        generalist_model: ChatOllama instance for the generalist (e.g. llama3.1:8b).
                          Acts as the director: routes, plans, and synthesizes.
        expert_model:     ChatOllama instance for the industrial expert (e.g. qwen3.5:9b).
                          Fine-tuned via LLMOps. Has access to Qdrant RAG + MCP tools.
        checkpointer:     LangGraph PostgreSQL checkpointer (shared — same conversation memory).
        store:            LangGraph cross-thread store (shared — same user memory).
        mcp_tools_context: String description of available MCP tools, injected into the expert's prompt.
        enable_knowledge: Enable the knowledge-researcher sub-agent in the expert.
        enable_mcp:       Enable the mcp-orchestrator sub-agent in the expert.

    Returns:
        A compiled LangGraph graph (Generalist Orchestrator).
    """
    logger.info("[GeneralistOrchestrator] Assembling two-level orchestration hierarchy...")

    # ── Step 1: Build the Expert Industrial Agent (compiled LangGraph graph) ──
    # This is the existing agent — zero changes to its internals.
    expert_graph = create_industrial_agent(
        model=expert_model,
        checkpointer=checkpointer,
        store=store,
        mcp_tools_context=mcp_tools_context,
        enable_knowledge=enable_knowledge,
        enable_mcp=enable_mcp,
    )

    # ── Step 2: Wrap Expert as a CompiledSubAgent ──────────────────────────────
    # CompiledSubAgent gives the expert its own isolated context window.
    # The generalist only sees the expert's final answer — not its tool calls.
    industrial_subagent = CompiledSubAgent(
        name="industrial-expert",
        description=(
            "Expert AI specialized in industrial operations. "
            "Access to real-time SCADA/PLC telemetry, sensor data, equipment status, "
            "manufacturing KPIs, environmental monitoring, industrial safety regulations "
            "(OSHA, ISO, NOM), and internal company documents (manuals, audits, incident reports). "
            "Delegates MUST include the user's full question for proper context."
        ),
        graph=expert_graph,
    )

    logger.info("[GeneralistOrchestrator] Expert CompiledSubAgent registered: industrial-expert")

    # ── Step 3: Assemble the Generalist Orchestrator ───────────────────────────
    # The generalist runs its own Ollama model and uses the expert as a sub-agent.
    orchestrator = create_deep_agent(
        model=generalist_model,
        system_prompt=GENERALIST_SYSTEM_PROMPT,
        subagents=[industrial_subagent],
        checkpointer=checkpointer,
        store=store,
    )

    logger.info("[GeneralistOrchestrator] Generalist Orchestrator assembled successfully.")
    return orchestrator
