"""
Reactive Agent Service.

High-level interface to invoke the Reactive Orchestrator for event processing.
Unlike AgentService, this service does not stream UI callbacks; it processes
events in the background and returns structured text.
"""

import json
from typing import Optional

from loguru import logger
from langchain_core.messages import HumanMessage

from app.core.llm import LLMFactory
from app.domain.agent.reactive_orchestrator import create_reactive_orchestrator
from app.persistence.reactiva.repositories.reactive_tool_config_repository import ReactiveToolConfigRepository
from app.domain.schemas.event import Event


# Isolated cache for reactive orchestrator graphs
_REACTIVE_GRAPH_CACHE = {}


class ReactiveAgentService:
    """Manages the lifecycle of Reactive Deep Agents."""

    async def _build_mcp_context(self, session) -> str:
        """Fetch all reactive tools and build the dynamic context string."""
        repo = ReactiveToolConfigRepository(session)
        tools = await repo.get_all()
        if not tools:
            return "No dynamic tools registered."

        context = "Available Reactive Tools:\n"
        for t in tools:
            context += f"\n- Name: {t.name}\n"
            context += f"  Description: {t.description}\n"
            if t.parameter_schema:
                schema_str = json.dumps(t.parameter_schema, indent=2, ensure_ascii=False)
                context += f"  Format: {schema_str}\n"

        return context

    async def _get_or_create_graph(self, tenant_id: str, session):
        """Fetch models and build the reactive orchestrator if not cached."""
        cache_key = f"reactive_orchestrator_{tenant_id}"
        
        # Disable cache temporarily to ensure clean tool reloading
        # if cache_key in _REACTIVE_GRAPH_CACHE:
        #     return _REACTIVE_GRAPH_CACHE[cache_key]["agent"]

        logger.info(f"[ReactiveAgentService] Assembling Reactive Orchestrator for tenant: {tenant_id}")

        # Instantiate unified vLLM models
        generalist_model = await LLMFactory.get_llm(model="aura_tenant_01-v2", temperature=0.7)
        expert_model = await LLMFactory.get_llm(model="aura_tenant_01-v2", temperature=0.0)
        
        expert_model_instance = {
            "aura_tenant_01-v2": await LLMFactory.get_llm(model="aura_tenant_01-v2", temperature=0)
        }

        # Contexts
        mcp_tools_context = await self._build_mcp_context(session)

        graph = create_reactive_orchestrator(
            generalist_model=generalist_model,
            expert_model=expert_model,
            expert_model_instance=expert_model_instance,
            mcp_tools_context=mcp_tools_context,
        )

        _REACTIVE_GRAPH_CACHE[cache_key] = {"agent": graph}
        return graph

    async def analyze(self, event: Event, session) -> tuple[str, Optional[str]]:
        """
        Analyze an event using the Reactive Orchestrator.
        Returns (analysis_text, plan_text).
        """
        tenant_id = event.tenant_id
        thread_id = f"event-{event.id}"
        
        logger.info(f"[{thread_id}] Reactive Agent Service: Starting analysis")

        graph = await self._get_or_create_graph(tenant_id, session)

        # Build the event context prompt
        event_ctx = (
            f"You have received a new industrial event. Please analyze it "
            f"and generate a remediation plan.\n\n"
            f"[REACTIVE EVENT CONTEXT]\n"
            f"Event ID: {event.id}\n"
            f"Source: {event.source_type}\n"
            f"Severity: {event.severity}\n"
            f"Title: {event.title}\n"
            f"Description: {event.description}\n"
            f"Payload: {json.dumps(event.raw_payload or {})}\n"
            f"---\n\n"
        )
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "session": session,
            }
        }

        output_text = ""
        
        try:
            # We don't stream callbacks, just wait for the final text
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=event_ctx)]},
                config=config,
            ):
                for node, state in chunk.items():
                    if "messages" in state and len(state["messages"]) > 0:
                        last_msg = state["messages"][-1]
                        if hasattr(last_msg, "content"):
                            output_text = last_msg.content
                            
        except Exception as e:
            logger.error(f"[{thread_id}] Reactive agent error: {e}")
            return f"Error during analysis: {e}", None

        # Parse output for ---PLAN--- separator
        analysis = output_text
        plan = None
        if "---PLAN---" in output_text:
            parts = output_text.split("---PLAN---", 1)
            analysis = parts[0].strip()
            plan = parts[1].strip()

        logger.info(f"[{thread_id}] Analysis complete.")
        return analysis, plan

    async def execute_plan(self, event: Event, plan: str, session) -> list:
        """
        If we ever need the agent to auto-execute steps in the plan.
        Currently just returns a mock trace since actual execution involves
        human-in-the-loop logic or safe external service calls.
        """
        # Feature to implement later: parse plan and execute safe actions
        logger.info(f"[ReactiveAgentService] Executing plan for event {event.id}")
        return [{"status": "skipped", "message": "Auto-execution not yet fully implemented"}]
