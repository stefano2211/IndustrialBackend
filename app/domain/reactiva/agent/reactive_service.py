"""
Reactive Agent Service.

High-level interface to invoke the Reactive Orchestrator for event processing.
Unlike AgentService, this service does not stream UI callbacks; it processes
events in the background and returns structured text.

Output flow:
  analyze()      → (analysis_text, plan_text, execute_instruction)
  execute_plan() → launches ComputerUseSubagent with execute_instruction
"""

import json
from datetime import datetime
from typing import Optional

from loguru import logger
from langchain_core.messages import HumanMessage

from app.core.llm import LLMFactory
from app.core.config import settings
from app.domain.proactiva.agent.service import _vllm_model_exists
from app.domain.reactiva.agent.reactive_orchestrator import create_reactive_orchestrator
from app.persistence.reactiva.repositories.reactive_tool_config_repository import ReactiveToolConfigRepository
from app.domain.reactiva.schemas.event import Event


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

        # ── LoRA probe + fallback (shared pattern with AgentService) ────────
        lora_target = settings.system1_historico_model
        if not settings.system1_force_base_model:
            lora_ready = await _vllm_model_exists(settings.vllm_base_url, lora_target)
        else:
            lora_ready = False

        model_name = lora_target if lora_ready else settings.system1_base_model
        if lora_ready:
            logger.info(f"[ReactiveAgentService] LoRA '{lora_target}' available — using it.")
        else:
            logger.warning(
                f"[ReactiveAgentService] LoRA '{lora_target}' NOT loaded "
                f"(force_base={settings.system1_force_base_model}). "
                f"Falling back to base model: {settings.system1_base_model}"
            )

        # Instantiate unified vLLM models with the resolved model name
        generalist_model = await LLMFactory.get_llm(model=model_name, temperature=0.7)
        expert_model = await LLMFactory.get_llm(model=model_name, temperature=0.0)
        
        expert_model_instance = {
            model_name: await LLMFactory.get_llm(model=model_name, temperature=0)
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

    async def analyze(self, event: Event, session) -> tuple[str, Optional[str], Optional[str]]:
        """
        Analyze an event using the Reactive Orchestrator.

        Returns:
            (analysis_text, plan_text, execute_instruction)

            execute_instruction is the content of the ---EXECUTE--- section when
            sistema1-vl is available and the plan requires GUI interaction.
            It is passed as-is to execute_plan() → ComputerUseSubagent.
        """
        tenant_id = event.tenant_id
        thread_id = f"event-{event.id}"

        logger.info(f"[{thread_id}] Reactive Agent Service: Starting analysis")

        graph = await self._get_or_create_graph(tenant_id, session)

        # Build the event context prompt
        event_ctx = (
            f"You have received a new industrial event. Please analyze it, "
            f"generate a remediation plan, and — if sistema1-vl is available "
            f"and GUI interaction is required — include an ---EXECUTE--- section "
            f"with a self-contained instruction for the Computer Use agent.\n\n"
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
            return f"Error during analysis: {e}", None, None

        # ── Parse three-section output ─────────────────────────────────────────
        # Section 1: Analysis  (before ---PLAN---)
        # Section 2: Plan      (between ---PLAN--- and ---EXECUTE---)
        # Section 3: Execute   (after ---EXECUTE---)
        analysis = output_text
        plan: Optional[str] = None
        execute_instruction: Optional[str] = None

        if "---PLAN---" in output_text:
            pre_plan, _, post_plan = output_text.partition("---PLAN---")
            analysis = pre_plan.strip()

            if "---EXECUTE---" in post_plan:
                plan_part, _, execute_part = post_plan.partition("---EXECUTE---")
                plan = plan_part.strip()
                execute_instruction = execute_part.strip()
                logger.info(f"[{thread_id}] Execute instruction extracted ({len(execute_instruction)} chars).")
            else:
                plan = post_plan.strip()

        logger.info(f"[{thread_id}] Analysis complete.")
        return analysis, plan, execute_instruction

    async def execute_plan(
        self,
        event: Event,
        plan: str,
        session,
        execute_instruction: Optional[str] = None,
    ) -> list:
        """
        Execute the remediation plan autonomously via the Computer Use agent.

        If reactive_computer_use_enabled is False, skips silently.
        If execute_instruction is provided (from the ---EXECUTE--- section),
        it is passed as the instruction to the ComputerUseSubagent.
        Otherwise, the full plan text is used as a fallback instruction.

        Returns:
            List of action dicts with type, result, and timestamp.
        """
        logger.info(f"[ReactiveAgentService] execute_plan() called for event {event.id}")

        if not settings.reactive_computer_use_enabled:
            logger.warning(
                "[ReactiveAgentService] reactive_computer_use_enabled=False — "
                "skipping Computer Use execution."
            )
            return [{"status": "skipped", "reason": "computer_use_disabled"}]

        # ── Resolve VL model — probe LoRA, fallback to base ────────────────────
        vl_lora_name = getattr(settings, "system1_vl_model", "aura_tenant_01-vl")
        base_model_name = getattr(settings, "system1_base_model", settings.local_vllm_model)
        vllm_url = getattr(settings, "local_vllm_url", "")

        vl_ready = await _vllm_model_exists(vllm_url, vl_lora_name) if vllm_url else False
        resolved_model = vl_lora_name if vl_ready else base_model_name

        if not vl_ready:
            logger.warning(
                f"[ReactiveAgentService] VL LoRA '{vl_lora_name}' not loaded — "
                f"falling back to base model '{resolved_model}'."
            )

        try:
            vl_model = await LLMFactory.get_llm(model=resolved_model, temperature=0.0)
        except Exception as e:
            logger.error(f"[ReactiveAgentService] Could not initialize VL model: {e}")
            return [{"status": "error", "message": str(e)}]

        # ── Assemble the Computer Use graph ────────────────────────────────────
        from app.domain.proactiva.agent.subagents.computer_use_subagent import create_computer_use_agent
        graph = create_computer_use_agent(vision_llm=vl_model, vl_replay_buffer=None)

        # ── Build the instruction ───────────────────────────────────────────────
        # Prefer the ---EXECUTE--- section (concise, single-paragraph instruction).
        # Fall back to the full plan text with the event header prepended.
        if execute_instruction:
            instruction = (
                f"[EVENTO INDUSTRIAL: {event.title} | Severidad: {event.severity.upper()}]\n\n"
                f"{execute_instruction}"
            )
        else:
            instruction = (
                f"[EVENTO INDUSTRIAL: {event.title} | Severidad: {event.severity.upper()}]\n\n"
                f"Ejecuta el siguiente plan de remediación paso a paso:\n\n{plan}"
            )

        logger.info(
            f"[ReactiveAgentService] Launching Computer Use loop for event {event.id}. "
            f"Model: {resolved_model} | Instruction length: {len(instruction)} chars."
        )

        # ── Run the Observe-Think-Act loop ──────────────────────────────────────
        thread_id = f"execute-{event.id}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "tenant_id": event.tenant_id,
            }
        }

        result_summary = ""
        steps_taken = 0

        try:
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=instruction)]},
                config=config,
            ):
                for node, state in chunk.items():
                    if state.get("is_complete"):
                        result_summary = state.get("result_summary", "")
                        steps_taken = state.get("steps_taken", 0)

        except Exception as e:
            logger.error(f"[ReactiveAgentService] Computer Use loop failed: {e}")
            return [{
                "type": "computer_use_error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }]

        logger.info(
            f"[ReactiveAgentService] Computer Use loop complete for event {event.id}. "
            f"Steps: {steps_taken} | Result: {result_summary[:120]}"
        )

        return [{
            "type": "computer_use",
            "model": resolved_model,
            "steps_taken": steps_taken,
            "result": result_summary,
            "timestamp": datetime.utcnow().isoformat(),
        }]
