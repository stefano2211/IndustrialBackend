"""
Event Processor — decides what to do based on event severity.

Severity routing:
  low      → analyze → save analysis → completed
  medium   → analyze → save → notify SSE → wait for human approval
  high     → analyze + build plan → auto-execute via AgentService → completed
  critical → analyze + build plan → auto-execute immediately → completed
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from app.domain.schemas.event import Event
from app.persistence.db import async_session_factory
from app.persistence.reactiva.repositories.event_repository import EventRepository
from app.core.reactiva.event_queue import broadcast_sse


class EventProcessor:
    """Processes a single Event: analyzes, plans, executes, persists."""

    async def process(self, event: Event) -> None:
        """Entry point. Runs inside the worker loop."""
        async with async_session_factory() as session:
            repo = EventRepository(session)
            try:
                await self._run(event, repo)
            except Exception as exc:
                logger.error(f"[EventProcessor] Failed processing event {event.id}: {exc}")
                await repo.update_status(event.id, "failed")
                await broadcast_sse({"event": "status_update", "data": {"id": str(event.id), "status": "failed"}})

    async def _run(self, event: Event, repo: EventRepository) -> None:
        severity = event.severity.lower()

        await repo.update_status(event.id, "analyzing")
        await broadcast_sse({"event": "status_update", "data": {"id": str(event.id), "status": "analyzing"}})

        analysis, plan, execute_instruction = await self._analyze(event)
        await repo.update_analysis(event.id, analysis=analysis, plan=plan)

        if severity == "low":
            await repo.update_status(event.id, "completed")
            await broadcast_sse({"event": "analysis_ready", "data": {"id": str(event.id), "status": "completed", "analysis": analysis}})
            logger.info(f"[EventProcessor] LOW event {event.id} analyzed and completed.")

        elif severity == "medium":
            await repo.update_status(event.id, "awaiting_approval")
            await broadcast_sse({"event": "analysis_ready", "data": {"id": str(event.id), "status": "awaiting_approval", "analysis": analysis}})
            logger.info(f"[EventProcessor] MEDIUM event {event.id} awaiting human approval.")

        elif severity in ("high", "critical"):
            await repo.update_status(event.id, "executing")
            await broadcast_sse({"event": "status_update", "data": {"id": str(event.id), "status": "executing"}})
            actions = await self._execute(event, plan, execute_instruction)
            await repo.update_analysis(event.id, analysis=analysis, plan=plan, actions=actions)
            await repo.update_status(event.id, "completed")
            await broadcast_sse({"event": "status_update", "data": {"id": str(event.id), "status": "completed", "actions": actions}})
            logger.info(f"[EventProcessor] {severity.upper()} event {event.id} auto-executed.")

    async def _analyze(self, event: Event) -> tuple[str, Optional[str], Optional[str]]:
        """
        Calls ReactiveAgentService to perform a diagnosis and remediation plan.

        Returns:
            (analysis_text, plan_text, execute_instruction)
            execute_instruction is the ---EXECUTE--- section, or None if unavailable.
        """
        try:
            from app.domain.agent.reactive_service import ReactiveAgentService

            reactive_service = ReactiveAgentService()
            async with async_session_factory() as session:
                analysis, plan, execute_instruction = await reactive_service.analyze(event, session)
            return analysis, plan, execute_instruction

        except Exception as exc:
            logger.warning(f"[EventProcessor] ReactiveAgentService unavailable, using fallback: {exc}")
            analysis = (
                f"Auto-detected {event.severity} event from {event.source_type}. "
                f"Title: {event.title}. Description: {event.description}."
            )
            plan = "Manual review required." if event.severity in ("high", "critical") else None
            return analysis, plan, None

    async def _execute_approved(self, event: Event) -> None:
        """Called by the approve endpoint for human-approved medium events."""
        async with async_session_factory() as session:
            repo = EventRepository(session)
            try:
                actions = await self._execute(event, event.agent_plan)
                await repo.update_analysis(event.id, analysis=event.agent_analysis or "", plan=event.agent_plan, actions=actions)
                await repo.update_status(event.id, "completed")
                await broadcast_sse({"event": "status_update", "data": {"id": str(event.id), "status": "completed"}})
                logger.info(f"[EventProcessor] Approved event {event.id} executed and completed.")
            except Exception as exc:
                logger.error(f"[EventProcessor] Approved execution failed for {event.id}: {exc}")
                await repo.update_status(event.id, "failed")
                await broadcast_sse({"event": "status_update", "data": {"id": str(event.id), "status": "failed"}})

    async def _execute(self, event: Event, plan: Optional[str], execute_instruction: Optional[str] = None) -> list:
        """
        Executes the remediation plan via ReactiveAgentService (Computer Use loop).
        Returns a list of action dicts performed.
        """
        actions = []
        if not plan:
            return actions
        try:
            from app.domain.agent.reactive_service import ReactiveAgentService

            reactive_service = ReactiveAgentService()
            async with async_session_factory() as session:
                actions = await reactive_service.execute_plan(
                    event, plan, session, execute_instruction=execute_instruction
                )
        except Exception as exc:
            logger.error(f"[EventProcessor] Execution failed for event {event.id}: {exc}")
            actions = [{"type": "error", "content": str(exc), "timestamp": datetime.utcnow().isoformat()}]
        return actions
