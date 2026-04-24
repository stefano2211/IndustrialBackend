"""
EventProcessorService
=====================
Background worker that consumes events from the asyncio Queue and
dispatches them to the EventProcessor. Runs for the entire application
lifespan (started in main.py).
"""

import asyncio
from typing import Optional

from loguru import logger

from app.core.reactiva.event_queue import get_event_queue, broadcast_sse
from app.domain.schemas.event import Event
from app.persistence.db import async_session_factory
from app.persistence.reactiva.repositories.event_repository import EventRepository
from app.domain.reactiva.events.processor import EventProcessor


class EventProcessorService:
    """Manages the event queue worker loop."""

    def __init__(self):
        self._processor = EventProcessor()
        self._running = False

    async def run(self) -> None:
        """Infinite consumer loop. Cancelled on app shutdown."""
        self._running = True
        queue = get_event_queue()
        logger.info("[EventProcessorService] Worker loop started.")
        try:
            while True:
                event: Event = await queue.get()
                logger.info(f"[EventProcessorService] Dequeued event {event.id} ({event.severity})")
                asyncio.create_task(self._process_safe(event))
                queue.task_done()
        except asyncio.CancelledError:
            logger.info("[EventProcessorService] Worker loop cancelled.")
            self._running = False

    async def _process_safe(self, event: Event) -> None:
        try:
            await self._processor.process(event)
        except Exception as exc:
            logger.error(f"[EventProcessorService] Unhandled error for event {event.id}: {exc}")

    async def execute_approved(self, event: Event) -> None:
        """Public wrapper to execute a human-approved medium event."""
        await self._processor._execute_approved(event)

    async def enqueue_event(
        self,
        source_type: str,
        severity: str,
        title: str,
        description: str,
        raw_payload: Optional[dict] = None,
        tenant_id: str = "default",
        triggered_by_user_id: Optional[str] = None,
    ) -> Event:
        """
        Persist the event to DB, push to queue, and broadcast SSE.
        Returns the created Event.
        """
        event = Event(
            source_type=source_type,
            severity=severity,
            title=title,
            description=description,
            raw_payload=raw_payload,
            tenant_id=tenant_id,
            triggered_by_user_id=triggered_by_user_id,
            status="pending",
        )

        async with async_session_factory() as session:
            repo = EventRepository(session)
            event = await repo.create(event)

        queue = get_event_queue()
        await queue.put(event)

        await broadcast_sse({"event": "new_event", "data": {"id": str(event.id), "severity": event.severity, "status": "pending", "title": event.title}})
        logger.info(f"[EventProcessorService] Enqueued event {event.id} from {source_type}")
        return event
