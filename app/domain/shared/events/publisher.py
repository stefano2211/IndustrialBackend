"""
EventPublisher — Shared utility to emit reactive events.

Decouples proactive domains (e.g. DB Collector) from the reactive
EventProcessorService by extracting the enqueue logic into a shared
publisher that any domain can use without importing reactiva internals.
"""

from typing import Optional
from loguru import logger

from app.core.reactiva.event_queue import get_event_queue, broadcast_sse
from app.domain.reactiva.schemas.event import Event
from app.persistence.db import async_session_factory
from app.persistence.reactiva.repositories.event_repository import EventRepository


class EventPublisher:
    """Publishes reactive events to the queue and database."""

    async def publish(
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

        await broadcast_sse(
            {
                "event": "new_event",
                "data": {
                    "id": str(event.id),
                    "severity": event.severity,
                    "status": "pending",
                    "title": event.title,
                },
            }
        )
        logger.info(f"[EventPublisher] Published event {event.id} from {source_type}")
        return event
