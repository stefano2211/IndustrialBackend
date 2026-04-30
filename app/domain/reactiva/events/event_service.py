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

from app.core.reactiva.event_queue import get_event_queue
from app.domain.reactiva.schemas.event import Event
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
                asyncio.create_task(self._process_safe(event, queue))
        except asyncio.CancelledError:
            logger.info("[EventProcessorService] Worker loop cancelled.")
            self._running = False

    async def _process_safe(self, event: Event, queue: asyncio.Queue) -> None:
        try:
            await self._processor.process(event)
        except Exception as exc:
            logger.error(f"[EventProcessorService] Unhandled error for event {event.id}: {exc}")
        finally:
            queue.task_done()

    async def execute_approved(self, event: Event) -> None:
        """Public wrapper to execute a human-approved medium event."""
        await self._processor._execute_approved(event)

