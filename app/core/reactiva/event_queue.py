"""
Shared asyncio Queue for the reactive event layer.

The queue acts as the bridge between event producers (webhook ingestion,
DB Collector anomaly detection, manual triggers) and the event processor
worker that runs in the FastAPI lifespan.
"""

import asyncio
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from app.domain.schemas.event import Event

_event_queue: Optional[asyncio.Queue] = None
_sse_subscribers: List[asyncio.Queue] = []


def get_event_queue() -> asyncio.Queue:
    """Return (and lazily initialize) the singleton event queue."""
    global _event_queue
    if _event_queue is None:
        _event_queue = asyncio.Queue(maxsize=500)
    return _event_queue


def get_sse_subscribers() -> List[asyncio.Queue]:
    """Return the mutable list of active SSE subscriber queues."""
    return _sse_subscribers


async def broadcast_sse(payload: dict) -> None:
    """Push a payload to all connected SSE clients."""
    dead: List[asyncio.Queue] = []
    for sub_queue in _sse_subscribers:
        try:
            sub_queue.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(sub_queue)
    for d in dead:
        try:
            _sse_subscribers.remove(d)
        except ValueError:
            pass  # already removed by another concurrent broadcast
