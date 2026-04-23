"""
Reactive Events API
===================
Endpoints for the reactive event dashboard.

Routes:
  POST /events/ingest          ← x-api-key  (external sensors / webhooks)
  POST /events/manual          ← JWT         (operator manual trigger)
  GET  /events                 ← JWT         (list with filters)
  GET  /events/stream          ← JWT         (SSE real-time feed)
  GET  /events/{id}            ← JWT         (event detail)
  POST /events/{id}/approve    ← JWT         (human-in-the-loop approval)
  POST /events/{id}/reject     ← JWT         (human-in-the-loop rejection)
"""

import asyncio
import json
import uuid
from typing import Annotated, Optional, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.db import get_session
from app.api.deps import get_current_user
from app.core.config import settings
from app.core.reactiva.event_queue import get_sse_subscribers
from app.domain.schemas.event import Event
from app.domain.schemas.user import User
from app.domain.reactiva.events.schemas import (
    EventIngestRequest,
    EventManualRequest,
    EventApprovalRequest,
    EventResponse,
    EventListResponse,
)
from app.domain.reactiva.events.event_service import EventProcessorService
from app.persistence.reactiva.repositories.event_repository import EventRepository

router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _get_user_from_token_str(token_str: str, session: AsyncSession) -> User:
    """Shared helper: decode JWT and return User."""
    from jose import jwt, JWTError
    from app.core.security import ALGORITHM
    from app.domain.schemas.token import TokenPayload
    from app.domain.proactiva.services.user_service import UserService
    try:
        payload = jwt.decode(token_str, settings.secret_key, algorithms=[ALGORITHM])
        token_data = TokenPayload(**payload)
    except (JWTError, Exception):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_service = UserService(session)
    user = await user_service.get_by_id(token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def get_current_user_flexible(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    token: Optional[str] = None,
) -> User:
    """Auth dependency that accepts Bearer header OR ?token= query param (for SSE)."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return await _get_user_from_token_str(auth_header[7:], session)
    if token:
        return await _get_user_from_token_str(token, session)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


def _verify_api_key(key: Optional[str] = Depends(_api_key_header)) -> str:
    """Validate X-API-Key for machine-to-machine ingestion."""
    if not key or key != settings.mothership_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )
    return key


def _get_event_service() -> EventProcessorService:
    return EventProcessorService()


# ── Ingest (machine-to-machine) ───────────────────────────────────────────────

@router.post("/ingest", response_model=EventResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_event(
    payload: EventIngestRequest,
    _key: str = Depends(_verify_api_key),
):
    """Ingest an event from an external sensor or automated system."""
    svc = _get_event_service()
    event = await svc.enqueue_event(
        source_type=payload.source_type,
        severity=payload.severity,
        title=payload.title,
        description=payload.description,
        raw_payload=payload.raw_payload,
        tenant_id=payload.tenant_id,
    )
    return EventResponse.model_validate(event)


# ── Manual trigger (operator) ─────────────────────────────────────────────────

@router.post("/manual", response_model=EventResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_manual_event(
    payload: EventManualRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Operator-triggered manual event."""
    svc = _get_event_service()
    event = await svc.enqueue_event(
        source_type="manual",
        severity=payload.severity,
        title=payload.title,
        description=payload.description,
        raw_payload=payload.raw_payload,
        triggered_by_user_id=str(current_user.id),
    )
    return EventResponse.model_validate(event)


# ── List events ───────────────────────────────────────────────────────────────

@router.get("", response_model=EventListResponse)
async def list_events(
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_user)],
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    source_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return a paginated list of events with optional filters."""
    repo = EventRepository(session)
    items, total = await repo.list_all(
        severity=severity,
        status=status,
        source_type=source_type,
        limit=limit,
        offset=offset,
    )
    return EventListResponse(
        total=total,
        items=[EventResponse.model_validate(e) for e in items],
    )


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get("/stream")
async def events_stream(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user_flexible)],
):
    """
    Server-Sent Events stream for real-time dashboard updates.
    Each message is a JSON-encoded SSEEventPayload.
    """
    sub_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    subscribers = get_sse_subscribers()
    subscribers.append(sub_queue)

    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield "data: {\"event\": \"connected\"}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(sub_queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            if sub_queue in subscribers:
                subscribers.remove(sub_queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Event detail ──────────────────────────────────────────────────────────────

@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get a single event by ID."""
    repo = EventRepository(session)
    event = await repo.get_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return EventResponse.model_validate(event)


# ── Human-in-the-loop: approve ────────────────────────────────────────────────

@router.post("/{event_id}/approve", response_model=EventResponse)
async def approve_event(
    event_id: uuid.UUID,
    payload: EventApprovalRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Approve a medium-severity event that is awaiting human approval.
    Triggers the execution phase via the event queue.
    """
    repo = EventRepository(session)
    event = await repo.get_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.status != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Event is not awaiting approval (current status: {event.status})",
        )

    if payload.notes:
        notes_text = f"\n[Operator notes]: {payload.notes}"
        event.description += notes_text

    event.status = "executing"
    event = await repo.save(event)

    svc = _get_event_service()
    asyncio.create_task(svc.execute_approved(event))

    return EventResponse.model_validate(event)


# ── Human-in-the-loop: reject ─────────────────────────────────────────────────

@router.post("/{event_id}/reject", response_model=EventResponse)
async def reject_event(
    event_id: uuid.UUID,
    payload: EventApprovalRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Reject a medium-severity event waiting for approval."""
    repo = EventRepository(session)
    event = await repo.get_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.status != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Event is not awaiting approval (current status: {event.status})",
        )

    if payload.notes:
        event.description += f"\n[Rejection reason]: {payload.notes}"

    event = await repo.update_status(event.id, "failed")
    return EventResponse.model_validate(event)
