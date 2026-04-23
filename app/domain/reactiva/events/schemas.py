"""Pydantic API schemas for the reactive event layer."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventIngestRequest(BaseModel):
    """Payload sent by external sensors or webhooks."""
    source_type: str = Field(description="sensor | webhook | db_collector")
    severity: str = Field(description="low | medium | high | critical")
    title: str = Field(default="")
    description: str = Field(default="")
    raw_payload: Optional[Dict[str, Any]] = None
    tenant_id: str = Field(default="default")


class EventManualRequest(BaseModel):
    """Payload for operator-triggered manual events."""
    severity: str = Field(default="medium", description="low | medium | high | critical")
    title: str
    description: str
    raw_payload: Optional[Dict[str, Any]] = None


class EventApprovalRequest(BaseModel):
    """Payload for human-in-the-loop approval/rejection."""
    notes: Optional[str] = None


class EventResponse(BaseModel):
    """Public representation of an event."""
    id: uuid.UUID
    tenant_id: str
    source_type: str
    severity: str
    status: str
    title: str
    description: str
    raw_payload: Optional[Dict[str, Any]]
    agent_analysis: Optional[str]
    agent_plan: Optional[str]
    actions_taken: Optional[List[Any]]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    triggered_by_user_id: Optional[str]

    class Config:
        from_attributes = True


class EventListResponse(BaseModel):
    total: int
    items: List[EventResponse]


class SSEEventPayload(BaseModel):
    """Payload pushed to SSE clients on event state changes."""
    event: str = Field(description="SSE event name (new_event | status_update | analysis_ready)")
    data: EventResponse
