"""SQLModel schema for the reactive Event table."""

import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON


class Event(SQLModel, table=True):
    """Represents a reactive event ingested by the system."""

    __tablename__ = "event"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    tenant_id: str = Field(default="default", index=True)

    source_type: str = Field(
        description="sensor | db_collector | manual | webhook",
        index=True,
    )
    severity: str = Field(
        description="low | medium | high | critical",
        index=True,
    )
    status: str = Field(
        default="pending",
        description="pending | analyzing | awaiting_approval | executing | completed | failed",
        index=True,
    )
    title: str = Field(default="")
    description: str = Field(default="")

    raw_payload: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Raw data from the event source",
    )
    agent_analysis: Optional[str] = Field(
        default=None,
        description="Fast System-1 analysis text",
    )
    agent_reasoning: Optional[str] = Field(
        default=None,
        description="Model reasoning/thinking trace (from <think> blocks)",
    )
    agent_plan: Optional[str] = Field(
        default=None,
        description="Detailed orchestrator plan",
    )
    actions_taken: Optional[list] = Field(
        default=None,
        sa_column=Column(JSON),
        description="List of actions executed by the agent",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    resolved_at: Optional[datetime] = Field(default=None)

    triggered_by_user_id: Optional[uuid.UUID] = Field(
        default=None,
        foreign_key="user.id",
        description="User UUID who created a manual event",
    )
    approved_by_user_id: Optional[uuid.UUID] = Field(
        default=None,
        foreign_key="user.id",
        description="User UUID who approved the event",
    )
    rejected_by_user_id: Optional[uuid.UUID] = Field(
        default=None,
        foreign_key="user.id",
        description="User UUID who rejected the event",
    )
