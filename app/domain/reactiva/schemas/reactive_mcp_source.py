"""SQLModel schema for reactive MCP sources — system-scoped, not per-user."""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING

from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from app.domain.reactiva.schemas.reactive_tool_config import ReactiveToolConfig


class ReactiveMCPSourceBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    url: str = Field(...)
    type: str = Field(default="rest")  # rest, sse, stdio
    is_enabled: bool = Field(default=True)


class ReactiveMCPSource(ReactiveMCPSourceBase, table=True):
    """Reactive MCP sources are system-scoped (tenant_id) not user-scoped."""

    __tablename__ = "reactive_mcp_source"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    tools: List["ReactiveToolConfig"] = Relationship(
        back_populates="source", cascade_delete=True
    )


class ReactiveMCPSourceCreate(ReactiveMCPSourceBase):
    pass


class ReactiveMCPSourceRead(ReactiveMCPSourceBase):
    id: uuid.UUID
    tenant_id: str
    created_at: datetime


class ReactiveMCPSourceUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None
    is_enabled: Optional[bool] = None
