"""SQLModel schema for reactive tool configurations — linked to ReactiveMCPSource."""

import uuid
from typing import Optional, Dict, Any, TYPE_CHECKING

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON

if TYPE_CHECKING:
    from app.domain.schemas.reactive_mcp_source import ReactiveMCPSource


class ReactiveToolConfigBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: str
    api_url: str
    method: str = "GET"
    headers: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    auth_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    system_prompt: str = ""
    parameter_schema: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    source_id: Optional[uuid.UUID] = Field(
        default=None, foreign_key="reactive_mcp_source.id", index=True
    )


class ReactiveToolConfig(ReactiveToolConfigBase, table=True):
    __tablename__ = "reactive_tool_config"

    id: Optional[int] = Field(default=None, primary_key=True)

    source: Optional["ReactiveMCPSource"] = Relationship(back_populates="tools")


class ReactiveToolConfigCreate(ReactiveToolConfigBase):
    pass


class ReactiveToolConfigUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    api_url: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    parameter_schema: Optional[Dict[str, Any]] = None


class ReactiveToolConfigRead(ReactiveToolConfigBase):
    id: int
