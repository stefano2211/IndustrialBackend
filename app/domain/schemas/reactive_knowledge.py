"""SQLModel schemas for reactive Knowledge Bases — system-scoped (tenant, not per-user).

Reactive KBs store emergency SOPs, maintenance manuals, incident response protocols,
and any reference documents the reactive agent needs to consult during event processing.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlmodel import SQLModel, Field, Relationship


class ReactiveKnowledgeBase(SQLModel, table=True):
    """System-scoped knowledge base for the reactive domain."""

    __tablename__ = "reactive_knowledge_base"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    documents: List["ReactiveKnowledgeDocument"] = Relationship(
        back_populates="knowledge_base", cascade_delete=True
    )


class ReactiveKnowledgeDocument(SQLModel, table=True):
    """A document belonging to a reactive knowledge base."""

    __tablename__ = "reactive_knowledge_document"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    knowledge_base_id: uuid.UUID = Field(
        foreign_key="reactive_knowledge_base.id", index=True
    )
    filename: str
    file_id: str = Field(index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    knowledge_base: ReactiveKnowledgeBase = Relationship(back_populates="documents")


# --- DTOs for API ---

class ReactiveKnowledgeBaseCreate(SQLModel):
    name: str
    description: Optional[str] = None


class ReactiveKnowledgeBaseUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ReactiveKnowledgeDocumentRead(SQLModel):
    id: uuid.UUID
    filename: str
    file_id: str
    created_at: datetime


class ReactiveKnowledgeBaseRead(SQLModel):
    id: uuid.UUID
    tenant_id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class ReactiveKnowledgeBaseDetailRead(ReactiveKnowledgeBaseRead):
    documents: List[ReactiveKnowledgeDocumentRead] = []
