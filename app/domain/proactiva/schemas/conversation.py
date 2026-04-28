import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


class Conversation(SQLModel, table=True):
    """Tracks chat threads per user."""
    model_config = {"protected_namespaces": ()}
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    thread_id: str = Field(unique=True, index=True)
    title: str = Field(default="New Chat")
    is_archived: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ChatMessage(SQLModel, table=True):
    """Persists each message in a conversation thread."""
    model_config = {"protected_namespaces": ()}
    
    __tablename__ = "chatmessage"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: str = Field(index=True)
    role: str  # 'user' or 'assistant'
    content: str
    model_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ConversationCreate(SQLModel):
    title: str = "New Chat"


class ConversationRead(SQLModel):
    id: uuid.UUID
    thread_id: str
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class MessageRead(SQLModel):
    """Represents a single message."""
    role: str  # 'user' or 'assistant'
    content: str

