import uuid
from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship


class Conversation(SQLModel, table=True):
    """Tracks chat threads per user."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    thread_id: str = Field(unique=True, index=True)
    title: str = Field(default="New Chat")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(SQLModel, table=True):
    """Persists each message in a conversation thread."""
    __tablename__ = "chatmessage"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: str = Field(index=True)
    role: str  # 'user' or 'assistant'
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationCreate(SQLModel):
    title: str = "New Chat"


class ConversationRead(SQLModel):
    id: uuid.UUID
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageRead(SQLModel):
    """Represents a single message."""
    role: str  # 'user' or 'assistant'
    content: str

