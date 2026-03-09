import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel

class PromptBase(SQLModel):
    title: str
    description: Optional[str] = None
    query: str
    icon: Optional[str] = None
    is_enabled: bool = True

class Prompt(PromptBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PromptCreate(PromptBase):
    pass

class PromptRead(PromptBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

class PromptUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    query: Optional[str] = None
    icon: Optional[str] = None
    is_enabled: Optional[bool] = None
