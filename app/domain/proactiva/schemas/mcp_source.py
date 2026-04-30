from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
import uuid
from datetime import datetime, timezone

if TYPE_CHECKING:
    from app.domain.proactiva.schemas.tool_config import ToolConfig

class MCPSourceBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    url: str = Field(...)
    type: str = Field(default="rest") # rest, sse, stdio
    is_enabled: bool = Field(default=True)

class MCPSource(MCPSourceBase, table=True):
    __tablename__ = "mcp_source"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Relationship to tools
    tools: List["ToolConfig"] = Relationship(back_populates="source", cascade_delete=True)

class MCPSourceCreate(MCPSourceBase):
    pass

class MCPSourceRead(MCPSourceBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime

class MCPSourceUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None
    is_enabled: Optional[bool] = None
