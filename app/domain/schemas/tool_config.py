from sqlmodel import SQLModel, Field
from typing import Optional, Dict, Any
from sqlalchemy import Column, JSON

class ToolConfigBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: str
    api_url: str
    method: str = "GET"
    headers: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    auth_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    system_prompt: str
    parameter_schema: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

class ToolConfig(ToolConfigBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class ToolConfigCreate(ToolConfigBase):
    pass

class ToolConfigUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    api_url: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    parameter_schema: Optional[Dict[str, Any]] = None

class ToolConfigRead(ToolConfigBase):
    id: int
