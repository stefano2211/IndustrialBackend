from sqlmodel import SQLModel, Field
from typing import Optional, Dict, Any
from sqlalchemy import Column, JSON

class ToolConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str
    api_url: str
    method: str = "GET"
    headers: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    auth_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    system_prompt: str
    parameter_schema: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
