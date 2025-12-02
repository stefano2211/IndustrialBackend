from pydantic import BaseModel
from typing import Optional, Dict, Any

class ToolConfigBase(BaseModel):
    name: str
    description: str
    api_url: str
    method: str = "GET"
    headers: Optional[Dict[str, Any]] = {}
    auth_config: Optional[Dict[str, Any]] = {}
    system_prompt: str
    parameter_schema: Optional[Dict[str, Any]] = {}

class ToolConfigCreate(ToolConfigBase):
    pass

class ToolConfigUpdate(BaseModel):
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
