from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, JSON

class ModelBase(SQLModel):
    model_config = {"protected_namespaces": ()}
    id: Optional[str] = Field(default=None, primary_key=True, description="Slug-based identifier for the model. Generated if not provided.")
    name: str = Field(index=True, description="Display name of the model")
    base_model_id: str = Field(description="Underlying provider and model name (e.g., openai:gpt-4o)")
    description: Optional[str] = Field(default=None, description="Short description of the model")
    tags: List[str] = Field(default=[], sa_type=JSON, description="List of tags")
    
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt for the model")
    
    # Model parameters
    params: Dict[str, Any] = Field(default={}, sa_type=JSON, description="Model parameters like temperature, max_tokens, etc.")
    
    # Associations
    knowledge_ids: List[int] = Field(default=[], sa_type=JSON, description="IDs of associated knowledge bases")
    tool_ids: List[int] = Field(default=[], sa_type=JSON, description="IDs of associated tools")
    skill_ids: List[int] = Field(default=[], sa_type=JSON, description="IDs of associated skills")
    
    # UI/Runtime Capabilities
    capabilities: Dict[str, bool] = Field(default={}, sa_type=JSON, description="Enabled capabilities (vision, file_upload, etc.)")
    default_features: Dict[str, bool] = Field(default={}, sa_type=JSON, description="Default features (web_search, etc.)")
    builtin_tools: Dict[str, bool] = Field(default={}, sa_type=JSON, description="Enabled built-in tools")
    
    tts_voice: Optional[str] = Field(default=None, description="Selected TTS voice")
    enabled: bool = Field(default=True, description="Whether the model is active and can be used")

class Model(ModelBase, table=True):
    pass

class ModelCreate(ModelBase):
    pass

class ModelRead(ModelBase):
    pass

class ModelUpdate(SQLModel):
    name: Optional[str] = None
    base_model_id: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    knowledge_ids: Optional[List[int]] = None
    tool_ids: Optional[List[int]] = None
    skill_ids: Optional[List[int]] = None
    capabilities: Optional[Dict[str, bool]] = None
    default_features: Optional[Dict[str, bool]] = None
    builtin_tools: Optional[Dict[str, bool]] = None
    tts_voice: Optional[str] = None
    enabled: Optional[bool] = None
