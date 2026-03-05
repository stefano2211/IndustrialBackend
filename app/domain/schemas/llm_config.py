from typing import Optional
from sqlmodel import SQLModel, Field

class LLMConfigBase(SQLModel):
    model_config = {"protected_namespaces": ()}
    role: str = Field(primary_key=True, description="Role of the LLM (e.g., orchestrator, subagent)")
    provider: str = Field(description="LLM provider name (e.g., openai, anthropic, gemini)")
    model_name: str = Field(description="Specific model identifier")

class LLMConfig(LLMConfigBase, table=True):
    pass

class LLMConfigRead(LLMConfigBase):
    pass

class LLMConfigUpdate(SQLModel):
    model_config = {"protected_namespaces": ()}
    provider: Optional[str] = None
    model_name: Optional[str] = None
