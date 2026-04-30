from typing import Optional
from sqlmodel import SQLModel, Field

class SystemSettings(SQLModel, table=True):
    """
    Global application settings persistance.
    Only one row is expected in this table, mapped to id=1.
    """
    __tablename__ = "systemsetting"
    
    id: int = Field(default=1, primary_key=True)
    
    # Auth Settings
    auth_default_user_role: str = Field(default="pending")
    auth_enable_sign_ups: bool = Field(default=True)
    
    # Feature Settings
    feature_enable_api_keys: bool = Field(default=True)
    feature_jwt_expiration: str = Field(default="4w")
    feature_enable_community_sharing: bool = Field(default=True)
    
    # Documents Settings
    document_chunk_size: int = Field(default=1000)
    document_chunk_overlap: int = Field(default=100)
    
    # Retrieval Settings
    retrieval_search_results: int = Field(default=5)

    # Provider Settings (vLLM)
    vllm_enabled: bool = Field(default=True)
    vllm_base_url: str = Field(default="http://vllm:8000/v1")

    # Provider Settings (OpenRouter)
    openrouter_enabled: bool = Field(default=False)
    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")


# --- API Models for General Settings ---
class SystemSettingsGeneralRead(SQLModel):
    auth_default_user_role: str
    auth_enable_sign_ups: bool
    feature_enable_api_keys: bool
    feature_jwt_expiration: str
    feature_enable_community_sharing: bool
    
    # Providers
    vllm_enabled: bool
    vllm_base_url: str
    openrouter_enabled: bool
    openrouter_api_key: Optional[str]
    openrouter_base_url: str

class SystemSettingsGeneralUpdate(SQLModel):
    auth_default_user_role: Optional[str] = None
    auth_enable_sign_ups: Optional[bool] = None
    feature_enable_api_keys: Optional[bool] = None
    feature_jwt_expiration: Optional[str] = None
    feature_enable_community_sharing: Optional[bool] = None
    
    # Providers
    vllm_enabled: Optional[bool] = None
    vllm_base_url: Optional[str] = None
    openrouter_enabled: Optional[bool] = None
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: Optional[str] = None


# --- API Models for Document Settings ---
class SystemSettingsDocumentsRead(SQLModel):
    document_chunk_size: int
    document_chunk_overlap: int
    retrieval_search_results: int

class SystemSettingsDocumentsUpdate(SQLModel):
    document_chunk_size: Optional[int] = None
    document_chunk_overlap: Optional[int] = None
    retrieval_search_results: Optional[int] = None
