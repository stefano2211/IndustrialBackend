from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str 
    qdrant_port: int 
    qdrant_collection: str 
    embedding_model: str 
    
    # OpenRouter / LLM
    openrouter_api_key: str
    openrouter_model: str 
    
    # Celery
    celery_broker_url: str 
    celery_result_backend: str 
    
    # MinIO 1.0
    minio_endpoint: str 
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str 
    minio_secure: bool 
    
    # Postgres
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "aura_db"

    # Security
    secret_key: str
    access_token_expire_minutes: int = 30 # Default 30 mins

    # LLM Providers (Optional keys for other providers)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Defaults
    default_llm_provider: str = "openrouter"
    default_llm_model: Optional[str] = None # Will use openrouter_model if None
    
    # Orchestrator specific
    orchestrator_llm_provider: Optional[str] = None
    orchestrator_llm_model: Optional[str] = None
    
    # Subagent specific
    subagent_llm_provider: Optional[str] = None
    subagent_llm_model: Optional[str] = None

    # LLM Resilience
    llm_max_retries: int = 10
    llm_request_timeout: int = 120

    # Ingestion Pipeline — NER Performance
    ner_batch_size: int = 5          # Chunks grouped per LLM call (reduces total calls)
    ner_max_concurrency: int = 3     # Max parallel batch requests to the LLM
    ner_retry_max_attempts: int = 5  # Max retries on rate-limit (429) errors
    
    # Extractor specific (Google LangExtract works best with Gemini)
    extractor_llm_provider: str = "gemini"
    extractor_llm_model: str = "gemini-1.5-pro"
    
    model_config = {"env_file": ".env"}

settings = Settings()