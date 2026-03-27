from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str 
    qdrant_port: int 
    qdrant_collection: str = "documents"
    embedding_model: str = "nomic-embed-text"
    
    
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

    
    # Edge / Local Model configuration
    ollama_base_url: str = "http://ollama:11434"

    # OpenRouter configuration
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Defaults
    default_llm_provider: str = "ollama"
    default_llm_model: Optional[str] = "qwen3.5:9b" # Expert industrial model (fine-tuned via LLMOps)

    # Generalist Orchestrator (Magentic-One layer — also runs on Ollama)
    generalist_llm_model: str = "llama3.1:8b"  # General-purpose director model

    # MLOps Architecture (Cloud Sync)
    mothership_api_url: str = "http://localhost:8001" # Default to local ApiLLMOps port
    mothership_api_key: str = "12345678" # Default key for sync
    mlops_tenant_id: str = "aura_tenant_01" # Tenant identifier for this edge node
    edge_public_url: str = "http://localhost:8000" # Public URL of this edge node (for OTA webhook)


    # LLM Resilience
    llm_max_retries: int = 10
    llm_request_timeout: int = 120

    # Ingestion Pipeline — NER Performance
    ner_batch_size: int = 5          # Chunks grouped per LLM call (reduces total calls)
    ner_max_concurrency: int = 3     # Max parallel batch requests to the LLM
    ner_retry_max_attempts: int = 5  # Max retries on rate-limit (429) errors
    
    model_config = {"env_file": ".env"}

settings = Settings()