from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str 
    qdrant_port: int 
    qdrant_collection: str = "documents"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    sparse_embedding_model: str = "prithivida/Splade_PP_en_v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    
    
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

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
    ]

    # Security
    secret_key: str
    access_token_expire_minutes: int = 30 # Default 30 mins

    
    # Edge / Local Model configuration — Qwen3.5-2B (8GB VRAM local test)
    vllm_base_url: str = "http://vllm:8000/v1"

    # Defaults
    default_llm_provider: str = "vllm"
    default_llm_model: Optional[str] = "Qwen/Qwen3.5-2B"

    # Generalist Orchestrator
    generalist_llm_model: str = "Qwen/Qwen3.5-2B"

    # Sistema 1 — Using base model directly (no LoRA for local 8GB VRAM)
    system1_base_model: str = "Qwen/Qwen3.5-2B"
    system1_historico_model: str = "Qwen/Qwen3.5-2B"
    system1_model: str = "Qwen/Qwen3.5-2B"
    system1_enabled: bool = True
    system1_force_base_model: bool = True  # Force base model — no LoRA loading

    # Computer Use — disabled for local 8GB VRAM
    # proactive: enables sistema1-vl in chat; reactive: controlled by reactive_computer_use_enabled below
    computer_use_enabled: bool = False
    computer_use_demo_mode: bool = False
    computer_use_max_steps: int = 15
    computer_use_context_screenshots: int = 3

    # OmniParser V2 — disabled for local 8GB VRAM
    omniparser_enabled: bool = False
    omniparser_model_dir: str = "/omniparser/weights"

    # Playwright Hybrid Mode — when True, browser tasks use Playwright
    # (Accessibility Tree + semantic actions) instead of raw mss + xdotool.
    # Falls back to xdotool for non-browser / native desktop tasks.
    playwright_enabled: bool = False
    playwright_headless: bool = True  # False to see the browser via VNC (debugging)

    # MLOps Architecture (Cloud Sync)
    mothership_api_url: str = "http://localhost:8001"
    mothership_api_key: str = "default-mothership-secret-key"
    mlops_tenant_id: str = "aura_tenant_01"
    edge_public_url: str = "http://localhost:8000"


    # LLM Resilience
    llm_max_retries: int = 10
    llm_request_timeout: int = 120

    # Ingestion Pipeline — NER Performance
    ner_batch_size: int = 5          # Chunks grouped per LLM call (reduces total calls)
    ner_max_concurrency: int = 3     # Max parallel batch requests to the LLM
    ner_retry_max_attempts: int = 5  # Max retries on rate-limit (429) errors

    # ── Reactive Domain — isolated namespaces (same infra containers) ─────
    reactive_qdrant_collection: str = "reactive_documents"
    reactive_minio_bucket: str = "reactive-bucket"
    reactive_computer_use_enabled: bool = False  # Requires system1_enabled + computer_use infra (VL LoRA). Enables autonomous GUI actions for high/critical reactive events.
    
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()