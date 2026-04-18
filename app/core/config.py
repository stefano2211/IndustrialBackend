from pydantic_settings import BaseSettings
from pathlib import Path
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

    
    # Edge / Local Model configuration
    vllm_base_url: str = "http://vllm-sistema1:8000/v1"
    vllm_orchestrator_url: str = "http://vllm-orchestrator:8000/v1"

    # OpenRouter configuration - REMOVED

    # Defaults
    default_llm_provider: str = "vllm"
    default_llm_model: Optional[str] = "aura_tenant_01-v2" # Expert LoRA tag in vLLM

    # Generalist Orchestrator (larger — for routing decisions)
    generalist_llm_model: str = "Qwen/Qwen3.5-9B"  # Director model

    # Sistema 1 — Fine-tuned models (9B base + their LoRA)
    system1_base_model: str = "Qwen/Qwen3.5-9B"        # Base backbone for ALL Sistema 1 subagents (hardcoded, upgraded 2B→9B)
    system1_historico_model: str = "aura_tenant_01-v2"  # Text LoRA tag in vLLM (historico subagent)
    system1_model: str = "aura_tenant_01-vl"             # VL LoRA tag in vLLM (VL + computer-use subagents)
    system1_enabled: bool = True                         # Toggle; set False if neither LoRA is deployed yet
    system1_force_base_model: bool = False              # True = usar modelo base sin intentar cargar LoRAs (para dev sin LoRAs)

    # Computer Use — Macrohard Digital Optimus Local
    computer_use_enabled: bool = True          # Feature flag global
    computer_use_demo_mode: bool = False       # True=mock screenshots, False=pantalla real
    computer_use_max_steps: int = 15           # Máximo de pasos por tarea (evita loops infinitos)
    computer_use_context_screenshots: int = 3  # Historial de screenshots pasados inyectados al VLM (UI-TARS style)

    # OmniParser V2 — Set-of-Marks grounding (Microsoft)
    # Activar cuando los pesos estén descargados en omniparser_model_dir
    omniparser_enabled: bool = False
    omniparser_model_dir: str = "/omniparser/weights"  # Ruta local de los pesos (icon_detect + icon_caption)

    # MLOps Architecture (Cloud Sync)
    mothership_api_url: str = "http://localhost:8001" # Default to local ApiLLMOps port
    mothership_api_key: str = "default-mothership-secret-key" # Must match ApiLLMOps MOTHERSHIP_API_KEY env
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