"""
FastAPI Application Entry Point.

Initializes the app, middleware, database, and LangGraph memory.
"""

import os
import asyncio
import warnings
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.middleware import GlobalExceptionHandler

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

LORA_DIR = Path("/loras")


async def _auto_register_loras():
    """Discover LoRA adapters on disk and register them in vLLM at startup.

    vLLM loses dynamically-loaded LoRAs on restart. This function scans
    /loras/ for directories containing adapter_config.json and loads them
    via the vLLM runtime LoRA API, ensuring they are always available.
    """
    from app.core.config import settings

    if not LORA_DIR.exists():
        logger.debug("[LoRA AutoRegister] /loras/ directory not found — skipping.")
        return

    adapters = [
        d for d in LORA_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    ]
    if not adapters:
        logger.info("[LoRA AutoRegister] No adapters found in /loras/.")
        return

    vllm_base = settings.vllm_base_url.rstrip("/")       # http://vllm:8000/v1
    vllm_host = vllm_base.removesuffix("/v1")             # http://vllm:8000

    # vLLM is guaranteed healthy by Docker healthcheck (service_healthy).
    # Quick sanity check before proceeding.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{vllm_host}/v1/models")
            if r.status_code != 200:
                logger.warning(f"[LoRA AutoRegister] vLLM responded {r.status_code} — skipping.")
                return
    except Exception as e:
        logger.warning(f"[LoRA AutoRegister] vLLM not reachable: {e} — skipping.")
        return

    logger.info("[LoRA AutoRegister] vLLM is ready. Registering adapters...")

    for adapter_dir in adapters:
        lora_name = adapter_dir.name
        lora_path = f"/loras/{lora_name}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{vllm_host}/v1/load_lora_adapter",
                    json={
                        "lora_name": lora_name,
                        "lora_path": lora_path,
                    },
                )
                if resp.status_code == 200:
                    logger.success(f"[LoRA AutoRegister] ✓ Loaded '{lora_name}' into vLLM.")
                else:
                    logger.warning(f"[LoRA AutoRegister] vLLM responded {resp.status_code} for '{lora_name}': {resp.text}")
        except Exception as e:
            logger.error(f"[LoRA AutoRegister] Failed to load '{lora_name}': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init DB and LangGraph memory on startup, cleanup on shutdown."""
    await init_db()

    checkpointer = await get_checkpointer()
    store = await get_store()

    app.state.checkpointer = checkpointer
    app.state.store = store

    # Auto-register LoRA adapters into vLLM (guaranteed healthy by Docker healthcheck)
    await _auto_register_loras()

    # Start the DB Collector scheduler (loads all enabled sources as cron jobs)
    await collector_scheduler.start()

    # Start the reactive event processor worker loop
    event_service = EventProcessorService()
    event_worker_task = asyncio.create_task(event_service.run())
    app.state.event_service = event_service

    # Auto-download OmniParser V2 weights if not present (non-blocking background task)
    if _settings.omniparser_enabled:
        asyncio.create_task(
            get_omniparser().ensure_weights(_settings.omniparser_model_dir)
        )

    yield

    event_worker_task.cancel()
    await collector_scheduler.shutdown()
    await close_pool()


from app.core.config import settings

app = FastAPI(
    title="IA Industrial - Document Analysis System (Edge AI)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GlobalExceptionHandler)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Kubernetes."""
    return {"status": "healthy", "service": "aura-backend"}
