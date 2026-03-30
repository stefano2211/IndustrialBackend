"""
FastAPI Application Entry Point.

Initializes the app, middleware, database, and LangGraph memory.
"""

import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Suppress Pydantic UserWarning about typing.NotRequired from external libraries
warnings.filterwarnings("ignore", category=UserWarning, message=".*typing.NotRequired is not a Python type.*")

from app.api.router import api_router
from app.persistence.db import init_db
from app.persistence.memoryAI.checkpointer import get_checkpointer, close_pool
from app.persistence.memoryAI.store import get_store
from app.core.llm import LLMFactory, LLMProvider
from app.domain.db_collector.scheduler import collector_scheduler

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def _warmup_ollama():
    """Pre-load the LLM model into VRAM during app startup.
    This runs in the background so it doesn't block FastAPI startup.
    """
    try:
        logger.info("🔥 [Warmup] Pre-loading LLM model into VRAM...")
        import asyncio
        
        async def _do_warmup():
            try:
                llm = await LLMFactory.get_llm(provider=LLMProvider.OLLAMA)
                await llm.ainvoke("warmup")
                logger.info("✅ [Warmup] LLM model loaded and ready.")
            except Exception as e:
                logger.warning(f"⚠️ [Warmup] Failed background LLM load (non-fatal): {e}")

        asyncio.create_task(_do_warmup())
    except Exception as e:
        logger.warning(f"⚠️ [Warmup] Could not initiate LLM pre-load: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init DB and LangGraph memory on startup, cleanup on shutdown."""
    await init_db()

    checkpointer = await get_checkpointer()
    store = await get_store()

    app.state.checkpointer = checkpointer
    app.state.store = store

    # await _warmup_ollama()

    # Start the DB Collector scheduler (loads all enabled sources as cron jobs)
    await collector_scheduler.start()

    yield

    await collector_scheduler.shutdown()
    await close_pool()


app = FastAPI(
    title="IA Industrial - Document Analysis System (Edge AI)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
