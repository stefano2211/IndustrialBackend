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


# Warmup no longer needed for vLLM resident model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init DB and LangGraph memory on startup, cleanup on shutdown."""
    await init_db()

    checkpointer = await get_checkpointer()
    store = await get_store()

    app.state.checkpointer = checkpointer
    app.state.store = store

    # Start the DB Collector scheduler (loads all enabled sources as cron jobs)

    # Start the DB Collector scheduler (loads all enabled sources as cron jobs)
    await collector_scheduler.start()

    yield

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

app.include_router(api_router)
