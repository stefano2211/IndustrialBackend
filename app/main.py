"""
FastAPI Application Entry Point.

Initializes the app, middleware, database, and LangGraph memory.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.persistence.db import init_db
from app.persistence.memoryAI.checkpointer import get_checkpointer, close_pool
from app.persistence.memoryAI.store import get_store

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init DB and LangGraph memory on startup, cleanup on shutdown."""
    await init_db()

    checkpointer = await get_checkpointer()
    store = await get_store()

    app.state.checkpointer = checkpointer
    app.state.store = store

    yield

    await close_pool()


app = FastAPI(
    title="IA Industrial - Document Analysis System (MinIO + Celery)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
