from fastapi import FastAPI
from app.api.router import api_router
import os


UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from contextlib import asynccontextmanager
from app.persistence.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    
    # Initialize LangGraph Memory (Short-term & Long-term)
    from app.persistence.memoryAI.checkpointer import get_checkpointer, close_pool
    from app.persistence.memoryAI.store import get_store
    from app.domain.agent.workflow import reload_with_memory
    
    checkpointer = await get_checkpointer()
    store = await get_store()
    reload_with_memory(checkpointer, store)
    
    yield
    
    # Cleanup memory pools
    await close_pool()

app = FastAPI(
    title="Aura Research - Document Analysis System (MinIO + Celery)",
    lifespan=lifespan
)

app.include_router(api_router)
