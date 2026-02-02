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
    yield

app = FastAPI(
    title="Aura Research - Document Analysis System (MinIO + Celery)",
    lifespan=lifespan
)

app.include_router(api_router)
