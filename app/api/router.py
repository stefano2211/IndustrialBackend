from fastapi import APIRouter
from app.api.endpoints import documents, chat, system, tools

api_router = APIRouter()

api_router.include_router(system.router, tags=["system"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
