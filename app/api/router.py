from fastapi import APIRouter
from app.api.endpoints import documents, chat, system, tools, auth, users, conversations, knowledge, prompts, models, admin

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
api_router.include_router(system.router, tags=["system"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
api_router.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])

