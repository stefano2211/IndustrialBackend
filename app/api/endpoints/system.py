from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from sqlmodel import select

from app.persistence.db import get_session
from app.domain.schemas.user import User
from app.domain.schemas.conversation import Conversation

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "aura-api"}

@router.get("/stats")
async def get_system_stats(session: AsyncSession = Depends(get_session)):
    # Fetch real counts from DB
    user_count_stmt = select(func.count()).select_from(User)
    conv_count_stmt = select(func.count()).select_from(Conversation)
    
    user_result = await session.execute(user_count_stmt)
    conv_result = await session.execute(conv_count_stmt)
    
    return {
        "active_users": user_result.scalar() or 0,
        "total_conversations": conv_result.scalar() or 0,
        "status": "nominal"
    }

    
