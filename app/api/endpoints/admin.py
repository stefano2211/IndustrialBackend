"""Admin endpoints — user management and system analytics (admin only)."""

from typing import List
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.api import deps
from app.persistence.db import get_session
from app.domain.schemas.user import User, UserRead
from app.domain.schemas.conversation import Conversation, ChatMessage
from app.domain.schemas.model import Model
from app.domain.services.user_service import UserService
from app.core.config import settings

router = APIRouter()


def require_admin(user: User):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")


# ── Users ──────────────────────────────────────────


@router.get("/users", response_model=List[UserRead])
async def list_users(
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    service = UserService(session)
    return await service.list_users()


class RoleUpdate(BaseModel):
    is_superuser: bool


@router.put("/users/{user_id}/role", response_model=UserRead)
async def update_user_role(
    user_id: UUID,
    body: RoleUpdate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    service = UserService(session)
    user = await service.update_user_role(user_id, body.is_superuser)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    if str(user_id) == str(current_user.id):
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    service = UserService(session)
    deleted = await service.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


# ── Settings ─────────────────────────────────────────

from app.domain.schemas.settings import (
    SystemSettingsGeneralRead, 
    SystemSettingsGeneralUpdate,
    SystemSettingsDocumentsRead, 
    SystemSettingsDocumentsUpdate
)
from app.persistence.repositories.settings_repository import SettingsRepository

@router.get("/settings/general", response_model=SystemSettingsGeneralRead)
async def get_general_settings(
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    repo = SettingsRepository(session)
    return await repo.get_settings()


@router.put("/settings/general", response_model=SystemSettingsGeneralRead)
async def update_general_settings(
    body: SystemSettingsGeneralUpdate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    repo = SettingsRepository(session)
    return await repo.update_settings(body)


@router.get("/settings/documents", response_model=SystemSettingsDocumentsRead)
async def get_document_settings(
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    repo = SettingsRepository(session)
    return await repo.get_settings()


@router.put("/settings/documents", response_model=SystemSettingsDocumentsRead)
async def update_document_settings(
    body: SystemSettingsDocumentsUpdate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)
    repo = SettingsRepository(session)
    return await repo.update_settings(body)


# ── Analytics ──────────────────────────────────────


class DailyMessages(BaseModel):
    date: str
    count: int


class ModelUsageItem(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    rank: int
    model: str
    messages: int
    tokens: int
    percentage: float


class UserActivityItem(BaseModel):
    rank: int
    username: str
    email: str
    messages: int
    tokens: int


class AnalyticsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    total_messages: int
    total_tokens: int
    total_chats: int
    total_users: int
    daily_messages: List[DailyMessages]
    model_usage: List[ModelUsageItem]
    user_activity: List[UserActivityItem]


@router.get("/stats", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = 7,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
):
    require_admin(current_user)

    from datetime import datetime, timezone, timedelta
    from sqlalchemy import cast, Date, text

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

    # Total counts
    total_messages = (
        await session.execute(
            select(func.count(ChatMessage.id)).where(
                ChatMessage.role == "assistant",
                ChatMessage.created_at >= cutoff,
            )
        )
    ).scalar() or 0

    total_chats = (
        await session.execute(
            select(func.count(Conversation.id)).where(Conversation.created_at >= cutoff)
        )
    ).scalar() or 0

    total_users = (await session.execute(select(func.count(User.id)))).scalar() or 0

    # Estimate tokens (avg ~4 chars per token)
    total_content_len = (
        await session.execute(
            select(func.sum(func.length(ChatMessage.content))).where(
                ChatMessage.role == "assistant",
                ChatMessage.created_at >= cutoff,
            )
        )
    ).scalar() or 0
    total_tokens = total_content_len // 4

    # Daily messages (assistant only)
    daily_rows = (
        await session.execute(
            select(
                cast(ChatMessage.created_at, Date).label("date"),
                func.count(ChatMessage.id).label("count"),
            )
            .where(ChatMessage.role == "assistant", ChatMessage.created_at >= cutoff)
            .group_by(cast(ChatMessage.created_at, Date))
            .order_by(cast(ChatMessage.created_at, Date))
        )
    ).all()

    # Fill in missing days with 0
    daily_map = {str(row.date): row.count for row in daily_rows}
    daily_messages = []
    for i in range(days):
        d = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days - 1 - i)).date()
        daily_messages.append(DailyMessages(date=str(d), count=daily_map.get(str(d), 0)))

    # User activity: join conversations + messages
    user_rows = (
        await session.execute(
            select(
                User.username,
                User.email,
                func.count(ChatMessage.id).label("msg_count"),
                func.coalesce(func.sum(func.length(ChatMessage.content)), 0).label("content_len"),
            )
            .join(Conversation, Conversation.user_id == User.id)
            .join(ChatMessage, ChatMessage.thread_id == Conversation.thread_id)
            .where(ChatMessage.role == "assistant", ChatMessage.created_at >= cutoff)
            .group_by(User.id, User.username, User.email)
            .order_by(func.count(ChatMessage.id).desc())
        )
    ).all()

    user_activity = [
        UserActivityItem(
            rank=i + 1,
            username=row.username,
            email=row.email,
            messages=row.msg_count,
            tokens=row.content_len // 4,
        )
        for i, row in enumerate(user_rows)
    ]

    # Model usage: Group assistant messages by model_id
    model_usage_rows = (
        await session.execute(
            select(
                ChatMessage.model_id,
                func.count(ChatMessage.id).label("msg_count"),
                func.coalesce(func.sum(func.length(ChatMessage.content)), 0).label("content_len"),
            )
            .where(ChatMessage.role == "assistant", ChatMessage.created_at >= cutoff)
            .group_by(ChatMessage.model_id)
            .order_by(func.count(ChatMessage.id).desc())
        )
    ).all()

    model_usage = []
    for i, row in enumerate(model_usage_rows):
        m_id = row.model_id or "unknown"
        # Try to find a human-readable name for the model
        m_name = m_id
        if m_id != "unknown":
            db_m = await session.get(Model, m_id)
            if db_m:
                m_name = db_m.name
                
        model_usage.append(
            ModelUsageItem(
                rank=i + 1,
                model=m_name,
                messages=row.msg_count,
                tokens=row.content_len // 4,
                percentage=(row.msg_count / total_messages * 100) if total_messages > 0 else 0,
            )
        )

    # Fallback if no records have model_id yet but there are messages
    if not model_usage and total_messages > 0:
        model_name = settings.default_llm_model or "Unified-LLM"
        model_usage.append(
            ModelUsageItem(
                rank=1,
                model=model_name,
                messages=total_messages,
                tokens=total_tokens,
                percentage=100.0,
            )
        )

    return AnalyticsResponse(
        total_messages=total_messages,
        total_tokens=total_tokens,
        total_chats=total_chats,
        total_users=total_users,
        daily_messages=daily_messages,
        model_usage=model_usage,
        user_activity=user_activity,
    )
