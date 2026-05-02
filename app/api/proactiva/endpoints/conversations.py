"""Conversation endpoints — thin HTTP handlers delegating to ConversationService."""

from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.persistence.db import get_session
from app.domain.shared.schemas.user import User
from app.domain.proactiva.schemas.conversation import (
    ConversationCreate,
    ConversationRead,
    MessageRead,
)
from app.domain.proactiva.services.conversation_service import ConversationService
from app.domain.exceptions import NotFoundError

router = APIRouter()


async def get_conversation_service(
    session: AsyncSession = Depends(get_session),
) -> ConversationService:
    return ConversationService(session)


@router.post("", response_model=ConversationRead)
async def create_conversation(
    body: ConversationCreate,
    current_user: User = Depends(deps.get_current_user),
    service: ConversationService = Depends(get_conversation_service),
) -> Any:
    return await service.create_conversation(user_id=current_user.id, data=body)


@router.get("", response_model=List[ConversationRead])
async def list_conversations(
    include_archived: bool = False,
    current_user: User = Depends(deps.get_current_user),
    service: ConversationService = Depends(get_conversation_service),
) -> Any:
    return await service.list_conversations(user_id=current_user.id, include_archived=include_archived)


@router.get("/{thread_id}/messages", response_model=List[MessageRead])
async def get_conversation_messages(
    thread_id: str,
    current_user: User = Depends(deps.get_current_user),
    service: ConversationService = Depends(get_conversation_service),
) -> Any:
    try:
        messages = await service.get_messages(
            thread_id=thread_id, user_id=current_user.id
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [MessageRead(role=m.role, content=m.content, reasoning_content=getattr(m, 'reasoning_content', None)) for m in messages]


@router.delete("/{thread_id}")
async def delete_conversation(
    thread_id: str,
    current_user: User = Depends(deps.get_current_user),
    service: ConversationService = Depends(get_conversation_service),
) -> Any:
    try:
        await service.delete_conversation(
            thread_id=thread_id, user_id=current_user.id
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"detail": "Conversation deleted"}


@router.patch("/{thread_id}/archive", response_model=ConversationRead)
async def archive_conversation(
    thread_id: str,
    archive: bool = True,
    current_user: User = Depends(deps.get_current_user),
    service: ConversationService = Depends(get_conversation_service),
) -> Any:
    try:
        return await service.archive_conversation(
            thread_id=thread_id, user_id=current_user.id, archive=archive
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
