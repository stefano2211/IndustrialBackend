import uuid
from typing import Annotated, Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from app.api import deps
from app.persistence.db import get_session
from app.domain.schemas.user import User
from app.domain.schemas.conversation import (
    Conversation,
    ConversationCreate,
    ConversationRead,
    ChatMessage,
    MessageRead,
)
from loguru import logger

router = APIRouter()


@router.post("", response_model=ConversationRead)
async def create_conversation(
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(deps.get_current_user)],
    body: ConversationCreate,
) -> Any:
    """Create a new conversation thread."""
    conversation = Conversation(
        user_id=current_user.id,
        thread_id=str(uuid.uuid4()),
        title=body.title,
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.get("", response_model=List[ConversationRead])
async def list_conversations(
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """List all conversations for the current user, newest first."""
    statement = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    result = await session.execute(statement)
    conversations = list(result.scalars().all())
    return conversations


@router.get("/{thread_id}/messages", response_model=List[MessageRead])
async def get_conversation_messages(
    thread_id: str,
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """
    Load messages for a conversation from the database.
    Returns messages ordered by creation time (oldest first).
    """
    # Verify the conversation belongs to this user
    conv_stmt = select(Conversation).where(
        Conversation.thread_id == thread_id,
        Conversation.user_id == current_user.id,
    )
    result = await session.execute(conv_stmt)
    conversation = result.scalars().first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Query messages for this thread
    msg_stmt = (
        select(ChatMessage)
        .where(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.asc())
    )
    res_msg = await session.execute(msg_stmt)
    db_messages = list(res_msg.scalars().all())

    return [MessageRead(role=m.role, content=m.content) for m in db_messages]


@router.delete("/{thread_id}")
async def delete_conversation(
    thread_id: str,
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """Delete a conversation and its messages."""
    statement = select(Conversation).where(
        Conversation.thread_id == thread_id,
        Conversation.user_id == current_user.id,
    )
    result = await session.execute(statement)
    conversation = result.scalars().first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete messages first
    msg_stmt = select(ChatMessage).where(ChatMessage.thread_id == thread_id)
    res_msg = await session.execute(msg_stmt)
    messages = list(res_msg.scalars().all())
    for msg in messages:
        await session.delete(msg)

    await session.delete(conversation)
    await session.commit()
    return {"detail": "Conversation deleted"}

