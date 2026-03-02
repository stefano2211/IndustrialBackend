import uuid
from typing import Annotated, Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.api import deps
from app.core.database import get_session
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
def create_conversation(
    *,
    session: Annotated[Session, Depends(get_session)],
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
    session.commit()
    session.refresh(conversation)
    return conversation


@router.get("", response_model=List[ConversationRead])
def list_conversations(
    *,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """List all conversations for the current user, newest first."""
    statement = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = session.exec(statement).all()
    return conversations


@router.get("/{thread_id}/messages", response_model=List[MessageRead])
def get_conversation_messages(
    thread_id: str,
    *,
    session: Annotated[Session, Depends(get_session)],
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
    conversation = session.exec(conv_stmt).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Query messages for this thread
    msg_stmt = (
        select(ChatMessage)
        .where(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.asc())
    )
    db_messages = session.exec(msg_stmt).all()

    return [MessageRead(role=m.role, content=m.content) for m in db_messages]


@router.delete("/{thread_id}")
def delete_conversation(
    thread_id: str,
    *,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """Delete a conversation and its messages."""
    statement = select(Conversation).where(
        Conversation.thread_id == thread_id,
        Conversation.user_id == current_user.id,
    )
    conversation = session.exec(statement).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete messages first
    msg_stmt = select(ChatMessage).where(ChatMessage.thread_id == thread_id)
    messages = session.exec(msg_stmt).all()
    for msg in messages:
        session.delete(msg)

    session.delete(conversation)
    session.commit()
    return {"detail": "Conversation deleted"}

