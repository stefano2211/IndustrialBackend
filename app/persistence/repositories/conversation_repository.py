"""Conversation repository — Data access layer for Conversation and ChatMessage models."""

import uuid
from typing import List, Optional
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.schemas.conversation import Conversation, ChatMessage


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, conversation: Conversation) -> Conversation:
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def get_by_thread_id(
        self, thread_id: str, user_id: uuid.UUID
    ) -> Optional[Conversation]:
        statement = select(Conversation).where(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def list_by_user(self, user_id: uuid.UUID) -> List[Conversation]:
        statement = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update(self, conversation: Conversation) -> Conversation:
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def delete(self, conversation: Conversation):
        await self.session.delete(conversation)
        await self.session.commit()

    async def get_messages(self, thread_id: str) -> List[ChatMessage]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def delete_messages(self, thread_id: str):
        statement = select(ChatMessage).where(ChatMessage.thread_id == thread_id)
        result = await self.session.execute(statement)
        messages = list(result.scalars().all())
        for msg in messages:
            await self.session.delete(msg)
        await self.session.commit()
