"""Conversation service — Business logic for conversation management."""

import uuid
from typing import List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas.conversation import (
    Conversation,
    ConversationCreate,
    ChatMessage,
)
from app.domain.exceptions import NotFoundError
from app.persistence.repositories.conversation_repository import ConversationRepository


class ConversationService:
    def __init__(self, session: AsyncSession):
        self.repository = ConversationRepository(session)

    async def create_conversation(
        self, user_id: uuid.UUID, data: ConversationCreate
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            thread_id=str(uuid.uuid4()),
            title=data.title,
        )
        return await self.repository.create(conversation)

    async def list_conversations(self, user_id: uuid.UUID) -> List[Conversation]:
        return await self.repository.list_by_user(user_id)

    async def get_conversation(
        self, thread_id: str, user_id: uuid.UUID
    ) -> Conversation:
        conversation = await self.repository.get_by_thread_id(thread_id, user_id)
        if not conversation:
            raise NotFoundError("Conversation", thread_id)
        return conversation

    async def get_messages(
        self, thread_id: str, user_id: uuid.UUID
    ) -> List[ChatMessage]:
        await self.get_conversation(thread_id, user_id)  # verify ownership
        return await self.repository.get_messages(thread_id)

    async def delete_conversation(
        self, thread_id: str, user_id: uuid.UUID
    ):
        conversation = await self.get_conversation(thread_id, user_id)
        await self.repository.delete_messages(thread_id)
        await self.repository.delete(conversation)

    async def get_or_create_conversation(
        self, thread_id: str, user_id: uuid.UUID, title: str = "New Chat"
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        conversation = await self.repository.get_by_thread_id(thread_id, user_id)
        if not conversation:
            conversation = Conversation(
                user_id=user_id, thread_id=thread_id, title=title
            )
            conversation = await self.repository.create(conversation)
        else:
            conversation.updated_at = datetime.now(timezone.utc)
            conversation = await self.repository.update(conversation)
        return conversation
