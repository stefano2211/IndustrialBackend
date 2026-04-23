import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.schemas.knowledge import (
    KnowledgeBase,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeDocument,
)
from app.domain.exceptions import NotFoundError
from app.persistence.proactiva.repositories.knowledge_repository import KnowledgeRepository


class KnowledgeService:
    def __init__(self, session: AsyncSession):
        self.repository = KnowledgeRepository(session)

    async def create_knowledge_base(
        self, user_id: uuid.UUID, data: KnowledgeBaseCreate
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            user_id=user_id, name=data.name, description=data.description
        )
        return await self.repository.create_knowledge_base(kb)

    async def get_knowledge_base(
        self, kb_id: uuid.UUID, user_id: uuid.UUID
    ) -> KnowledgeBase:
        kb = await self.repository.get_knowledge_base(kb_id, user_id)
        if not kb:
            raise NotFoundError("KnowledgeBase", kb_id)
        return kb

    async def list_knowledge_bases(
        self, user_id: uuid.UUID
    ) -> List[KnowledgeBase]:
        return await self.repository.list_knowledge_bases(user_id)

    async def update_knowledge_base(
        self,
        kb_id: uuid.UUID,
        user_id: uuid.UUID,
        data: KnowledgeBaseUpdate,
    ) -> KnowledgeBase:
        kb = await self.get_knowledge_base(kb_id, user_id)
        if data.name is not None:
            kb.name = data.name
        if data.description is not None:
            kb.description = data.description
        return await self.repository.update_knowledge_base(kb)

    async def delete_knowledge_base(
        self, kb_id: uuid.UUID, user_id: uuid.UUID
    ):
        kb = await self.get_knowledge_base(kb_id, user_id)
        await self.repository.delete_knowledge_base(kb)

    async def add_document_to_kb(
        self,
        kb_id: uuid.UUID,
        user_id: uuid.UUID,
        file_id: str,
        filename: str,
    ) -> KnowledgeDocument:
        await self.get_knowledge_base(kb_id, user_id)
        doc = KnowledgeDocument(
            knowledge_base_id=kb_id, file_id=file_id, filename=filename
        )
        return await self.repository.add_document(doc)

    async def get_kb_documents(
        self, kb_id: uuid.UUID, user_id: uuid.UUID
    ) -> List[KnowledgeDocument]:
        await self.get_knowledge_base(kb_id, user_id)
        return await self.repository.get_documents_by_kb(kb_id)

    async def remove_document_from_kb(
        self, kb_id: uuid.UUID, user_id: uuid.UUID, file_id: str
    ) -> bool:
        await self.get_knowledge_base(kb_id, user_id)
        docs = await self.repository.get_documents_by_kb(kb_id)
        for doc in docs:
            if doc.file_id == file_id:
                await self.repository.delete_document(doc)
                return True
        return False

    async def delete_document_by_file_id(self, file_id: str):
        """Delete a document record by its file_id (used when deleting from Qdrant)."""
        await self.repository.delete_document_by_file_id(file_id)
