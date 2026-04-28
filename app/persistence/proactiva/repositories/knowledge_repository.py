import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from app.domain.proactiva.schemas.knowledge import KnowledgeBase, KnowledgeDocument

class KnowledgeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_knowledge_base(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        self.session.add(knowledge_base)
        await self.session.commit()
        await self.session.refresh(knowledge_base)
        return knowledge_base

    async def get_knowledge_base(self, kb_id: uuid.UUID, user_id: uuid.UUID) -> Optional[KnowledgeBase]:
        statement = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def list_knowledge_bases(self, user_id: uuid.UUID) -> List[KnowledgeBase]:
        statement = select(KnowledgeBase).where(KnowledgeBase.user_id == user_id)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update_knowledge_base(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        self.session.add(knowledge_base)
        await self.session.commit()
        await self.session.refresh(knowledge_base)
        return knowledge_base

    async def delete_knowledge_base(self, knowledge_base: KnowledgeBase):
        await self.session.delete(knowledge_base)
        await self.session.commit()

    async def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get_documents_by_kb(self, kb_id: uuid.UUID) -> List[KnowledgeDocument]:
        statement = select(KnowledgeDocument).where(KnowledgeDocument.knowledge_base_id == kb_id)
        result = await self.session.execute(statement)
        return list(result.scalars().all())
        
    async def delete_document(self, document: KnowledgeDocument):
        await self.session.delete(document)
        await self.session.commit()

    async def delete_document_by_file_id(self, file_id: str):
        statement = select(KnowledgeDocument).where(KnowledgeDocument.file_id == file_id)
        result = await self.session.execute(statement)
        docs = result.scalars().all()
        for doc in docs:
            await self.session.delete(doc)
        await self.session.commit()
