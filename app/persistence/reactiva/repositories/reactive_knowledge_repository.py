"""Repository for reactive knowledge bases and documents."""

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.domain.reactiva.schemas.reactive_knowledge import (
    ReactiveKnowledgeBase,
    ReactiveKnowledgeDocument,
)


class ReactiveKnowledgeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Knowledge Bases ──────────────────────────────────────────────────

    async def create_kb(self, kb: ReactiveKnowledgeBase) -> ReactiveKnowledgeBase:
        self.session.add(kb)
        await self.session.commit()
        await self.session.refresh(kb)
        return kb

    async def get_kb_by_id(self, kb_id: uuid.UUID) -> Optional[ReactiveKnowledgeBase]:
        return await self.session.get(ReactiveKnowledgeBase, kb_id)

    async def list_kbs(self, tenant_id: str = "default") -> List[ReactiveKnowledgeBase]:
        stmt = select(ReactiveKnowledgeBase).where(
            ReactiveKnowledgeBase.tenant_id == tenant_id
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_kb(
        self, kb_id: uuid.UUID, data: dict
    ) -> Optional[ReactiveKnowledgeBase]:
        kb = await self.get_kb_by_id(kb_id)
        if not kb:
            return None
        for k, v in data.items():
            if v is not None:
                setattr(kb, k, v)
        self.session.add(kb)
        await self.session.commit()
        await self.session.refresh(kb)
        return kb

    async def delete_kb(self, kb_id: uuid.UUID) -> bool:
        kb = await self.get_kb_by_id(kb_id)
        if not kb:
            return False
        await self.session.delete(kb)
        await self.session.commit()
        return True

    # ── Documents ────────────────────────────────────────────────────────

    async def add_document(
        self, doc: ReactiveKnowledgeDocument
    ) -> ReactiveKnowledgeDocument:
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def list_documents(
        self, kb_id: uuid.UUID
    ) -> List[ReactiveKnowledgeDocument]:
        stmt = select(ReactiveKnowledgeDocument).where(
            ReactiveKnowledgeDocument.knowledge_base_id == kb_id
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_document(self, doc_id: uuid.UUID) -> bool:
        doc = await self.session.get(ReactiveKnowledgeDocument, doc_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True
