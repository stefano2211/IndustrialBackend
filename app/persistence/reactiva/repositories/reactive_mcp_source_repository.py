"""Repository for reactive MCP sources."""

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.domain.reactiva.schemas.reactive_mcp_source import ReactiveMCPSource


class ReactiveMCPSourceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, source_id: uuid.UUID) -> Optional[ReactiveMCPSource]:
        return await self.session.get(ReactiveMCPSource, source_id)

    async def list_all(self, tenant_id: str = "default") -> List[ReactiveMCPSource]:
        stmt = select(ReactiveMCPSource).where(
            ReactiveMCPSource.tenant_id == tenant_id
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, source: ReactiveMCPSource) -> ReactiveMCPSource:
        self.session.add(source)
        await self.session.commit()
        await self.session.refresh(source)
        return source

    async def update(
        self, source_id: uuid.UUID, data: dict
    ) -> Optional[ReactiveMCPSource]:
        source = await self.get_by_id(source_id)
        if not source:
            return None
        for k, v in data.items():
            if v is not None:
                setattr(source, k, v)
        self.session.add(source)
        await self.session.commit()
        await self.session.refresh(source)
        return source

    async def delete(self, source_id: uuid.UUID) -> bool:
        source = await self.get_by_id(source_id)
        if not source:
            return False
        await self.session.delete(source)
        await self.session.commit()
        return True
