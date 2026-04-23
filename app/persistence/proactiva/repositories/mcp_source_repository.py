from typing import List, Optional
import uuid
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.schemas.mcp_source import MCPSource

class MCPSourceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, user_id: uuid.UUID) -> List[MCPSource]:
        statement = select(MCPSource).where(MCPSource.user_id == user_id)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_id(self, source_id: uuid.UUID, user_id: uuid.UUID) -> Optional[MCPSource]:
        statement = select(MCPSource).where(MCPSource.id == source_id, MCPSource.user_id == user_id)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def create(self, source: MCPSource) -> MCPSource:
        self.session.add(source)
        await self.session.commit()
        await self.session.refresh(source)
        return source

    async def update(self, source: MCPSource) -> MCPSource:
        self.session.add(source)
        await self.session.commit()
        await self.session.refresh(source)
        return source

    async def delete(self, source: MCPSource):
        await self.session.delete(source)
        await self.session.commit()
