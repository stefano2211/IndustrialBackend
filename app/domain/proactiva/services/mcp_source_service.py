from typing import List, Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.persistence.proactiva.repositories.mcp_source_repository import MCPSourceRepository
from app.domain.schemas.mcp_source import MCPSource, MCPSourceCreate, MCPSourceUpdate
from app.domain.exceptions import NotFoundError

class MCPSourceService:
    def __init__(self, session: AsyncSession):
        self.repository = MCPSourceRepository(session)

    async def get_all(self, user_id: uuid.UUID) -> List[MCPSource]:
        return await self.repository.get_all(user_id)

    async def get_by_id(self, source_id: uuid.UUID, user_id: uuid.UUID) -> MCPSource:
        source = await self.repository.get_by_id(source_id, user_id)
        if not source:
            raise NotFoundError(f"MCP Source with id {source_id} not found for this user")
        return source

    async def create(self, user_id: uuid.UUID, source_in: MCPSourceCreate) -> MCPSource:
        source = MCPSource(**source_in.dict())
        source.user_id = user_id
        return await self.repository.create(source)

    async def update(self, source_id: uuid.UUID, user_id: uuid.UUID, source_in: MCPSourceUpdate) -> MCPSource:
        source = await self.get_by_id(source_id, user_id)
        update_data = source_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(source, field, value)
        return await self.repository.update(source)

    async def delete(self, source_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        source = await self.get_by_id(source_id, user_id)
        await self.repository.delete(source)
        return True
