import uuid
from typing import List, Optional
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.schemas.tool_config import ToolConfig

class ToolConfigRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all_by_user(self, user_id: uuid.UUID) -> List[ToolConfig]:
        from app.domain.schemas.mcp_source import MCPSource
        statement = select(ToolConfig).join(MCPSource).where(MCPSource.user_id == user_id)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_all(self) -> List[ToolConfig]:
        statement = select(ToolConfig)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_source(self, source_id: uuid.UUID) -> List[ToolConfig]:
        statement = select(ToolConfig).where(ToolConfig.source_id == source_id)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_name(self, name: str) -> Optional[ToolConfig]:
        statement = select(ToolConfig).where(ToolConfig.name == name)
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_id(self, tool_id: int) -> Optional[ToolConfig]:
        return await self.session.get(ToolConfig, tool_id)

    async def create(self, tool: ToolConfig) -> ToolConfig:
        self.session.add(tool)
        await self.session.commit()
        await self.session.refresh(tool)
        return tool

    async def update(self, tool: ToolConfig) -> ToolConfig:
        self.session.add(tool)
        await self.session.commit()
        await self.session.refresh(tool)
        return tool

    async def delete(self, tool: ToolConfig):
        await self.session.delete(tool)
        await self.session.commit()
