"""ToolConfig service — Business logic for tool configurations."""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas.tool_config import ToolConfig, ToolConfigCreate, ToolConfigUpdate
from app.domain.exceptions import NotFoundError
from app.persistence.repositories.tool_config_repository import ToolConfigRepository


class ToolConfigService:
    def __init__(self, session: AsyncSession):
        self.repository = ToolConfigRepository(session)

    async def get_all(self) -> List[ToolConfig]:
        return await self.repository.get_all()

    async def get_by_name(self, name: str) -> Optional[ToolConfig]:
        return await self.repository.get_by_name(name)

    async def get_by_id(self, tool_id: int) -> ToolConfig:
        tool = await self.repository.get_by_id(tool_id)
        if not tool:
            raise NotFoundError("ToolConfig", tool_id)
        return tool

    async def create(self, tool_in: ToolConfigCreate) -> ToolConfig:
        tool = ToolConfig.model_validate(tool_in)
        return await self.repository.create(tool)

    async def update(self, tool_id: int, tool_in: ToolConfigUpdate) -> ToolConfig:
        tool = await self.get_by_id(tool_id)
        update_data = tool_in.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(tool, key, value)
        return await self.repository.update(tool)

    async def delete(self, tool_id: int) -> bool:
        tool = await self.repository.get_by_id(tool_id)
        if not tool:
            return False
        await self.repository.delete(tool)
        return True
