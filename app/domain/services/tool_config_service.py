from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.models.tool_config import ToolConfig
from app.domain.schemas.tool_config import ToolConfigCreate, ToolConfigUpdate
from typing import List, Optional

class ToolConfigService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> List[ToolConfig]:
        statement = select(ToolConfig)
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def get_by_name(self, name: str) -> Optional[ToolConfig]:
        statement = select(ToolConfig).where(ToolConfig.name == name)
        result = await self.session.execute(statement)
        return result.scalars().first()
    
    async def get_by_id(self, id: int) -> Optional[ToolConfig]:
        return await self.session.get(ToolConfig, id)

    async def create(self, tool_in: ToolConfigCreate) -> ToolConfig:
        tool = ToolConfig.model_validate(tool_in)
        self.session.add(tool)
        await self.session.commit()
        await self.session.refresh(tool)
        return tool

    async def update(self, id: int, tool_in: ToolConfigUpdate) -> Optional[ToolConfig]:
        tool = await self.get_by_id(id)
        if not tool:
            return None
        
        update_data = tool_in.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(tool, key, value)
            
        self.session.add(tool)
        await self.session.commit()
        await self.session.refresh(tool)
        return tool

    async def delete(self, id: int) -> bool:
        tool = await self.get_by_id(id)
        if not tool:
            return False
        await self.session.delete(tool)
        await self.session.commit()
        return True
