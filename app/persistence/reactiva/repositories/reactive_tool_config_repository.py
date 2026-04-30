"""Repository for reactive tool configurations."""

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.domain.reactiva.schemas.reactive_tool_config import ReactiveToolConfig


class ReactiveToolConfigRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_name(self, name: str) -> Optional[ReactiveToolConfig]:
        stmt = select(ReactiveToolConfig).where(ReactiveToolConfig.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_source(
        self, source_id: uuid.UUID
    ) -> List[ReactiveToolConfig]:
        stmt = select(ReactiveToolConfig).where(
            ReactiveToolConfig.source_id == source_id
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all(self, tenant_id: Optional[str] = None) -> List[ReactiveToolConfig]:
        """Get all reactive tool configs, optionally filtered by tenant."""
        stmt = select(ReactiveToolConfig)
        if tenant_id is not None:
            from app.domain.reactiva.schemas.reactive_mcp_source import ReactiveMCPSource
            stmt = (
                stmt.join(ReactiveMCPSource, ReactiveMCPSource.id == ReactiveToolConfig.source_id)
                .where(ReactiveMCPSource.tenant_id == tenant_id)
            )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, tool_config: ReactiveToolConfig) -> ReactiveToolConfig:
        self.session.add(tool_config)
        await self.session.commit()
        await self.session.refresh(tool_config)
        return tool_config

    async def update(
        self, tool_id: int, data: dict
    ) -> Optional[ReactiveToolConfig]:
        tool = await self.session.get(ReactiveToolConfig, tool_id)
        if not tool:
            return None
        for k, v in data.items():
            if v is not None:
                setattr(tool, k, v)
        self.session.add(tool)
        await self.session.commit()
        await self.session.refresh(tool)
        return tool

    async def delete(self, tool_id: int) -> bool:
        tool = await self.session.get(ReactiveToolConfig, tool_id)
        if not tool:
            return False
        await self.session.delete(tool)
        await self.session.commit()
        return True
