"""Repository for DbSource CRUD operations."""

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.domain.schemas.db_source import DbSource


class DbSourceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all_enabled(self) -> List[DbSource]:
        """Returns all sources with is_enabled=True."""
        stmt = select(DbSource).where(DbSource.is_enabled == True)  # noqa: E712
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all(self) -> List[DbSource]:
        """Returns all sources (enabled + disabled)."""
        result = await self.session.execute(select(DbSource))
        return list(result.scalars().all())

    async def get_by_id(self, source_id: uuid.UUID) -> Optional[DbSource]:
        return await self.session.get(DbSource, source_id)

    async def create(self, source: DbSource) -> DbSource:
        self.session.add(source)
        await self.session.commit()
        await self.session.refresh(source)
        return source

    async def save(self, source: DbSource) -> DbSource:
        """Update (add to session + commit). Also used for metadata updates."""
        self.session.add(source)
        await self.session.commit()
        await self.session.refresh(source)
        return source

    async def delete(self, source: DbSource) -> None:
        await self.session.delete(source)
        await self.session.commit()
