"""Event repository — Data access layer for the reactive Event model."""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from sqlmodel import select

from app.domain.reactiva.schemas.event import Event


class EventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, event: Event) -> Event:
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event

    async def get_by_id(self, event_id: uuid.UUID) -> Optional[Event]:
        return await self.session.get(Event, event_id)

    async def list_all(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Event], int]:
        stmt = select(Event)
        if severity:
            stmt = stmt.where(Event.severity == severity)
        if status:
            stmt = stmt.where(Event.status == status)
        if source_type:
            stmt = stmt.where(Event.source_type == source_type)
        # Total count (before pagination)
        count_stmt = select(func.count(Event.id)).select_from(stmt.subquery())
        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar() or 0
        # Paginated items
        stmt = stmt.order_by(Event.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def update_status(self, event_id: uuid.UUID, status: str) -> Optional[Event]:
        event = await self.get_by_id(event_id)
        if not event:
            return None
        event.status = status
        event.updated_at = datetime.utcnow()
        if status in ("completed", "failed"):
            event.resolved_at = datetime.utcnow()
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event

    async def update_analysis(
        self,
        event_id: uuid.UUID,
        analysis: str,
        plan: Optional[str] = None,
        actions: Optional[list] = None,
    ) -> Optional[Event]:
        event = await self.get_by_id(event_id)
        if not event:
            return None
        event.agent_analysis = analysis
        if plan is not None:
            event.agent_plan = plan
        if actions is not None:
            event.actions_taken = actions
        event.updated_at = datetime.utcnow()
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event

    async def save(self, event: Event) -> Event:
        event.updated_at = datetime.utcnow()
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event
