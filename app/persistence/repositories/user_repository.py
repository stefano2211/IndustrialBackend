"""User repository — Data access layer for User model."""

from typing import Optional, List
from uuid import UUID
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.schemas.user import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_email(self, email: str) -> Optional[User]:
        statement = select(User).where(User.email == email)
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        statement = select(User).where(User.id == user_id)
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def list_all(self) -> List[User]:
        statement = select(User).order_by(User.created_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def create(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def update(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def delete(self, user: User) -> None:
        await self.session.delete(user)
        await self.session.commit()
