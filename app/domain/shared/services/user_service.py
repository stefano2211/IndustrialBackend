"""User service — Business logic for user management."""

from typing import Optional, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from app.domain.shared.schemas.user import User, UserCreate, UserUpdate
from app.core.security import get_password_hash, verify_password
from app.persistence.shared.user_repository import UserRepository


class UserService:
    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)

    async def get_by_email(self, email: str) -> Optional[User]:
        return await self.repository.get_by_email(email)

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        return await self.repository.get_by_id(user_id)

    async def list_users(self) -> List[User]:
        return await self.repository.list_all()

    async def create_user(self, user_in: UserCreate) -> User:
        db_user = User(
            username=user_in.username,
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            is_active=user_in.is_active,
            is_superuser=user_in.is_superuser,
        )
        return await self.repository.create(db_user)

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        user = await self.repository.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user

    async def update_user(self, user: User, user_in: UserUpdate) -> User:
        user_data = user_in.model_dump(exclude_unset=True)
        for key, value in user_data.items():
            if key == "password":
                setattr(user, "hashed_password", get_password_hash(value))
            else:
                setattr(user, key, value)

        user.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return await self.repository.update(user)

    async def update_user_role(self, user_id: UUID, is_superuser: bool) -> Optional[User]:
        user = await self.repository.get_by_id(user_id)
        if not user:
            return None
        user.is_superuser = is_superuser
        user.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return await self.repository.update(user)

    async def delete_user(self, user_id: UUID) -> bool:
        user = await self.repository.get_by_id(user_id)
        if not user:
            return False
        await self.repository.delete(user)
        return True
