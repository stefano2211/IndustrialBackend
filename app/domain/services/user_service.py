from sqlmodel import select
from typing import Optional
from app.domain.schemas.user import User, UserCreate, UserUpdate
from app.core.security import get_password_hash, verify_password
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_email(self, email: str) -> Optional[User]:
        statement = select(User).where(User.email == email)
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def create_user(self, user_in: UserCreate) -> User:
        db_user = User(
            username=user_in.username,
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            is_active=user_in.is_active,
            is_superuser=user_in.is_superuser,
        )
        self.session.add(db_user)
        await self.session.commit()
        await self.session.refresh(db_user)
        return db_user

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        user = await self.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def update_user(self, user: User, user_in: UserUpdate) -> User:
        user_data = user_in.model_dump(exclude_unset=True)
        for key, value in user_data.items():
            if key == "password":
                setattr(user, "hashed_password", get_password_hash(value))
            else:
                setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
