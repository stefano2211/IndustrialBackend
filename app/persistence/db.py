from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from typing import AsyncGenerator

# Import all models so SQLModel.metadata.create_all creates their tables
from app.domain.schemas.user import User  # noqa: F401
from app.domain.schemas.conversation import Conversation, ChatMessage  # noqa: F401

# Construct Database URL
# Ensure these variables are in your .env or settings
DATABASE_URL = f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

engine = create_async_engine(DATABASE_URL, echo=False, future=True)

async def init_db():
    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all) # Uncomment to reset DB
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
