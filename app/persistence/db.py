"""Database session management."""

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from typing import AsyncGenerator

# Import all models so SQLModel.metadata.create_all creates their tables
from app.domain.schemas.user import User  # noqa: F401
from app.domain.schemas.conversation import Conversation, ChatMessage  # noqa: F401
from app.domain.schemas.knowledge import KnowledgeBase, KnowledgeDocument  # noqa: F401
from app.domain.schemas.prompt import Prompt  # noqa: F401
from app.domain.schemas.llm_config import LLMConfig  # noqa: F401
from app.domain.schemas.model import Model  # noqa: F401

DATABASE_URL = (
    f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Create session factory ONCE at module level (not per request)
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
