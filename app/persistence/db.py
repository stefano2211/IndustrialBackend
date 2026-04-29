"""Database session management."""

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from typing import AsyncGenerator

# Import all models so SQLModel.metadata.create_all creates their tables
from app.domain.shared.schemas.user import User  # noqa: F401
from app.domain.proactiva.schemas.conversation import Conversation, ChatMessage  # noqa: F401
from app.domain.proactiva.schemas.knowledge import KnowledgeBase, KnowledgeDocument  # noqa: F401
from app.domain.proactiva.schemas.prompt import Prompt  # noqa: F401
from app.domain.proactiva.schemas.llm_config import LLMConfig  # noqa: F401
from app.domain.proactiva.schemas.model import Model  # noqa: F401
from app.domain.proactiva.schemas.mcp_source import MCPSource  # noqa: F401
from app.domain.proactiva.schemas.tool_config import ToolConfig  # noqa: F401
from app.domain.schemas.db_source import DbSource  # noqa: F401
from app.domain.reactiva.schemas.event import Event  # noqa: F401
# Reactive domain schemas
from app.domain.reactiva.schemas.reactive_mcp_source import ReactiveMCPSource  # noqa: F401
from app.domain.reactiva.schemas.reactive_tool_config import ReactiveToolConfig  # noqa: F401
from app.domain.reactiva.schemas.reactive_knowledge import ReactiveKnowledgeBase, ReactiveKnowledgeDocument  # noqa: F401

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
    """Get database session with automatic rollback on error."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
