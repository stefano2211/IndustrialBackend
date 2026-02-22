from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from app.core.config import settings
import os

# Connection string for psycopg (no +asyncpg prefix)
DB_URL = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

# Internal pool instance
_pool: AsyncConnectionPool = None

async def get_pool():
    """Returns the shared AsyncConnectionPool instance."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=DB_URL,
            max_size=10,
            kwargs={"autocommit": True}
        )
    return _pool

async def get_checkpointer():
    """
    Returns an AsyncPostgresSaver instance.
    Initializes the connection pool on first call.
    """
    pool = await get_pool()
    
    # Initialize and setup tables if necessary
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver

async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
