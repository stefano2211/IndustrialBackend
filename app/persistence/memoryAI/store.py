from langgraph.store.postgres.aio import AsyncPostgresStore
from app.core.config import settings
from app.persistence.memoryAI.checkpointer import get_pool

# Internal store instance
_store: AsyncPostgresStore = None

async def get_store():
    """
    Returns an AsyncPostgresStore instance.
    Initializes the store on first call using the shared pool.
    """
    global _store
    if _store is None:
        pool = await get_pool()
        _store = AsyncPostgresStore(pool)
        # Ensure tables exist
        await _store.setup()
    
    return _store
