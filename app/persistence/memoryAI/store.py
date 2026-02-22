from langgraph.store.postgres.aio import AsyncPostgresStore
from app.core.config import settings
from app.persistence.memoryAI.checkpointer import DB_URL

# Internal store instance
_store: AsyncPostgresStore = None

async def get_store():
    """
    Returns an AsyncPostgresStore instance.
    Initializes the store on first call.
    """
    global _store
    if _store is None:
        _store = AsyncPostgresStore.from_conn_string(DB_URL)
        # Ensure tables exist
        await _store.setup()
    
    return _store
