"""PostgreSQL async connector using asyncpg (already a project dependency)."""

from typing import Any, Dict, List

import asyncpg

from .base import BaseDbConnector


class PostgresqlConnector(BaseDbConnector):
    """
    Connects to PostgreSQL via asyncpg.
    Supports standard DSN format:
        postgresql://user:password@host:port/database
    """

    async def fetch(self, connection_string: str, query: str) -> List[Dict[str, Any]]:
        conn = await asyncpg.connect(connection_string)
        try:
            rows = await conn.fetch(query)
            # asyncpg Row objects are dict-like; convert to plain dicts
            return [dict(row) for row in rows]
        finally:
            await conn.close()
