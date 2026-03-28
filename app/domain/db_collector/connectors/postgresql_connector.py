"""PostgreSQL async connector using asyncpg (already a project dependency)."""

import re
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
        # asyncpg strictly expects "postgresql://" or "postgres://"
        if connection_string.startswith("postgresql+"):
            connection_string = re.sub(r"^postgresql\+[a-zA-Z0-9_]+://", "postgresql://", connection_string)

        conn = None
        try:
            conn = await asyncpg.connect(connection_string)
            rows = await conn.fetch(query)
            # asyncpg Row objects are dict-like; convert to plain dicts
            return [dict(row) for row in rows]
        finally:
            if conn is not None:
                await conn.close()

