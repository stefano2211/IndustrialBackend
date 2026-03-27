"""MySQL / MariaDB async connector using aiomysql."""

from typing import Any, Dict, List
from urllib.parse import urlparse

from .base import BaseDbConnector


class MysqlConnector(BaseDbConnector):
    """
    Connects to MySQL/MariaDB via aiomysql.
    Supports standard DSN format:
        mysql://user:password@host:port/database
    """

    async def fetch(self, connection_string: str, query: str) -> List[Dict[str, Any]]:
        try:
            import aiomysql
        except ImportError:
            raise ImportError(
                "aiomysql is required for MySQL connections. "
                "Add 'aiomysql' to your dependencies."
            )

        parsed = urlparse(connection_string)
        conn = await aiomysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password or "",
            db=parsed.path.lstrip("/"),
            cursorclass=aiomysql.DictCursor,
        )
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(query)
                rows = await cursor.fetchall()
                return list(rows)  # Already dicts due to DictCursor
        finally:
            conn.close()
