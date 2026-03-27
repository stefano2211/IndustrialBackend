"""SQLite async connector using aiosqlite — useful for local PLC logs or test DBs."""

from typing import Any, Dict, List
from urllib.parse import urlparse

from .base import BaseDbConnector


class SqliteConnector(BaseDbConnector):
    """
    Connects to a local SQLite file via aiosqlite.
    Supported DSN formats:
        sqlite:///absolute/path/to/file.db
        sqlite:////absolute/path/to/file.db  (four slashes = absolute on Unix)
    """

    async def fetch(self, connection_string: str, query: str) -> List[Dict[str, Any]]:
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for SQLite connections. "
                "Add 'aiosqlite' to your dependencies."
            )

        # Extract the path from the DSN
        parsed = urlparse(connection_string)
        # urlparse puts the path in .path; for sqlite:///tmp/foo.db → path = /tmp/foo.db
        db_path = parsed.path or connection_string.replace("sqlite:///", "")

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
