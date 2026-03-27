"""Abstract base class for all database connectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDbConnector(ABC):
    """
    Strategy interface for database connectors.
    Each implementation handles one engine type and returns
    rows as a list of dicts (column → value).
    """

    @abstractmethod
    async def fetch(self, connection_string: str, query: str) -> List[Dict[str, Any]]:
        """
        Execute the given query against the database at connection_string.

        Args:
            connection_string: Plain-text URI (already decrypted).
            query: SQL SELECT or engine-specific query descriptor.

        Returns:
            List of row dicts. Empty list if no data or query returns nothing.

        Raises:
            Exception: Propagates any connection or query errors upward
                       so the caller can catch and log them.
        """
        ...
