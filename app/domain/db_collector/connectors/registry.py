"""
Connector Registry — Maps DbSourceType → connector instance (Strategy Pattern).

To add a new database type:
  1. Create a new file `my_connector.py` implementing BaseDbConnector.
  2. Add one entry to CONNECTOR_REGISTRY below.
  That is the only change required (Open/Closed Principle).
"""

from app.domain.schemas.db_source import DbSourceType

from .base import BaseDbConnector
from .postgresql_connector import PostgresqlConnector
from .mysql_connector import MysqlConnector
from .sqlite_connector import SqliteConnector
from .mongodb_connector import MongodbConnector

CONNECTOR_REGISTRY: dict[DbSourceType, BaseDbConnector] = {
    DbSourceType.POSTGRESQL: PostgresqlConnector(),
    DbSourceType.MYSQL:      MysqlConnector(),
    DbSourceType.SQLITE:     SqliteConnector(),
    DbSourceType.MONGODB:    MongodbConnector(),
}


def get_connector(db_type: DbSourceType) -> BaseDbConnector:
    """Returns the connector for the given db_type or raises ValueError."""
    connector = CONNECTOR_REGISTRY.get(db_type)
    if connector is None:
        raise ValueError(
            f"Unsupported DbSourceType: '{db_type}'. "
            f"Available: {list(CONNECTOR_REGISTRY.keys())}"
        )
    return connector
