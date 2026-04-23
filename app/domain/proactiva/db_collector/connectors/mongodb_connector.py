"""MongoDB async connector using motor."""

from typing import Any, Dict, List
import json

from .base import BaseDbConnector


class MongodbConnector(BaseDbConnector):
    """
    Connects to MongoDB via motor (async driver).

    The `query` field must be a JSON string with the following keys:
        - "collection" (required): Name of the MongoDB collection.
        - "filter"     (optional): MongoDB filter document. Default: {} (all docs).
        - "limit"      (optional): Max number of documents to return. Default: 100.
        - "projection" (optional): Fields to include/exclude. Default: None (all fields).

    Example query value:
        {"collection": "sensor_readings", "filter": {"active": true}, "limit": 50}
    """

    async def fetch(self, connection_string: str, query: str) -> List[Dict[str, Any]]:
        try:
            import motor.motor_asyncio as motor_async
        except ImportError:
            raise ImportError(
                "motor is required for MongoDB connections. "
                "Install it via: pip install motor"
            )

        # Parse the query descriptor
        try:
            query_config = json.loads(query)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"MongoDB query must be a valid JSON string. Got: {query!r}"
            ) from exc

        collection_name = query_config.get("collection")
        if not collection_name:
            raise ValueError("MongoDB query JSON must include a 'collection' key.")

        filter_doc   = query_config.get("filter", {})
        limit        = int(query_config.get("limit", 100))
        projection   = query_config.get("projection", None)

        client = motor_async.AsyncIOMotorClient(connection_string)
        try:
            # Infer DB name from the connection string path (standard Mongo URI)
            db_name = client.get_default_database().name
            collection = client[db_name][collection_name]

            cursor = collection.find(filter_doc, projection).limit(limit)
            docs = await cursor.to_list(length=limit)

            # Convert ObjectId and other non-serializable types to strings
            rows = []
            for doc in docs:
                row = {}
                for key, val in doc.items():
                    if key == "_id":
                        row["_id"] = str(val)
                    else:
                        row[key] = val
                rows.append(row)
            return rows
        finally:
            client.close()
