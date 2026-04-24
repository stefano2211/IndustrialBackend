"""Qdrant manager for the reactive domain — isolated collection.

Inherits all methods from QdrantManager but points to the
`reactive_documents` collection. Same Qdrant host and client,
different namespace.
"""

from app.persistence.vector import QdrantManager
from app.core.config import settings


class ReactiveQdrantManager(QdrantManager):
    """Qdrant wrapper for the reactive domain — isolated collection."""

    def __init__(self):
        super().__init__()
        self.collection_name = settings.reactive_qdrant_collection
        self._initialized = False  # Force re-check for the reactive collection
