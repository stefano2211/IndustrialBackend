"""
Shared ingestion components — used by both proactiva and reactiva pipelines.

Canonical imports:
    from app.domain.shared.ingestion.embedder import Embedder
    from app.domain.shared.ingestion.document_loader import DocumentLoader
    from app.domain.shared.ingestion.text_splitter import HierarchicalSplitter
"""

from app.domain.shared.ingestion.embedder import Embedder
from app.domain.shared.ingestion.document_loader import DocumentLoader
from app.domain.shared.ingestion.text_splitter import HierarchicalSplitter

__all__ = ["Embedder", "DocumentLoader", "HierarchicalSplitter"]
