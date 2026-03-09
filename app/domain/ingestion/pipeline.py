"""
Document ingestion pipeline.

Orchestrates the full document processing flow:
  Load → Classify → Split → NER → Embed → Store

Dependencies are injected for testability (DIP).
"""

import asyncio
import uuid
from typing import Optional, Any

from qdrant_client.http.models import PointStruct
from loguru import logger

from app.core.config import settings
from app.domain.ingestion.document_loader import DocumentLoader
from app.domain.ingestion.text_splitter import HierarchicalSplitter
from app.domain.ingestion.embedder import Embedder
from app.persistence.vector import QdrantManager
from app.domain.models.ner import get_extractor


class DocumentProcessor:
    """
    Orchestrator for the document ingestion pipeline.

    Dependencies can be overridden in __init__ for testing.
    """

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        splitter: HierarchicalSplitter | None = None,
        embedder: Embedder | None = None,
        vector_store: QdrantManager | None = None,
    ):
        self.loader = loader or DocumentLoader()
        self.splitter = splitter or HierarchicalSplitter()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or QdrantManager()
        self._extractor = None

    def _get_extractor(self):
        """Lazy load LangExtract extractor only when needed."""
        if self._extractor is None:
            self._extractor = get_extractor(device="cpu")
        return self._extractor

    async def process(
        self,
        file_path: str,
        user_id: str,
        doc_id: str | None = None,
        knowledge_base_id: str | None = None,
        session: Optional[Any] = None,
    ):
        doc_id = doc_id or str(uuid.uuid4())
        logger.info(f"Processing document: {file_path} for user: {user_id}")

        # 1. Load
        docs = self.loader.load(file_path)
        full_text = "\n".join([doc.page_content for doc in docs])
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

        # 2. Classify
        extractor = self._get_extractor()
        doc_category = await extractor.classify_document(full_text, session=session)
        logger.info(f"Document classified as: {doc_category}")

        # 3. Split
        chunks = self.splitter.split_documents(docs)

        # 4. NER Enrichment (per chunk, with configurable throttle)
        enriched_chunks = []
        throttle = settings.ner_throttle_seconds
        for i, chunk in enumerate(chunks):
            if i > 0 and throttle > 0:
                await asyncio.sleep(throttle)

            entities = await extractor.extract_entities(
                chunk.page_content, session=session
            )
            chunk.metadata["entities"] = entities
            chunk.metadata["doc_category"] = doc_category
            enriched_chunks.append(chunk)
            logger.debug(f"Chunk {i}: Extracted {len(entities)} entity types")

        # 5. Embed
        texts = [chunk.page_content for chunk in enriched_chunks]
        vectors = self.embedder.embed_documents(texts)

        # 6. Store
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "doc_id": doc_id,
                        "user_id": user_id,
                        "chunk_index": i,
                        "knowledge_base_id": knowledge_base_id,
                    },
                },
            )
            for i, (chunk, vector) in enumerate(zip(enriched_chunks, vectors))
        ]
        self.vector_store.upsert(points)
        logger.success(
            f"Document {doc_id} ({doc_category}) processed: "
            f"{len(chunks)} chunks with NER + classification"
        )
        return {"doc_id": doc_id, "chunks": len(chunks), "category": doc_category}