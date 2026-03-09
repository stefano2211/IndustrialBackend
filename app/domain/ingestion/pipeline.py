"""
Document ingestion pipeline.

Orchestrates the full document processing flow:
  Load → Classify → Split → NER (batched + concurrent) → Embed → Store

Dependencies are injected for testability (DIP).
"""

import asyncio
import uuid
from typing import Optional, Any, List

from qdrant_client.http.models import PointStruct
from langchain_core.documents import Document
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
        total_chunks = len(chunks)
        logger.info(f"Document split into {total_chunks} chunks")

        # 4. NER Enrichment — Batched + Concurrent
        enriched_chunks = await self._enrich_chunks_with_ner(
            chunks, extractor, doc_category, session
        )

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
            f"{total_chunks} chunks with NER + classification"
        )
        return {"doc_id": doc_id, "chunks": total_chunks, "category": doc_category}

    # ------------------------------------------------------------------
    #  Batched + Concurrent NER Enrichment
    # ------------------------------------------------------------------

    async def _enrich_chunks_with_ner(
        self,
        chunks: List[Document],
        extractor,
        doc_category: str,
        session: Optional[Any],
    ) -> List[Document]:
        """
        Enrich chunks with NER entities using batch processing and concurrency.

        Strategy:
          1. Divide chunks into batches of `ner_batch_size`
          2. Process up to `ner_max_concurrency` batches in parallel
          3. Each batch sends a SINGLE LLM call for all its chunks
        """
        batch_size = settings.ner_batch_size
        max_concurrency = settings.ner_max_concurrency

        # Split chunks into batches
        batches: List[List[Document]] = [
            chunks[i : i + batch_size]
            for i in range(0, len(chunks), batch_size)
        ]

        logger.info(
            f"NER enrichment: {len(chunks)} chunks → {len(batches)} batches "
            f"(size={batch_size}, concurrency={max_concurrency})"
        )

        # Process batches concurrently with a semaphore
        semaphore = asyncio.Semaphore(max_concurrency)
        all_entity_results: List[List[dict]] = [None] * len(batches)

        async def process_batch(batch_idx: int, batch: List[Document]):
            async with semaphore:
                texts = [chunk.page_content for chunk in batch]
                logger.debug(
                    f"Processing NER batch {batch_idx + 1}/{len(batches)} "
                    f"({len(texts)} chunks)"
                )
                entities_list = await extractor.extract_entities_batch(
                    texts, session=session
                )
                all_entity_results[batch_idx] = entities_list

        # Launch all batches concurrently (semaphore limits parallelism)
        tasks = [
            process_batch(i, batch) for i, batch in enumerate(batches)
        ]
        await asyncio.gather(*tasks)

        # Flatten results and assign entities to each chunk
        enriched_chunks = []
        for batch_idx, batch in enumerate(batches):
            entities_list = all_entity_results[batch_idx] or [{} for _ in batch]
            for chunk, entities in zip(batch, entities_list):
                chunk.metadata["entities"] = entities
                chunk.metadata["doc_category"] = doc_category
                enriched_chunks.append(chunk)

        logger.info(f"NER enrichment complete: {len(enriched_chunks)} chunks enriched")
        return enriched_chunks