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
from pathlib import Path
from langchain_core.documents import Document
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
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

    async def process_file(
        self,
        file_path: Path | str,
        user_id: str,
        doc_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        session: Optional[Any] = None,
    ) -> dict:
        """Process a single file through the entire pipeline."""
        doc_id = doc_id or str(uuid.uuid4()) # Generate doc_id early for metadata
        logger.info(f"Processing document: {file_path} for user: {user_id}")

        # 1. Fetch dynamic settings
        chunk_size = 1000
        chunk_overlap = 200
        if session:
            from app.persistence.repositories.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            chunk_size = system_settings.document_chunk_size
            chunk_overlap = system_settings.document_chunk_overlap

        # 2. Extract Text & Classify Category
        # Assuming self.loader has an extract_text method or similar
        # The original code used self.loader.load which returns a list of Documents
        # The new snippet implies a direct text extraction.
        # For now, let's adapt to the original loader's output and then classify.
        docs = self.loader.load(file_path)
        full_text = "\n".join([doc.page_content for doc in docs])
        for doc in docs:
            doc.metadata["doc_id"] = doc_id # Ensure doc_id is set for initial docs

        extractor = self._get_extractor()
        doc_category = await extractor.classify_document(full_text, session=session)
        logger.info(f"Document classified as: {doc_category}")

        # 3. Split
        # The original code used self.splitter.split_documents(docs)
        # The new snippet introduces RecursiveCharacterTextSplitter.
        # Let's use the new splitter with dynamic chunk sizes.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        # Assuming the splitter takes the full_text and returns Document objects
        # or we need to convert the original docs to text for this splitter.
        # For consistency with the new splitter, let's split the full_text.
        # We'll need to re-add metadata to these new chunks.
        split_texts = text_splitter.create_documents([full_text])
        chunks = []
        for i, split_doc in enumerate(split_texts):
            # Re-apply original document metadata and add chunk-specific metadata
            new_metadata = {
                "doc_id": doc_id,
                "user_id": user_id,
                "chunk_index": i,
                "knowledge_base_id": knowledge_base_id,
                "doc_category": doc_category, # Add category here for early access
                **docs[0].metadata # Inherit other metadata from the first original doc
            }
            chunks.append(Document(page_content=split_doc.page_content, metadata=new_metadata))

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