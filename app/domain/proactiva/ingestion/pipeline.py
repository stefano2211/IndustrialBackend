"""
Document ingestion pipeline.

Orchestrates the full document processing flow:
  Load ? Classify ? Split ? NER (batched + concurrent) ? Embed ? Store

Dependencies are injected for testability (DIP).
"""

import uuid
from typing import Optional, Any

from qdrant_client.http import models as qmodels
from qdrant_client.http.models import PointStruct
from pathlib import Path
from langchain_core.documents import Document
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.domain.shared.ingestion.document_loader import DocumentLoader
from app.domain.shared.ingestion.text_splitter import HierarchicalSplitter
from app.domain.shared.ingestion.embedder import Embedder
from app.persistence.vector import QdrantManager


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
            from app.persistence.proactiva.repositories.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            chunk_size = system_settings.document_chunk_size
            chunk_overlap = system_settings.document_chunk_overlap

        # 2. Extract Text
        docs = self.loader.load(file_path)
        for doc in docs:
            doc.metadata["doc_id"] = doc_id # Ensure doc_id is set for initial docs

        doc_category = "document" # Default category since we removed classification
        logger.info(f"Document category set to default: {doc_category}")

        # 3. Split — Two-stage: Hierarchical (section detection) ? Recursive (size enforcement)
        # Stage 1: the HierarchicalSplitter adds section metadata to each chunk
        hierarchical_chunks = self.splitter.split_documents(docs)

        # Stage 2: enforce chunk_size limits using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        split_chunks = text_splitter.split_documents(hierarchical_chunks)
        chunks = []
        for i, split_doc in enumerate(split_chunks):
            new_metadata = {
                "doc_id": doc_id,
                "user_id": user_id,
                "chunk_index": i,
                "knowledge_base_id": knowledge_base_id,
                "doc_category": doc_category,
                "section": split_doc.metadata.get("section", "No section"),
                **{k: v for k, v in split_doc.metadata.items() if k not in ("doc_id", "user_id", "chunk_index", "knowledge_base_id", "doc_category")},
            }
            chunks.append(Document(page_content=split_doc.page_content, metadata=new_metadata))

        total_chunks = len(chunks)
        logger.info(f"Document split into {total_chunks} chunks")

        # 4. Embed (Dual: Dense + Sparse SPLADE)
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors = await self.embedder.embed_documents(texts)
        sparse_vectors = await self.embedder.embed_sparse_documents(texts)

        # 5. Store
        points = []
        for i, (chunk, d_vec, s_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d_vec,
                        "sparse": qmodels.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist()
                        )
                    },
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
            )
        await self.vector_store.upsert(points)
        logger.success(
            f"Document {doc_id} processed: "
            f"{total_chunks} chunks"
        )
        return {"doc_id": doc_id, "chunks": total_chunks, "category": doc_category}

