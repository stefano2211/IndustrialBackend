"""
Reactive document ingestion pipeline.

Identical processing flow to the proactive pipeline (Load → Split → Embed → Store)
but stores vectors in the reactive Qdrant collection and tags metadata with
tenant_id instead of user_id.

Used for: SOPs, emergency procedures, maintenance manuals, incident response protocols.
"""

import uuid
from typing import Optional, Any
from pathlib import Path

from qdrant_client.http import models as qmodels
from qdrant_client.http.models import PointStruct
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.domain.proactiva.ingestion.document_loader import DocumentLoader
from app.domain.proactiva.ingestion.text_splitter import HierarchicalSplitter
from app.domain.proactiva.ingestion.embedder import Embedder
from app.persistence.reactiva.reactive_vector import ReactiveQdrantManager


class ReactiveDocumentProcessor:
    """
    Ingestion pipeline for reactive domain documents.

    Reuses shared components (loader, splitter, embedder) but writes
    to the isolated reactive Qdrant collection.
    """

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        splitter: HierarchicalSplitter | None = None,
        embedder: Embedder | None = None,
        vector_store: ReactiveQdrantManager | None = None,
    ):
        self.loader = loader or DocumentLoader()
        self.splitter = splitter or HierarchicalSplitter()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or ReactiveQdrantManager()

    async def process_file(
        self,
        file_path: Path | str,
        tenant_id: str = "default",
        doc_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        session: Optional[Any] = None,
    ) -> dict:
        """Process a file into the reactive knowledge base."""
        doc_id = doc_id or str(uuid.uuid4())
        logger.info(f"[ReactiveIngestion] Processing: {file_path} for tenant: {tenant_id}")

        # 1. Fetch dynamic settings
        chunk_size = 1000
        chunk_overlap = 200
        if session:
            from app.persistence.proactiva.repositories.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            chunk_size = system_settings.document_chunk_size
            chunk_overlap = system_settings.document_chunk_overlap

        # 2. Extract text
        docs = self.loader.load(file_path)
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

        # 3. Split — Hierarchical → Recursive
        hierarchical_chunks = self.splitter.split_documents(docs)
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
                "tenant_id": tenant_id,
                "chunk_index": i,
                "knowledge_base_id": knowledge_base_id,
                "doc_category": "reactive_document",
                "section": split_doc.metadata.get("section", "No section"),
                **{
                    k: v
                    for k, v in split_doc.metadata.items()
                    if k not in ("doc_id", "tenant_id", "chunk_index", "knowledge_base_id", "doc_category")
                },
            }
            chunks.append(Document(page_content=split_doc.page_content, metadata=new_metadata))

        total_chunks = len(chunks)
        logger.info(f"[ReactiveIngestion] Document split into {total_chunks} chunks")

        # 4. Embed (Dual: Dense + Sparse SPLADE)
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors = await self.embedder.embed_documents(texts)
        sparse_vectors = await self.embedder.embed_sparse_documents(texts)

        # 5. Store in reactive Qdrant collection
        points = []
        for i, (chunk, d_vec, s_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d_vec,
                        "sparse": qmodels.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "text": chunk.page_content,
                        "metadata": {
                            **chunk.metadata,
                            "doc_id": doc_id,
                            "tenant_id": tenant_id,
                            "chunk_index": i,
                            "knowledge_base_id": knowledge_base_id,
                        },
                    },
                )
            )
        await self.vector_store.upsert(points)
        logger.success(
            f"[ReactiveIngestion] Document {doc_id} processed: {total_chunks} chunks "
            f"→ reactive collection"
        )
        return {"doc_id": doc_id, "chunks": total_chunks, "category": "reactive_document"}
