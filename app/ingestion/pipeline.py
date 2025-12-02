# app/ingestion/pipeline.py
from uuid import uuid4
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.text_splitter import HierarchicalSplitter
from app.ingestion.embedder import Embedder
from app.storage.vector import QdrantManager
from app.models.ner import get_extractor  
from qdrant_client.http.models import PointStruct
from loguru import logger
import uuid

class DocumentProcessor:
    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = HierarchicalSplitter()
        self.embedder = Embedder()
        self.vector_store = QdrantManager()
        self.extractor = None 

    def _get_extractor(self):
        """Lazy load GLiNER2 solo cuando se necesita."""
        if self.extractor is None:
            self.extractor = get_extractor(device="cpu")
        return self.extractor

    def process(self, file_path: str, doc_id: str = None):
        doc_id = doc_id or str(uuid4())
        logger.info(f"Processing document: {file_path}")

        # 1. Load
        docs = self.loader.load(file_path)
        full_text = "\n".join([doc.page_content for doc in docs])  # Texto completo para clasificación
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

        # 2. Clasificación del documento completo (NUEVO: Antes del split)
        extractor = self._get_extractor()
        doc_category = extractor.classify_document(full_text)
        logger.info(f"Document classified as: {doc_category}")

        # 3. Split
        chunks = self.splitter.split_documents(docs)

        # 4. NER Enrichment (por chunk)
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            entities = extractor.extract_entities(chunk.page_content)
            chunk.metadata["entities"] = entities
            chunk.metadata["doc_category"] = doc_category  
            enriched_chunks.append(chunk)
            logger.debug(f"Chunk {i}: Extracted {len(entities)} entity types")

        # 5. Embed (usa chunks enriquecidos)
        texts = [chunk.page_content for chunk in enriched_chunks]
        vectors = self.embedder.embed_documents(texts)

        # 6. Store con UUID válidos y metadata rica (incluye categoría)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "doc_id": doc_id,
                        "chunk_index": i
                    }
                }
            )
            for i, (chunk, vector) in enumerate(zip(enriched_chunks, vectors))
        ]
        self.vector_store.upsert(points)
        logger.success(f"Document {doc_id} ({doc_category}) processed and stored ({len(chunks)} chunks with GLiNER2 NER + classification)")
        return {"doc_id": doc_id, "chunks": len(chunks), "category": doc_category}