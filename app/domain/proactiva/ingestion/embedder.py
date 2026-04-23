import asyncio
from fastembed import TextEmbedding, SparseTextEmbedding
from app.core.config import settings
from typing import List

class Embedder:
    def __init__(self):
        # BGE-small-en-v1.5 or the one in settings (Dense)
        self.dense_model = TextEmbedding(model_name=settings.embedding_model)
        # SPLADE (Sparse)
        self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_embedding_model)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings densos."""
        return await asyncio.to_thread(
            lambda: [e.tolist() for e in self.dense_model.embed(texts)]
        )

    async def embed_sparse_documents(self, texts: List[str]):
        """Genera embeddings dispersos (SPLADE)."""
        return await asyncio.to_thread(
            lambda: list(self.sparse_model.embed(texts))
        )

    async def embed_query(self, text: str) -> List[float]:
        """Genera un embedding denso para una consulta."""
        return await asyncio.to_thread(
            lambda: next(self.dense_model.embed([text])).tolist()
        )

    async def embed_sparse_query(self, text: str):
        """Genera un embedding disperso para una consulta."""
        return await asyncio.to_thread(
            lambda: next(self.sparse_model.embed([text]))
        )