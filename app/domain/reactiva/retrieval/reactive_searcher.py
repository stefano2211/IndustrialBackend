"""Semantic searcher for the reactive domain — isolated Qdrant collection.

Reuses the shared Embedder (nomic + SPLADE) and Reranker (BGE) but queries
the `reactive_documents` collection. Filters by tenant_id instead of user_id.
"""

import asyncio
from typing import Optional, Any

from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from loguru import logger

from app.domain.shared.ingestion.embedder import Embedder
from app.persistence.reactiva.reactive_vector import ReactiveQdrantManager
from app.domain.shared.retrieval.reranker import Reranker


class ReactiveSemanticSearcher:
    """Hybrid semantic search for the reactive domain's knowledge base."""

    def __init__(self):
        self.embedder = Embedder()                    # Shared — same embedding models
        self.vector_store = ReactiveQdrantManager()    # Isolated — reactive collection
        self.reranker = Reranker()                     # Shared — same reranker model

    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        limit: Optional[int] = None,
        knowledge_base_id: Optional[str] = None,
        session: Optional[Any] = None,
        min_score: Optional[float] = None,
    ):
        """Search the reactive knowledge base with Hybrid Dense+Sparse → RRF → Rerank."""
        final_limit = limit or 5

        if session:
            from app.persistence.shared.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            if limit is None:
                final_limit = system_settings.retrieval_search_results

        # 1. Embed query in parallel (Dense + Sparse)
        query_dense, query_sparse = await asyncio.gather(
            self.embedder.embed_query(query),
            self.embedder.embed_sparse_query(query),
        )

        # 2. Build filter — scoped by tenant, not user
        conditions = [
            FieldCondition(
                key="metadata.tenant_id",
                match=MatchValue(value=tenant_id),
            )
        ]
        if knowledge_base_id:
            conditions.append(
                FieldCondition(
                    key="metadata.knowledge_base_id",
                    match=MatchValue(value=knowledge_base_id),
                )
            )
        filter_dict = Filter(must=conditions) if conditions else None

        # 3. Hybrid Search with RRF
        candidates_pool_size = max(20, final_limit * 4)
        prefetch_list = [
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist(),
                ),
                using="sparse",
                limit=candidates_pool_size,
            ),
            qmodels.Prefetch(
                query=query_dense,
                using="dense",
                limit=candidates_pool_size,
            ),
        ]
        fusion_query = qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)

        logger.debug(f"[ReactiveSearcher] Executing hybrid query (RRF) for: {query[:50]}...")
        hits = await self.vector_store.search(
            query=fusion_query,
            prefetch=prefetch_list,
            limit=candidates_pool_size,
            filter_dict=filter_dict,
        )

        if not hits:
            logger.warning("[ReactiveSearcher] No results found in reactive collection.")
            return []

        # 4. Format candidates
        candidates = [
            {
                "text": hit.payload["text"],
                "score": float(hit.score),
                "metadata": {
                    **hit.payload.get("metadata", {}),
                    "source": hit.payload.get("metadata", {}).get("source", "documento"),
                    "section": hit.payload.get("metadata", {}).get("section", ""),
                },
            }
            for hit in hits
        ]

        # 5. Rerank
        results = await self.reranker.rerank(query, candidates, top_k=final_limit)
        logger.info(f"[ReactiveSearcher] Final results after hybrid + rerank: {len(results)}")
        return results
