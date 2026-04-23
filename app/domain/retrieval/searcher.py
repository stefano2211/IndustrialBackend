import asyncio
from app.domain.proactiva.ingestion.embedder import Embedder
from app.persistence.vector import QdrantManager
from app.domain.retrieval.reranker import Reranker
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import Optional, Any
from loguru import logger

class SemanticSearcher:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = QdrantManager()
        self.reranker = Reranker()

    async def search(
        self, 
        query: str, 
        user_id: str,
        limit: Optional[int] = None, 
        knowledge_base_id: Optional[str] = None,
        session: Optional[Any] = None,
        min_score: Optional[float] = None,
    ):
        # Fetch dynamic settings once
        final_limit = 5
        if session:
            from app.persistence.proactiva.repositories.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            if limit is None:
                final_limit = system_settings.retrieval_search_results
        if limit is not None:
            final_limit = limit

        # 1. Embed Query en PARALELO (Dense + Sparse son operaciones independientes)
        # asyncio.gather los lanza simultaneamente → ~30% menos latencia en el paso de embedding
        query_dense, query_sparse = await asyncio.gather(
            self.embedder.embed_query(query),
            self.embedder.embed_sparse_query(query),
        )

        # 2. Build Filter
        conditions = [
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]
        if knowledge_base_id:
            conditions.append(
                FieldCondition(
                    key="metadata.knowledge_base_id",
                    match=MatchValue(value=knowledge_base_id)
                )
            )
        filter_dict = Filter(must=conditions) if conditions else None

        # 3. Hybrid Search with RRF (Initial Retrieval)
        # Fetch a pool of candidates (4x final_limit) to be refined by the Reranker.
        # NOTE: RRF fusion scores are rank-based (sum of 1/(k+rank)), not cosine similarity.
        # A min_score threshold does not apply here; the Reranker handles quality filtering.
        candidates_pool_size = max(20, final_limit * 4)

        # prefetch is a top-level query_points argument — FusionQuery only takes fusion=
        prefetch_list = [
            # Branch 1: Sparse (Keyword importance via SPLADE)
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist()
                ),
                using="sparse",
                limit=candidates_pool_size
            ),
            # Branch 2: Dense (Semantic context)
            qmodels.Prefetch(
                query=query_dense,
                using="dense",
                limit=candidates_pool_size
            ),
        ]
        fusion_query = qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)

        logger.debug(f"[Searcher] Executing hybrid query (RRF) for: {query[:50]}...")
        hits = await self.vector_store.search(
            query=fusion_query,
            prefetch=prefetch_list,
            limit=candidates_pool_size,
            filter_dict=filter_dict
        )

        if not hits:
            logger.warning(f"[Searcher] No results found in hybrid retrieval.")
            return []

        # 4. Format candidates for Reranking
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

        # 5. Rerank (Final refinement stage)
        # The reranker will select the best `final_limit` results out of the `candidates`
        results = await self.reranker.rerank(query, candidates, top_k=final_limit)

        logger.info(f"[Searcher] Final results after hybrid + rerank: {len(results)}")
        return results