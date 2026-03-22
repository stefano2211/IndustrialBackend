from app.domain.ingestion.embedder import Embedder
from app.persistence.vector import QdrantManager
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import Optional, Any

class SemanticSearcher:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = QdrantManager()

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
        final_min_score = 0.45  # Default score threshold — filters out irrelevant chunks
        if session:
            from app.persistence.repositories.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            if limit is None:
                final_limit = system_settings.retrieval_search_results
            if min_score is None and hasattr(system_settings, 'retrieval_min_score'):
                final_min_score = system_settings.retrieval_min_score
        if limit is not None:
            final_limit = limit
        if min_score is not None:
            final_min_score = min_score

        query_vector = self.embedder.embed_query(query)
        
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
        
        results = await self.vector_store.search(
            query_vector, 
            limit=final_limit, 
            filter_dict=filter_dict
        )

        from loguru import logger
        logger.debug(f"Search results for query '{query}': {len(results)} hits found.")
        filtered_results = []
        for i, hit in enumerate(results):
            source = hit.payload.get("metadata", {}).get("source", "unknown")
            logger.debug(f"Hit {i}: score={hit.score:.4f}, source={source}")
            if hit.score >= final_min_score:
                filtered_results.append(hit)
            else:
                logger.debug(f"Hit {i} dropped (score {hit.score:.4f} < min {final_min_score})")

        logger.debug(f"After score filter ({final_min_score}): {len(filtered_results)}/{len(results)} hits kept")

        return [
            {
                "text": hit.payload["text"],
                "score": float(hit.score),
                "metadata": {
                    **hit.payload.get("metadata", {}),
                    "source": hit.payload.get("metadata", {}).get("source", "documento"),
                    "section": hit.payload.get("metadata", {}).get("section", ""),
                },
            }
            for hit in filtered_results
        ]