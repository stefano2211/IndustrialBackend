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
        session: Optional[Any] = None
    ):
        # Fetch dynamic settings
        final_limit = 5
        if limit is not None:
            final_limit = limit
        elif session:
            from app.persistence.repositories.settings_repository import SettingsRepository
            repo = SettingsRepository(session)
            system_settings = await repo.get_settings()
            final_limit = system_settings.retrieval_search_results

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
        
        results = self.vector_store.search(
            query_vector, 
            limit=final_limit, 
            filter_dict=filter_dict
        )

        return [
            {
                "text": hit.payload["text"],
                "score": float(hit.score),
                "metadata": hit.payload["metadata"],  
            }
            for hit in results
        ]