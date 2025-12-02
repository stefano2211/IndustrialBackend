from app.ingestion.embedder import Embedder
from app.storage.vector import QdrantManager
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import Optional

class SemanticSearcher:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = QdrantManager()

    def search(
        self, 
        query: str, 
        limit: int = 5, 
        entity_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ):
        query_vector = self.embedder.embed_query(query)
        
        conditions = []
        if entity_filter:
            conditions.append(
                FieldCondition(
                    key=f"metadata.entities.{entity_filter}",
                    match=MatchValue(value=True)  
                )
            )
        if category_filter:
            conditions.append(
                FieldCondition(
                    key="metadata.doc_category",
                    match=MatchValue(value=category_filter)
                )
            )
        
        filter_dict = Filter(must=conditions) if conditions else None
        
        results = self.vector_store.search(
            query_vector, 
            limit=limit, 
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