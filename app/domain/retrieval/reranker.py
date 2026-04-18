import asyncio
from sentence_transformers import CrossEncoder
from app.core.config import settings
from typing import List, Dict
from loguru import logger

class Reranker:
    """
    Second-stage reranker using Cross-Encoders.
    Calculates exact query-document interaction for superior precision.
    """
    def __init__(self):
        logger.info(f"Initializing Reranker with model: {settings.reranker_model}")
        try:
            self.model = CrossEncoder(settings.reranker_model)
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}. RAG will fallback to initial scores.")
            self.model = None

    async def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Reranks documents based on actual relevance to the query.
        """
        if not documents or not self.model:
            return documents[:top_k]

        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")

        # Prepare pairs: [query, document_text]
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Cross-encoder inference (scoring) — offloaded to thread to avoid blocking event loop
        # Returns raw scores (usually higher is more relevant)
        scores = await asyncio.to_thread(self.model.predict, pairs)
        
        # Update scores and track reranking impact
        for i, score in enumerate(scores):
            documents[i]["original_score"] = documents[i].get("score", 0.0)
            documents[i]["score"] = float(score) # Primary score is now the reranked one

        # Sort by the new score
        reranked = sorted(documents, key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Reranking complete. Top score: {reranked[0]['score']:.4f}")
        return reranked[:top_k]
