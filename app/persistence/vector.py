# app/persistence/vector.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings
from loguru import logger
import uuid

class QdrantManager:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection_name = settings.qdrant_collection
        self._initialized = False

    async def _ensure_collection(self):
        if self._initialized: return
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                logger.info(f"Collection '{self.collection_name}' created")
            self._initialized = True
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "409" in error_msg:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                self._initialized = True
            else:
                logger.error(f"Failed to check/create collection: {e}")
                raise e

    async def upsert(self, points: list[PointStruct]):
        await self._ensure_collection()
        valid_points = []
        for point in points:
            if not self._is_valid_uuid(point.id):
                point.id = str(uuid.uuid4())
            valid_points.append(point)
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=valid_points,
            wait=True
        )

    async def search(self, query_vector, limit=5, filter_dict=None):
        await self._ensure_collection()
        return await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_dict,
            with_payload=True
        )

    async def get_document_chunks(self, doc_id: str, user_id: str):
        await self._ensure_collection()
        filter_dict = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                ),
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        
        results, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_dict,
            limit=100, 
            with_payload=True,
            with_vectors=False
        )
        return results

    async def delete_document(self, doc_id: str, user_id: str):
        await self._ensure_collection()
        filter_dict = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                ),
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_dict
        )
        logger.info(f"Document {doc_id} deleted from Qdrant")

    @staticmethod
    def _is_valid_uuid(val):
        try:
            uuid.UUID(str(val))
            return True
        except (ValueError, AttributeError):
            return False