# app/core/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings
from loguru import logger
import uuid

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection_name = settings.qdrant_collection
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        try:
            # Intentar crear siempre; si existe, qdrant-client o la API lanzarán error
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created")
        except Exception as e:
            # Capturamos excepciones genéricas porque qdrant-client puede variar (UnexpectedResponse, RpcError)
            # Si el error indica que ya existe, lo ignoramos.
            error_msg = str(e).lower()
            if "already exists" in error_msg or "409" in error_msg:
                logger.info(f"Collection '{self.collection_name}' already exists.")
            else:
                logger.error(f"Failed to check/create collection: {e}")
                # No lanzamos raise para permitir que el servicio continúe si es solo un tema de conexión temporal,
                # aunque idealmente debería reintentarse.
                raise e

    def upsert(self, points: list[PointStruct]):
        # Aseguramos que todos los IDs sean UUID válidos
        valid_points = []
        for point in points:
            if not self._is_valid_uuid(point.id):
                # Generamos un UUID limpio
                point.id = str(uuid.uuid4())
            valid_points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=valid_points,
            wait=True
        )

    def search(self, query_vector, limit=5, filter_dict=None):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_dict,
            with_payload=True
        )

    def get_document_chunks(self, doc_id: str):
        """Recupera todos los chunks de un documento específico."""
        
        filter_dict = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
        

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_dict,
            limit=100, 
            with_payload=True,
            with_vectors=False
        )
        return results

    def delete_document(self, doc_id: str):
        """Elimina todos los chunks de un documento."""
        
        filter_dict = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
        
        self.client.delete(
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