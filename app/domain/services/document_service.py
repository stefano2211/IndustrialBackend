"""Document service — Business logic for document upload, retrieval, and deletion."""

import os
import shutil
import uuid
from typing import Optional

from fastapi import UploadFile
from loguru import logger

from app.persistence.blob import minio_client
from app.persistence.vector import QdrantManager
from app.worker.tasks import process_document_task


class DocumentService:
    """
    Handles document lifecycle: upload → process → retrieve → delete.

    Dependencies are injected via constructor for testability.
    """

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        upload_dir: str = "/tmp/uploads",
    ):
        self.qdrant = qdrant or QdrantManager()
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_document(
        self,
        file: UploadFile,
        user_id: str,
        knowledge_base_id: Optional[str] = None,
    ) -> dict:
        """Upload file to MinIO and enqueue processing task."""
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        temp_path = os.path.join(self.upload_dir, safe_filename)

        try:
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            minio_client.upload_file(temp_path, safe_filename)

            task = process_document_task.delay(
                safe_filename,
                file.filename,
                user_id=user_id,
                doc_id=file_id,
                knowledge_base_id=knowledge_base_id,
            )

            return {
                "task_id": task.id,
                "filename": file.filename,
                "status": "en cola",
                "file_id": file_id,
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_document_details(self, doc_id: str, user_id: str) -> Optional[dict]:
        """Retrieve aggregated details and metadata for a processed document."""
        chunks = self.qdrant.get_document_chunks(doc_id, user_id=user_id)
        if not chunks:
            return None

        first_chunk = chunks[0]
        category = first_chunk.payload.get("metadata", {}).get("doc_category", "unknown")
        filename = first_chunk.payload.get("metadata", {}).get("source", "unknown")

        # Consolidate unique entities from all chunks
        all_entities: dict[str, set] = {}
        for chunk in chunks:
            entities = chunk.payload.get("metadata", {}).get("entities", {})
            for label, values in entities.items():
                if label not in all_entities:
                    all_entities[label] = set()
                if isinstance(values, list):
                    all_entities[label].update(values)
                elif isinstance(values, str):
                    all_entities[label].add(values)

        consolidated_entities = {k: list(v) for k, v in all_entities.items()}

        chunks.sort(key=lambda x: x.payload.get("metadata", {}).get("chunk_index", 0))
        full_text = "\n\n".join(chunk.payload.get("text", "") for chunk in chunks)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "category": category,
            "total_chunks": len(chunks),
            "content": full_text,
            "entities": consolidated_entities,
        }

    def delete_document(self, doc_id: str, user_id: str) -> dict:
        """Delete all chunks for a document from Qdrant."""
        self.qdrant.delete_document(doc_id, user_id=user_id)
        return {"status": "deleted", "doc_id": doc_id}

    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """Get Celery task status."""
        from celery.result import AsyncResult
        from app.worker.tasks import celery_app

        task = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": task.state,
            "info": task.info if task.info else None,
        }
