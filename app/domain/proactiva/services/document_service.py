"""Document service — Business logic for document upload, retrieval, and deletion."""

import asyncio
import os
import shutil
import uuid
from typing import Optional

from fastapi import UploadFile, BackgroundTasks
from loguru import logger

from app.persistence.blob import minio_client
from app.persistence.vector import QdrantManager
from app.domain.proactiva.ingestion.pipeline import DocumentProcessor


def _write_bytes(path: str, data: bytes) -> None:
    """Synchronous helper for writing bytes to disk, intended for asyncio.to_thread."""
    with open(path, "wb") as f:
        f.write(data)


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
        self.processor = DocumentProcessor(vector_store=self.qdrant)
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_document(
        self,
        file: UploadFile,
        user_id: str,
        knowledge_base_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> dict:
        """Upload file to MinIO and enqueue processing task if background_tasks is supplied."""
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        temp_path = os.path.join(self.upload_dir, safe_filename)

        # Read async, then write in thread to avoid blocking the event loop
        content = await file.read()
        await asyncio.to_thread(_write_bytes, temp_path, content)

        minio_client.upload_file(temp_path, safe_filename)

        if background_tasks:
            background_tasks.add_task(
                self._process_and_cleanup,
                temp_path,
                safe_filename,
                user_id,
                file_id,
                knowledge_base_id
            )
            return {
                "task_id": file_id,
                "filename": file.filename,
                "status": "procesando", # Not blocks the HTTP response
                "file_id": file_id,
            }
        else:
            await self._process_and_cleanup(temp_path, safe_filename, user_id, file_id, knowledge_base_id)
            return {
                "task_id": file_id,
                "filename": file.filename,
                "status": "completado",
                "file_id": file_id,
            }

    async def _process_and_cleanup(self, temp_path: str, minio_filename: str, user_id: str, file_id: str, knowledge_base_id: Optional[str]):
        try:
            await self.processor.process_file(
                temp_path,
                user_id=user_id,
                doc_id=file_id,
                knowledge_base_id=knowledge_base_id,
            )
        except Exception as e:
            logger.error(f"Error processing document {file_id}: {e}")
            try:
                minio_client.delete_file(minio_filename)
                logger.info(f"Cleaned up MinIO object '{minio_filename}' after processing failure")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up MinIO object '{minio_filename}': {cleanup_err}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def get_document_details(self, doc_id: str, user_id: str) -> Optional[dict]:
        """Retrieve aggregated details and metadata for a processed document."""
        chunks = await self.qdrant.get_document_chunks(doc_id, user_id=user_id)
        if not chunks:
            return None

        first_chunk = chunks[0]
        category = first_chunk.payload.get("metadata", {}).get("doc_category", "unknown")
        filename = first_chunk.payload.get("metadata", {}).get("source", "unknown")

        chunks.sort(key=lambda x: x.payload.get("metadata", {}).get("chunk_index", 0))
        full_text = "\n\n".join(chunk.payload.get("text", "") for chunk in chunks)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "category": category,
            "total_chunks": len(chunks),
            "content": full_text,
        }

    async def delete_document(self, doc_id: str, user_id: str) -> dict:
        """Delete all chunks for a document from Qdrant."""
        await self.qdrant.delete_document(doc_id, user_id=user_id)
        return {"status": "deleted", "doc_id": doc_id}

    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """Simulate status for frontend polling on synchronous operations."""
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "info": {"status": "completado"}
        }
