"""Celery worker tasks for async document processing."""

import asyncio
import os

from celery import Celery
from loguru import logger

from app.core.config import settings
from app.domain.ingestion.pipeline import DocumentProcessor
from app.persistence.blob import minio_client

celery_app = Celery(
    "aura",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

processor = DocumentProcessor()


@celery_app.task(bind=True, name="process_document")
def process_document_task(
    self,
    object_key: str,
    filename: str,
    user_id: str,
    doc_id: str = None,
    knowledge_base_id: str = None,
):
    """Download file from MinIO, process through ingestion pipeline, and store vectors."""
    try:
        self.update_state(
            state="PROGRESS", meta={"status": "descargando", "filename": filename}
        )
        local_path = minio_client.download_file(object_key)

        self.update_state(
            state="PROGRESS", meta={"status": "procesando", "filename": filename}
        )
        result = asyncio.run(
            processor.process_file(
                local_path,
                user_id=user_id,
                doc_id=doc_id,
                knowledge_base_id=knowledge_base_id,
            )
        )

        os.unlink(local_path)
        source_url = minio_client.get_presigned_url(object_key)

        return {
            "status": "completado",
            "doc_id": result["doc_id"],
            "filename": filename,
            "source_url": source_url,
        }
    except Exception as e:
        logger.error(f"Error en tarea: {e}")
        raise


celery = celery_app