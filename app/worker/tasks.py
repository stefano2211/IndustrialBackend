from celery import Celery
from app.ingestion.pipeline import DocumentProcessor
from app.storage.blob import minio_client
from app.config import settings
import os
from loguru import logger

celery_app = Celery(
    "aura",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"]
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
def process_document_task(self, object_key: str, filename: str):
    try:
        self.update_state(state="PROGRESS", meta={"status": "descargando", "filename": filename})
        local_path = minio_client.download_file(object_key)
        self.update_state(state="PROGRESS", meta={"status": "procesando", "filename": filename})
        result = processor.process(local_path)
        os.unlink(local_path)
        source_url = minio_client.get_presigned_url(object_key)
        return {
            "status": "completado",
            "doc_id": result["doc_id"],
            "filename": filename,
            "source_url": source_url
        }
    except Exception as e:
        logger.error(f"Error en tarea: {e}")
        raise

celery = celery_app