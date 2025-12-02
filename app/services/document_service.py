from fastapi import UploadFile
import shutil
import os
import uuid
from app.storage.blob import minio_client
from app.worker.tasks import process_document_task
from app.storage.vector import QdrantManager
from loguru import logger

class DocumentService:
    def __init__(self):
        # Inyectamos dependencias si es necesario, por ahora instanciamos directo
        # Idealmente QdrantManager debería ser un singleton o inyectado
        self.qdrant = QdrantManager()
        self.upload_dir = "/tmp/uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_document(self, file: UploadFile):
        """Maneja la subida de archivos, almacenamiento en MinIO y encolado de tarea."""
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        temp_path = f"{self.upload_dir}/{safe_filename}"
        
        try:
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Subir a MinIO
            minio_client.upload_file(temp_path, safe_filename)
            
            # Encolar tarea de procesamiento
            task = process_document_task.delay(safe_filename, file.filename)
            
            return {
                "task_id": task.id, 
                "filename": file.filename, 
                "status": "en cola",
                "file_id": file_id
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_document_details(self, doc_id: str):
        """Recupera detalles y metadatos agregados de un documento."""
        chunks = self.qdrant.get_document_chunks(doc_id)
        
        if not chunks:
            return None
        
        # Agregamos metadatos del primer chunk
        first_chunk = chunks[0]
        category = first_chunk.payload.get("metadata", {}).get("doc_category", "unknown")
        filename = first_chunk.payload.get("metadata", {}).get("source", "unknown")
        
        # Consolidar entidades únicas de todos los chunks
        all_entities = {}
        for chunk in chunks:
            entities = chunk.payload.get("metadata", {}).get("entities", {})
            for label, values in entities.items():
                if label not in all_entities:
                    all_entities[label] = set()
                # values puede ser lista o string dependiendo de cómo se guardó, aseguramos lista
                if isinstance(values, list):
                    all_entities[label].update(values)
                elif isinstance(values, str):
                    all_entities[label].add(values)
                
        # Convertir sets a listas para JSON
        consolidated_entities = {k: list(v) for k, v in all_entities.items()}
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "category": category,
            "total_chunks": len(chunks),
            "entities": consolidated_entities
        }

    def delete_document(self, doc_id: str):
        """Elimina un documento de Qdrant."""
        self.qdrant.delete_document(doc_id)
        return {"status": "deleted", "doc_id": doc_id}

    def get_task_status(self, task_id: str):
        """Obtiene el estado de una tarea de Celery."""
        from celery.result import AsyncResult
        from app.worker.tasks import celery_app
        
        task = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": task.state,
            "info": task.info if task.info else None
        }
