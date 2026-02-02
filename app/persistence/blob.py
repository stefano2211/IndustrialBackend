from minio import Minio
from app.config import settings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import tempfile
from datetime import timedelta


class MinIOClient:
    def __init__(self):
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket()

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.success(f"Bucket '{self.bucket}' creado")
        else:
            logger.info(f"Bucket '{self.bucket}' ya existe")

    def upload_file(self, local_path: str, object_name: str):
        """Sube un archivo local a MinIO"""
        self.client.fput_object(self.bucket, object_name, local_path)
        logger.info(f"Archivo subido: {object_name}")

    def download_file(self, object_name: str) -> str:
        """Descarga un archivo de MinIO y lo guarda en un archivo temporal."""

        _, ext = os.path.splitext(object_name)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = temp_file.name
        temp_file.close()
        self.client.fget_object(self.bucket, object_name, temp_path)
        logger.info(f"Archivo descargado con extensión: {temp_path}")

        return temp_path

    def get_presigned_url(self, object_name: str, expires: int = 3600) -> str:
        """Genera una URL firmada para descargar el archivo"""
        return self.client.presigned_get_object(self.bucket, object_name, expires=timedelta(seconds=expires))

    def delete_file(self, object_name: str):
        """Elimina un archivo del bucket (opcional)"""
        self.client.remove_object(self.bucket, object_name)



minio_client = MinIOClient()