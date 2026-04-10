import httpx
import os
from typing import Optional
from loguru import logger
from app.core.config import settings

class MothershipClient:
    """
    Cliente HTTP encargado de la sincronización de MLOps.
    Transfiere el Dataset curado (Replay Buffer) hacia la ApiLLMOps (Cloud/Central)
    para disparar el entrenamiento intensivo en GPUs.
    """
    
    def __init__(self):
        self.base_url = settings.mothership_api_url.rstrip("/")
        self.api_key = settings.mothership_api_key
        
        self.headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json"
        }

    async def upload_dataset(self, file_path: str, tenant_id: str = "aura_tenant_01", tool_name: Optional[str] = None) -> bool:
        """Sube el archivo .jsonl al DataLake de la Mothership"""
        url = f"{self.base_url}/api/v1/datasets/upload"
        
        # Nombre de archivo dinámico: {tenant_id}_{tool_name}.jsonl
        # Si no hay tool_name, cae al tradicional master.jsonl
        suffix = f"_{tool_name}" if tool_name else "_master"
        remote_filename = f"{tenant_id}{suffix}.jsonl"
        
        logger.info(f"[Mothership Client] Subiendo Dataset a la nube ({remote_filename}): {self.base_url} ...")
        
        if not os.path.exists(file_path):
            logger.error(f"[Mothership Client] Archivo no encontrado: {file_path}")
            return False
            
        try:
            with open(file_path, "rb") as f:
                files = {"file": (remote_filename, f, "application/jsonlines")}
                data = {"tenant_id": tenant_id}
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=self.headers, data=data, files=files)
                    
                    if response.status_code == 200:
                        logger.success("[Mothership Client] Dataset subido exitosamente a la nube.")
                        return True
                    else:
                        logger.error(f"[Mothership Client] Error subiendo dataset: {response.text}")
                        return False
        except Exception as e:
            logger.error(f"[Mothership Client] Excepción de conexión: {e}")
            return False

    async def upload_vl_dataset(
        self,
        file_path: str,
        tenant_id: str = "aura_tenant_01",
        tool_name: str = "computer_use",
    ) -> bool:
        """
        Sube un dataset VL (.jsonl con screenshots+acciones) al datalake-vl de ApiLLMOps.
        Endpoint: POST /api/v1/vl/upload
        """
        url = f"{self.base_url}/api/v1/vl/upload"
        remote_filename = f"{tenant_id}_vl_{tool_name}.jsonl"

        logger.info(f"[Mothership Client] Subiendo Dataset VL ({remote_filename}) a {self.base_url}...")

        if not os.path.exists(file_path):
            logger.error(f"[Mothership Client] Archivo VL no encontrado: {file_path}")
            return False

        try:
            with open(file_path, "rb") as f:
                files = {"file": (remote_filename, f, "application/jsonlines")}
                data = {"tenant_id": tenant_id, "tool_name": tool_name}

                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=self.headers, data=data, files=files)

                    if response.status_code == 200:
                        logger.success("[Mothership Client] Dataset VL subido exitosamente.")
                        return True
                    else:
                        logger.error(f"[Mothership Client] Error subiendo dataset VL: {response.text}")
                        return False
        except Exception as e:
            logger.error(f"[Mothership Client] Excepción subiendo dataset VL: {e}")
            return False

    async def trigger_training_job(
        self,
        tenant_id: str = "aura_tenant_01",
        epochs: int = 3,
        webhook_url: Optional[str] = None,
    ) -> bool:
        """
        Dispara el pipeline de fine-tuning de texto (LoRA) en el Celery Worker de ApiLLMOps.
        Endpoint: POST /api/v1/training/job
        """
        url = f"{self.base_url}/api/v1/training/job"

        if not webhook_url:
            webhook_url = f"{settings.edge_public_url.rstrip('/')}/mlops/webhook/model-ready"

        payload = {
            "tenant_id": tenant_id,
            "epochs": epochs,
            "webhook_url": webhook_url,
        }

        logger.info(f"[Mothership Client] Disparando training de texto para {tenant_id} → {url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=self.headers, json=payload)

                if response.status_code == 200:
                    job_id = response.json().get("job_id", "?")
                    logger.success(f"[Mothership Client] Training de texto encolado. JobID: {job_id}")
                    return True
                else:
                    logger.error(f"[Mothership Client] Error disparando training de texto: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"[Mothership Client] Excepción disparando training de texto: {e}")
            return False

    async def trigger_vl_training_job(
        self,
        tenant_id: str = "aura_tenant_01",
        vl_epochs: int = 2,
        text_epochs: int = 1,
        webhook_url: Optional[str] = None,
    ) -> bool:
        """
        Dispara el pipeline VL unificado (2 fases) en el Celery Worker de ApiLLMOps.
        Endpoint: POST /api/v1/vl/training/job
        """
        url = f"{self.base_url}/api/v1/vl/training/job"

        if not webhook_url:
            webhook_url = f"{settings.edge_public_url.rstrip('/')}/mlops/webhook/model-ready"

        payload = {
            "tenant_id": tenant_id,
            "base_model": "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",  # 3B: edge-friendly (2.5 GB VRAM inferencia)
            "vl_epochs": vl_epochs,
            "text_epochs": text_epochs,
            "webhook_url": webhook_url,
        }

        logger.info(f"[Mothership Client] Disparando training VL para {tenant_id} → {url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=self.headers, json=payload)

                if response.status_code == 200:
                    job_id = response.json().get("job_id", "?")
                    logger.success(f"[Mothership Client] Training VL encolado. JobID: {job_id}")
                    return True
                else:
                    logger.error(f"[Mothership Client] Error disparando training VL: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"[Mothership Client] Excepción disparando training VL: {e}")
            return False


mothership_client = MothershipClient()
