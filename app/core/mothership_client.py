import httpx
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

    async def upload_dataset(self, file_path: str, tenant_id: str = "aura_tenant_01") -> bool:
        """Sube el archivo .jsonl al DataLake de la Mothership"""
        url = f"{self.base_url}/api/v1/datasets/upload"
        
        logger.info(f"[Mothership Client] Subiendo Dataset a la nube: {self.base_url} ...")
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (f"{tenant_id}_master.jsonl", f, "application/jsonlines")}
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

    async def trigger_training_job(self, tenant_id: str = "aura_tenant_01", epochs: int = 3, webhook_url: Optional[str] = None) -> bool:
        """Dispara el job de Fine-Tuning de Qwen en los workers de la nube"""
        url = f"{self.base_url}/api/v1/training/job"
        
        payload = {
            "tenant_id": tenant_id,
            "base_model": "unsloth/qwen2.5-7b-bnb-4bit",
            "epochs": epochs,
            "webhook_url": webhook_url
        }
        
        logger.info(f"[Mothership Client] Desencadenando entrenamiento LoRA en la nube: {url} ...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=self.headers, json=payload)
                
                if response.status_code == 200:
                    logger.success(f"[Mothership Client] Entrenamiento disparado. JobID: {response.json().get('job_id')}")
                    return True
                else:
                    logger.error(f"[Mothership Client] Error disparando entrenamiento: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"[Mothership Client] Excepción desencadenando Job: {e}")
            return False

mothership_client = MothershipClient()
