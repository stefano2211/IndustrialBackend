import os
import tarfile
import httpx
import hashlib
import asyncio
from loguru import logger

from app.core.config import settings


class MLOpsService:
    """
    MLOps Service — gestiona actualizaciones OTA de modelos desde la Mothership (ApiLLMOps).

    Responsabilidad única: recibir webhook del Worker → descargar tar.gz LoRA → extraer en /loras/ → notificar vLLM.
    La recolección de datos es responsabilidad exclusiva del DB Collector.
    """

    async def process_ota_webhook(self, new_model_tag: str, tenant_id: str = "aura_tenant_01", download_url: str = None):
        """
        Recibe una download_url pre-firmada desde el webhook del Worker.
        1. Descarga el artefacto tar.gz en streaming al /tmp/.
        2. Extrae el adaptador LoRA a /loras/ (volumen compartido con vLLM).
        3. Notifica a vLLM para carga dinámica sin reinicio.
        """
        logger.info(f"[MLOps OTA] Iniciando actualización OTA para adaptador LoRA: {new_model_tag}")

        tar_path = f"/tmp/{new_model_tag}.tar.gz"
        lora_base_dir = f"/loras/{new_model_tag}"
        hash_flag = f"/loras/{new_model_tag}/.artifact_hash"

        try:
            tar_url = download_url
            if not tar_url:
                raise ValueError("No se proporcionó una download_url en el webhook y el fallback no está configurado.")

            logger.info("[MLOps OTA] Iniciando descarga directa del adaptador por Push-Trigger...")

            # --- Descargar el .tar.gz en streaming ---
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream("GET", tar_url) as r:
                    r.raise_for_status()
                    with open(tar_path, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                logger.info(f"[MLOps OTA] Tarball descargado en: {tar_path}")

            # Idempotencia por hash: si el artefacto no cambio (mismo ciclo/retry), saltar extraccion
            _md5 = hashlib.md5()
            with open(tar_path, "rb") as _f:
                for _chunk in iter(lambda: _f.read(8 * 1024 * 1024), b""):
                    _md5.update(_chunk)
            current_hash = _md5.hexdigest()
            os.makedirs("/loras", exist_ok=True)
            prev_hash = None
            if os.path.exists(hash_flag):
                with open(hash_flag, "r") as _hf:
                    prev_hash = _hf.read().strip()
            
            if current_hash == prev_hash:
                logger.info(f"[MLOps OTA] Mismo artefacto (hash={current_hash[:8]}). Solo notificando a vLLM.")
            else:
                logger.info(f"[MLOps OTA] Artefacto nuevo o primera instalación (hash={current_hash[:8]}). Extrayendo...")
                def _extract_tar():
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(path="/loras", filter="data")
                await asyncio.to_thread(_extract_tar)
                
                # Garantizar que el directorio dinámico exista antes de escribir el hash
                os.makedirs(lora_base_dir, exist_ok=True)
                with open(hash_flag, "w") as _hf:
                    _hf.write(current_hash)
            
            logger.success(f"[MLOps OTA] Adaptador '{new_model_tag}' extraído en '{lora_base_dir}'.")

            # Notificar a vLLM para que cargue el adaptador dinámicamente
            # Requiere VLLM_ALLOW_RUNTIME_LORA_UPDATING=true en el contenedor vLLM
            vllm_base = settings.vllm_base_url.rstrip("/")  # e.g. http://vllm:8000/v1
            vllm_host = vllm_base.removesuffix("/v1")       # e.g. http://vllm:8000
            lora_path_in_container = f"/loras/{new_model_tag}"
            
            logger.info(f"[MLOps OTA] Notificando a vLLM para cargar adaptador: {new_model_tag} desde {lora_path_in_container}")
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{vllm_host}/v1/load_lora_adapter",
                        json={
                            "lora_name": new_model_tag,
                            "lora_path": lora_path_in_container,
                        },
                    )
                    if resp.status_code == 200:
                        logger.success(f"[MLOps OTA] vLLM cargó el adaptador '{new_model_tag}' correctamente.")
                    else:
                        logger.warning(f"[MLOps OTA] vLLM respondió {resp.status_code} al cargar LoRA: {resp.text}")
            except Exception as vllm_err:
                # No fallar el OTA completo si vLLM no responde — los pesos ya están en disco
                logger.error(f"[MLOps OTA] Error notificando a vLLM (pesos en disco): {vllm_err}")

        except Exception as e:
            import traceback
            logger.error(f"[MLOps OTA] Excepción durante el proceso OTA [{type(e).__name__}]: {str(e)}\n{traceback.format_exc()}")
            return {"status": "error", "tag": new_model_tag, "detail": str(e)}
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)
            logger.info("[MLOps OTA] Archivos temporales limpiados.")

        return {"status": "success", "tag": new_model_tag}

