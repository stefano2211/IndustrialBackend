"""
VL MLOps Service — OTA Update para modelos Vision-Language
===========================================================

Con Qwen3.5 unificado multimodal + vLLM, descargamos el LoRA VL como .tar.gz
y lo extraemos directamente en la carpeta dinámica de vLLM (/loras).
"""

import hashlib
import os
import asyncio
import httpx
from loguru import logger
import tarfile

from app.core.config import settings


class VLMLOpsService:
    """
    Gestiona el ciclo OTA para modelos Vision-Language (Qwen2.5-VL fine-tuned).
    Descarga el artefacto ZIP unificado conteniendo safetensors de LoRA para inyectar en vLLM.
    """

    async def process_vl_ota_webhook(
        self,
        model_tag: str,
        mmproj_tag: str = None, # Deprecated with vLLM
        tenant_id: str = "aura_tenant_01",
        download_url: str = None,
    ):
        """
        Procesa el webhook de modelo VL listo desde ApiLLMOps.

        Args:
            model_tag:  Tag del modelo principal (ej: "aura_tenant_01-vl")
            mmproj_tag: (Deprecated) Tag del vision projector.
            tenant_id:  ID del tenant
        """
        logger.info(f"[VL OTA] 🚀 Iniciando actualización OTA VL para vLLM: {model_tag}")

        tar_path = f"/tmp/{model_tag}.tar.gz"
        lora_base_dir = f"/loras/{model_tag}"
        hash_flag = f"/loras/{model_tag}/.artifact_hash"

        try:
            tar_url = download_url
            if not tar_url:
                raise ValueError("No se proporcionó una download_url en el webhook y el fallback no está configurado.")

            # Descargar el .tar.gz en streaming
            logger.info("[VL OTA] Descargando Tarball LoRA VL...")
            async with httpx.AsyncClient(timeout=3600.0) as client:
                 async with client.stream("GET", tar_url) as r:
                    r.raise_for_status()
                    with open(tar_path, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
            
            logger.info(f"[VL OTA] Tarball descargado en: {tar_path}")

            # Idempotencia por hash: saltar extraccion si el artefacto no cambio
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
                logger.info(f"[VL OTA] Mismo artefacto (hash={current_hash[:8]}). Solo notificando a vLLM.")
            else:
                logger.info(f"[VL OTA] Artefacto nuevo o primera instalación (hash={current_hash[:8]}). Extrayendo...")
                def _extract_tar():
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(path="/loras", filter="data")
                await asyncio.to_thread(_extract_tar)
                
                # Garantizar que el directorio dinámico exista antes de escribir el hash
                os.makedirs(lora_base_dir, exist_ok=True)
                with open(hash_flag, "w") as _hf:
                    _hf.write(current_hash)
            
            logger.success(f"[VL OTA] Adaptador VL '{model_tag}' extraído en '{lora_base_dir}'.")

            # --- BUG 7 fix: Notificar a vLLM para cargar el adaptador VL dinámicamente ---
            vllm_base = settings.vllm_base_url.rstrip("/")
            vllm_host = vllm_base.removesuffix("/v1")
            lora_path_in_container = f"/loras/{model_tag}"

            logger.info(f"[VL OTA] Notificando a vLLM para cargar adaptador VL: {model_tag}")
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{vllm_host}/v1/load_lora_adapter",
                        json={
                            "lora_name": model_tag,
                            "lora_path": lora_path_in_container,
                            "load_inplace": True,
                        },
                    )
                    if resp.status_code == 200:
                        logger.success(f"[VL OTA] vLLM cargó el adaptador VL '{model_tag}' correctamente.")
                    else:
                        logger.warning(f"[VL OTA] vLLM respondió {resp.status_code}: {resp.text}")
            except Exception as vllm_err:
                logger.error(f"[VL OTA] Error notificando a vLLM VL (pesos en disco): {vllm_err}")

        except Exception as e:
            logger.error(f"[VL OTA] ❌ Error en OTA VL: {e}")
            raise
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)
            logger.info("[VL OTA] Archivos temporales limpiados.")
            
        return {"status": "success", "tag": model_tag}
