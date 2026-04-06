"""
VL MLOps Service — OTA Update para modelos Vision-Language
===========================================================

Con la transición a vLLM y Qwen2.5-VL unificado, ya no requerimos descargar model.gguf 
y mmproj.mmproj por separado ni interactuar con la Blob API de Ollama. 
Los pesos LoRA para el sistema visual se descargan en el mismo `.tar.gz` 
y se extraen directamente en la carpeta dinámica de vLLM (/loras).
"""

import os
import asyncio
import httpx
from loguru import logger
import tarfile

from app.core.config import settings
from app.core.mothership_client import mothership_client


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
        lora_base_dir = f"./loras/{model_tag}"

        try:
            # ── PASO 1: Obtener presigned URL del registry VL ───────────────
            config_url = f"{mothership_client.base_url}/api/v1/vl/models/{tenant_id}/vl/config"
            headers = {"x-api-key": mothership_client.api_key}

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(config_url, headers=headers)
                resp.raise_for_status()
                config = resp.json()

            tar_url = config.get("gguf_url") or config.get("lora_url")
            if not tar_url:
                raise ValueError("La Mothership no retornó un URL para el adaptador LoRA VL.")

            # ── PASO 2: Descargar el .tar.gz en streaming ──────
            logger.info("[VL OTA] Descargando Tarball LoRA VL...")
            async with httpx.AsyncClient(timeout=3600.0) as client:
                 async with client.stream("GET", tar_url) as r:
                    r.raise_for_status()
                    with open(tar_path, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
            
            logger.info(f"[VL OTA] Tarball descargado en: {tar_path}")

            # ── PASO 3: Extraer adaptador nativamente para vLLM ──────────────
            logger.info(f"[VL OTA] Descomprimiendo pesos a {lora_base_dir} ...")
            os.makedirs("./loras", exist_ok=True)
            
            def _extract_tar():
                with tarfile.open(tar_path, "r:gz") as tar:
                    # BUG 4 fix: filter='data' previene path traversal (recomendado Python 3.12+)
                    tar.extractall(path="./loras", filter="data")
            
            await asyncio.to_thread(_extract_tar)
            
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
