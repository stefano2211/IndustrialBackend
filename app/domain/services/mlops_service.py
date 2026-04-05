import re
import os
import json
import httpx
import hashlib
import asyncio
import re as _re
from loguru import logger

from app.core.config import settings
from app.core.mothership_client import mothership_client


class MLOpsService:
    """
    MLOps Service — gestiona actualizaciones OTA de modelos desde la Mothership (ApiLLMOps).

    Responsabilidad única: recibir webhook de modelo listo → descargar GGUF → registrar en Ollama.
    La recolección de datos es responsabilidad exclusiva del DB Collector.
    """

    async def process_ota_webhook(self, new_model_tag: str, tenant_id: str = "aura_tenant_01"):
        """
        Recibe una señal del Hub Central de que el nuevo adaptador LoRA está listo.
        1. Obtiene presigned URL (.tar.gz) desde ApiLLMOps.
        2. Descarga el artefacto en streaming al /tmp/.
        3. Extrae directamente el adaptador LoRA a la carpeta montada en vLLM (/loras/).
        4. Limpia archivos temporales. No es necesario reiniciar vLLM.
        """
        import tarfile
        logger.info(f"[MLOps OTA] Iniciando actualización OTA para adaptador LoRA: {new_model_tag}")

        tar_path = f"/tmp/{new_model_tag}.tar.gz"
        lora_base_dir = f"./loras/{new_model_tag}"

        try:
            # --- PASO 1: Obtener presigned URL del adaptador desde Mothership ---
            # La API devuelve el url del snapshot Zippeado
            config_url = f"{mothership_client.base_url}/api/v1/models/{tenant_id}/latest/config"
            headers = {"x-api-key": mothership_client.api_key}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(config_url, headers=headers)
                resp.raise_for_status()
                config = resp.json()

            # En la Mothership hemos cambiado la lógica para que gguf_url devuelva el archivo tar.gz del adaptador
            tar_url = config.get("gguf_url") or config.get("lora_url")
            if not tar_url:
                raise ValueError("La Mothership no retornó un URL para el adaptador LoRA.")

            logger.info("[MLOps OTA] URLs de descarga obtenidas correctamente. Descargando adaptador...")

            # --- PASO 2: Descargar el .tar.gz en streaming ---
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream("GET", tar_url) as r:
                    r.raise_for_status()
                    with open(tar_path, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                logger.info(f"[MLOps OTA] Tarball descargado en: {tar_path}")

            # --- PASO 3: Extraer adaptador nativamente para vLLM ---
            logger.info(f"[MLOps OTA] Descomprimiendo pesos a {lora_base_dir} ...")
            os.makedirs("./loras", exist_ok=True)
            
            def _extract_tar():
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path="./loras")
            
            await asyncio.to_thread(_extract_tar)
            
            logger.success(f"[MLOps OTA] Adaptador '{new_model_tag}' disponible en '{lora_base_dir}' para inyección de vLLM dinámica.")

        except Exception as e:
            logger.error(f"[MLOps OTA] Excepción durante el proceso OTA: {e}")
            return {"status": "error", "tag": new_model_tag, "detail": str(e)}
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)
            logger.info("[MLOps OTA] Archivos temporales limpiados.")

        return {"status": "success", "tag": new_model_tag}

