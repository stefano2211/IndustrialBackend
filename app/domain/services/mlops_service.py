import re
import os
import httpx
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
        Recibe una señal del Hub Central de que el nuevo modelo adaptado está listo.
        1. Obtiene presigned URLs (.gguf + Modelfile) desde ApiLLMOps (MinIO presigned).
        2. Descarga los artefactos al /tmp/ en modo streaming (chunk_size=1MB).
        3. Parchea el Modelfile: FROM <relativa> → FROM /tmp/<tag>.gguf.
        4. Registra el modelo en Ollama vía API REST POST /api/create.
        """
        import re as _re
        logger.info(f"[MLOps OTA] Iniciando actualización OTA para modelo: {new_model_tag}")

        gguf_path = f"/tmp/{new_model_tag}.gguf"
        modelfile_path = f"/tmp/{new_model_tag}.Modelfile"

        try:
            # --- PASO 1: Obtener presigned URLs del registro de modelos de Mothership ---
            config_url = f"{mothership_client.base_url}/api/v1/models/{tenant_id}/latest/config"
            headers = {"x-api-key": mothership_client.api_key}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(config_url, headers=headers)
                resp.raise_for_status()
                config = resp.json()

            gguf_url = config["gguf_url"]
            modelfile_url = config["modelfile_url"]
            logger.info("[MLOps OTA] URLs de descarga obtenidas correctamente.")

            # --- PASO 2: Descargar .gguf y Modelfile en streaming ---
            async with httpx.AsyncClient(timeout=3600.0) as client:
                for url, dest_path, label in [
                    (gguf_url, gguf_path, "GGUF"),
                    (modelfile_url, modelfile_path, "Modelfile"),
                ]:
                    async with client.stream("GET", url) as r:
                        r.raise_for_status()
                        with open(dest_path, "wb") as f:
                            async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                    logger.info(f"[MLOps OTA] {label} descargado en: {dest_path}")

            # --- PASO 3: Parchear FROM del Modelfile para apuntar al .gguf local ---
            with open(modelfile_path, "r", encoding="utf-8") as f:
                modelfile_content = f.read()
            modelfile_content = _re.sub(
                r'^FROM\s+.*$',
                f'FROM {gguf_path}',
                modelfile_content,
                flags=_re.MULTILINE
            )
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)
            logger.info(f"[MLOps OTA] Modelfile parcheado. FROM apunta a: {gguf_path}")

            # --- PASO 4: Registrar el modelo en Ollama vía API REST (POST /api/create) ---
            logger.info(f"[MLOps OTA] Registrando modelo '{new_model_tag}' en Ollama...")
            create_payload = {
                "name": new_model_tag,
                "modelfile": modelfile_content,
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                ollama_url = f"{settings.ollama_base_url}/api/create"
                resp = await client.post(ollama_url, json=create_payload)
                if resp.status_code == 200:
                    logger.success(f"[MLOps OTA] Modelo '{new_model_tag}' registrado exitosamente.")
                else:
                    logger.error(f"[MLOps OTA] Falló la creación en Ollama (HTTP {resp.status_code}): {resp.text}")
                    raise Exception(f"Ollama API Error: {resp.text}")

        except Exception as e:
            logger.error(f"[MLOps OTA] Excepción durante el proceso OTA: {e}")
            return {"status": "error", "tag": new_model_tag, "detail": str(e)}
        finally:
            for path in [gguf_path, modelfile_path]:
                if os.path.exists(path):
                    os.remove(path)
            logger.info("[MLOps OTA] Archivos temporales limpiados.")

        return {"status": "success", "tag": new_model_tag}
