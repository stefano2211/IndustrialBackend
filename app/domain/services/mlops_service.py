import re
import os
import json
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
        4. Registra el modelo en Ollama vía API REST POST /api/create (streaming NDJSON).
        5. Limpia archivos temporales después de que Ollama termine.
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
            # CRITICAL: Ollama /api/create ALWAYS returns HTTP 200, even on failure.
            # Success/failure is only detectable by reading the NDJSON stream body:
            # each line is a JSON object; if any contains "error", the registration failed.
            # The GGUF file must remain on disk until Ollama finishes reading it.
            logger.info(f"[MLOps OTA] Registrando modelo '{new_model_tag}' en Ollama...")
            create_payload = {
                "name": new_model_tag,
                "modelfile": modelfile_content,
                "stream": True,
            }
            ollama_error = None
            async with httpx.AsyncClient(timeout=600.0) as client:
                ollama_url = f"{settings.ollama_base_url}/api/create"
                async with client.stream("POST", ollama_url, json=create_payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        status_msg = chunk.get("status", "")
                        if status_msg:
                            logger.info(f"[MLOps OTA] Ollama: {status_msg}")
                        if "error" in chunk:
                            ollama_error = chunk["error"]
                            break

            if ollama_error:
                raise Exception(f"Ollama registration failed: {ollama_error}")

            logger.success(f"[MLOps OTA] Modelo '{new_model_tag}' registrado exitosamente en Ollama.")

        except Exception as e:
            logger.error(f"[MLOps OTA] Excepción durante el proceso OTA: {e}")
            return {"status": "error", "tag": new_model_tag, "detail": str(e)}
        finally:
            # Cleanup runs AFTER Ollama has fully ingested the GGUF model (stream completed)
            for path in [gguf_path, modelfile_path]:
                if os.path.exists(path):
                    os.remove(path)
            logger.info("[MLOps OTA] Archivos temporales limpiados.")

        return {"status": "success", "tag": new_model_tag}
