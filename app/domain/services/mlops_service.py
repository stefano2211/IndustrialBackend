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

            # --- PASO 3: Calcular SHA256 para el Blob API (Requisito estricto Ollama 0.5+) ---
            logger.info(f"[MLOps OTA] Exponiendo GGUF interno vía Blob API a Ollama...")
            def _calc_sha():
                h = hashlib.sha256()
                with open(gguf_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                return f"sha256:{h.hexdigest()}"
            
            digest = await asyncio.to_thread(_calc_sha)
            
            async with httpx.AsyncClient() as client:
                # Comprobar si ya existe el blob
                res_head = await client.head(f"{settings.ollama_base_url}/api/blobs/{digest}", timeout=10.0)
                if res_head.status_code != 200:
                    logger.info(f"[MLOps OTA] GGUF no cacheado en VRAM. Subiendo Blob {digest}...")
                    
                    # Generador async para streaming y evitar saturar RAM con 3GB
                    async def file_streamer():
                        with open(gguf_path, "rb") as f:
                            while chunk := f.read(8192):
                                yield chunk
                                
                    res_blob = await client.post(
                        f"{settings.ollama_base_url}/api/blobs/{digest}", 
                        content=file_streamer(), 
                        timeout=1800.0
                    )
                    res_blob.raise_for_status()

                # --- PASO 4: Registrar el modelo en Ollama vía API REST (POST /api/create) ---
                logger.info(f"[MLOps OTA] Preparando registro de modelo con metadata original...")
                
                # Leemos el Modelfile original (que contiene el TEMPLATE y PARAMETERS de Unsloth)
                # y lo parcheamos para que apunte al alias 'model.gguf' definido en 'files'
                with open(modelfile_path, "r", encoding="utf-8") as f:
                    rich_modelfile = f.read()
                
                rich_modelfile = _re.sub(
                    r'^FROM\s+.*$',
                    'FROM model.gguf',
                    rich_modelfile,
                    flags=_re.MULTILINE
                )

                logger.info(f"[MLOps OTA] Registrando modelo '{new_model_tag}' en Ollama...")
                create_payload = {
                    "name": new_model_tag,
                    "files": {
                        "model.gguf": digest
                    },
                    "modelfile": rich_modelfile,
                    "stream": True,
                }
                
                ollama_error = None
                ollama_url = f"{settings.ollama_base_url}/api/create"
                async with client.stream("POST", ollama_url, json=create_payload, timeout=600.0) as r:
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
