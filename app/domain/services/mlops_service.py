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
                logger.info(f"[MLOps OTA] Preparando registro con Template Oficial de Tool Calling (Qwen2.5)...")
                
                # Inyectamos el template avanzado de Tools que Unsloth omitió
                QWEN_TOOL_TEMPLATE = """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <json-args>}
</tool_call>
{{- end }}<|im_end|>
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .Content }}
{{ .Content }}
{{- end }}
{{- if .ToolCalls }}
<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}<|im_end|>
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{- end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}
<|im_start|>assistant
{{- end }}"""

                logger.info(f"[MLOps OTA] Registrando modelo '{new_model_tag}' en Ollama (Fase 1: Ingestión de Pesos GGUF)...")
                create_payload = {
                    "name": new_model_tag,
                    "files": {
                        "model.gguf": digest
                    },
                    "modelfile": "FROM model.gguf\n",
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

                logger.info(f"[MLOps OTA] Fase 1 completa. Forzando Override de Template Tool-Calling (Fase 2) vía Ollama SDK...")
                
                # Usamos el SDK oficial de Python para evitar el bug de borrado de Modelfile de la REST API (v0.5+)
                import ollama
                ollama_client = ollama.Client(host=settings.ollama_base_url)
                
                def _compile_tools_template():
                    ollama_client.create(
                        model=new_model_tag,
                        from_=new_model_tag,
                        template=QWEN_TOOL_TEMPLATE,
                        parameters={"stop": ["<|im_start|>", "<|im_end|>"]}
                    )
                
                await asyncio.to_thread(_compile_tools_template)
                
                logger.success(f"[MLOps OTA] Modelo '{new_model_tag}' compilado con Tools Support Exitosamente.")

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
