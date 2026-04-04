"""
VL MLOps Service — OTA Update para modelos Vision-Language
===========================================================
Análogo a MLOpsService pero maneja DOS archivos GGUF:
  - model.gguf        → pesos del LLM + LoRA fusionado
  - mmproj.gguf       → vision projector (encoder visual)

El Modelfile de Ollama usa FROM + ADAPTER para registrar ambos.

Flujo OTA VL:
  1. GET /api/v1/vl/models/{tenant_id}/config → presigned URLs
  2. Download model.gguf + mmproj.gguf (streaming)
  3. POST /api/blobs/{sha256_model} → Ollama
  4. POST /api/blobs/{sha256_mmproj} → Ollama
  5. Construir Modelfile: FROM @sha256:model + ADAPTER @sha256:mmproj
  6. POST /api/create → Ollama registra el modelo VL
  7. Actualizar settings en runtime (system1_model, default_llm_model)
"""

import os
import hashlib
import asyncio
import httpx
from loguru import logger

from app.core.config import settings
from app.core.mothership_client import mothership_client

# Template Ollama para Qwen2.5-VL con soporte de imágenes y tool calling
QWEN_VL_MODELFILE_TEMPLATE = """FROM @sha256:{model_sha256}
ADAPTER @sha256:{mmproj_sha256}
PARAMETER num_ctx 8192
PARAMETER temperature 0.1
TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
\"\"\"
SYSTEM \"You are Aura, an industrial AI assistant with both vision and language capabilities.\"
"""


class VLMLOpsService:
    """
    Gestiona el ciclo OTA para modelos Vision-Language (Qwen2.5-VL fine-tuned).
    Descarga AMBOS artefactos (model + mmproj) y los registra en Ollama.
    """

    async def process_vl_ota_webhook(
        self,
        model_tag: str,
        mmproj_tag: str,
        tenant_id: str = "aura_tenant_01",
    ):
        """
        Procesa el webhook de modelo VL listo desde ApiLLMOps.

        Args:
            model_tag:  Tag del modelo principal (ej: "aura_tenant_01-vl")
            mmproj_tag: Tag del vision projector (ej: "aura_tenant_01-vl-mmproj")
            tenant_id:  ID del tenant (para construir la URL de config)
        """
        logger.info(f"[VL OTA] 🚀 Iniciando actualización OTA VL: {model_tag} + {mmproj_tag}")

        gguf_path = f"/tmp/{model_tag}.gguf"
        mmproj_path = f"/tmp/{mmproj_tag}.gguf"

        try:
            # ── PASO 1: Obtener presigned URLs del registry VL ───────────────
            config_url = (
                f"{mothership_client.base_url}/api/v1/vl/models/{tenant_id}/vl/config"
            )
            headers = {"x-api-key": mothership_client.api_key}

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(config_url, headers=headers)
                resp.raise_for_status()
                config = resp.json()

            gguf_url = config["gguf_url"]
            mmproj_url = config["mmproj_url"]
            logger.info("[VL OTA] URLs de descarga VL obtenidas.")

            # ── PASO 2: Descargar model.gguf y mmproj.gguf en streaming ──────
            async with httpx.AsyncClient(timeout=7200.0) as client:
                for url, dest_path, label in [
                    (gguf_url, gguf_path, "GGUF Principal"),
                    (mmproj_url, mmproj_path, "Vision Projector (mmproj)"),
                ]:
                    logger.info(f"[VL OTA] Descargando {label}...")
                    async with client.stream("GET", url) as r:
                        r.raise_for_status()
                        with open(dest_path, "wb") as f:
                            total = 0
                            async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                                total += len(chunk)
                    logger.info(
                        f"[VL OTA] {label} descargado: {dest_path} "
                        f"({total / (1024**3):.2f} GB)"
                    )

            # ── PASO 3: Calcular SHA256 de ambos archivos ────────────────────
            def _calc_sha(path: str) -> str:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                return h.hexdigest()

            model_sha, mmproj_sha = await asyncio.gather(
                asyncio.to_thread(_calc_sha, gguf_path),
                asyncio.to_thread(_calc_sha, mmproj_path),
            )
            logger.info(f"[VL OTA] SHA256 modelo: {model_sha[:16]}...")
            logger.info(f"[VL OTA] SHA256 mmproj: {mmproj_sha[:16]}...")

            # ── PASO 4: Subir ambos blobs a Ollama ───────────────────────────
            ollama_base = settings.ollama_base_url.rstrip("/")

            async with httpx.AsyncClient(timeout=3600.0) as client:
                for sha, path, label in [
                    (model_sha, gguf_path, "GGUF Principal"),
                    (mmproj_sha, mmproj_path, "mmproj"),
                ]:
                    digest = f"sha256:{sha}"
                    # HEAD check — si ya existe no re-subimos
                    head_resp = await client.head(
                        f"{ollama_base}/api/blobs/{digest}", timeout=10.0
                    )
                    if head_resp.status_code == 200:
                        logger.info(f"[VL OTA] {label} ya en Ollama blob cache. Skipping upload.")
                        continue

                    logger.info(f"[VL OTA] Subiendo {label} al blob cache de Ollama...")

                    def _stream_file(p):
                        with open(p, "rb") as f:
                            while True:
                                chunk = f.read(1024 * 1024)
                                if not chunk:
                                    break
                                yield chunk

                    async def _upload_blob(sha_digest, file_path):
                        async with httpx.AsyncClient(timeout=3600.0) as upload_client:
                            with open(file_path, "rb") as f:
                                resp = await upload_client.post(
                                    f"{ollama_base}/api/blobs/{sha_digest}",
                                    content=f,
                                    headers={"Content-Type": "application/octet-stream"},
                                )
                                resp.raise_for_status()

                    await _upload_blob(digest, path)
                    logger.info(f"[VL OTA] ✅ {label} blob subido a Ollama.")

            # ── PASO 5: Construir Modelfile con FROM + ADAPTER ───────────────
            modelfile_content = QWEN_VL_MODELFILE_TEMPLATE.format(
                model_sha256=model_sha,
                mmproj_sha256=mmproj_sha,
            )
            logger.info("[VL OTA] Modelfile VL construido con FROM + ADAPTER.")

            # ── PASO 6: Registrar modelo en Ollama vía /api/create ───────────
            model_name = model_tag  # ej: "aura_tenant_01-vl"
            logger.info(f"[VL OTA] Registrando modelo VL '{model_name}' en Ollama...")

            async with httpx.AsyncClient(timeout=600.0) as client:
                # Ollama /api/create puede tardar varios minutos en procesar
                async with client.stream(
                    "POST",
                    f"{ollama_base}/api/create",
                    json={"name": model_name, "modelfile": modelfile_content},
                    timeout=600.0,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.strip():
                            try:
                                import json
                                data = json.loads(line)
                                status = data.get("status", "")
                                if status:
                                    logger.info(f"[VL OTA] Ollama: {status}")
                            except Exception:
                                logger.debug(f"[VL OTA] Ollama raw: {line}")

            logger.info(f"[VL OTA] ✅ Modelo VL '{model_name}' registrado en Ollama exitosamente.")

            # ── PASO 7: Limpiar archivos temporales ──────────────────────────
            for path in [gguf_path, mmproj_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.debug(f"[VL OTA] Limpiado: {path}")
                except OSError:
                    pass

            logger.info(
                f"[VL OTA] 🎉 OTA VL completo. "
                f"Modelo '{model_name}' activo en Ollama con capacidad visual."
            )

        except Exception as e:
            logger.error(f"[VL OTA] ❌ Error en OTA VL: {e}")
            # Limpiar archivos parciales
            for path in [gguf_path, mmproj_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
            raise
