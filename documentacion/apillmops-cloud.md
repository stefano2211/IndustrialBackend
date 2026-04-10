# 🏗️ Auditoría de Arquitectura — ApiLLMOps (Mothership)

> **Rol:** Arquitecto de Software Senior  
> **Proyecto:** `ApiLLMOps` — Hub Central (Mothership) para Fine-Tuning y MLOps  
> **Última revisión:** Abril 2026

---

## 1. Visión General del Sistema

El sistema `ApiLLMOps` actúa como el **Mothership (Hub Central)** en una arquitectura Cloud-to-Edge. Su responsabilidad principal es recibir datos telemétricos o históricos desde múltiples nodos Edge (ej. `IndustrialBackend`), consolidarlos en un Data Lake (MinIO), orquestar procesos pesados de re-entrenamiento de LLMs (Fine-Tuning con LoRA) utilizando aceleración GPU via Unsloth, y coordinar el despliegue automático (OTA - Over-The-Air) de los nuevos adaptadores LoRA hacia los nodos Edge mediante webhooks y presigned URLs de S3.

### Stack Técnico

| Componente | Tecnología |
|------------|-----------|
| **API** | FastAPI + Uvicorn (Python 3.11) |
| **Task Queue** | Celery + Redis (Broker & Result Backend) |
| **Object Storage** | MinIO (S3-compatible) — 3 buckets: `datalake`, `datalake-vl`, `models` |
| **ML/Training** | PyTorch (cu128), Unsloth (QLoRA), Transformers ≥5.0, TRL, Pillow |
| **Formato Exportación** | Safetensors LoRA empaquetado en `.tar.gz` (compatible con vLLM) |
| **Infraestructura** | Docker Compose — `Dockerfile.api` (sin GPU) + `Dockerfile.worker` (CUDA 12.8) |

---

## 2. Arquitectura por Capas y Componentes

La estructura de carpetas sigue un patrón de diseño por capas (Domain-Driven Design simplificado) coherente con el Edge Node.

```text
┌─────────────────────────────────────────────────────────────┐
│                    Infraestructura Docker                    │
│  mops-api | mops-gpu-worker | mops-redis | mops-minio        │
│  mops-minio-init (crea buckets datalake + models al inicio)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI (app/main.py)                      │
│    Entrypoint, CORS, Healthcheck (/healthz)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   API Layer (app/api/)                       │
│  datasets.py    → POST /datasets/upload (texto, UUID shard)  │
│  vl_datasets.py → POST /vl/upload (screenshots+acciones)     │
│  training.py    → POST /training/job (encola Celery texto)   │
│  vl_training.py → POST /vl/training/job (encola Celery VL)   │
│  models.py      → GET /models/{id}/latest/config             │
│                   GET /vl/models/{id}/vl/config              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               Domain Layer (app/domain/)                     │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │   schemas/       │  │  services/                       │  │
│  │ TrainingJobRequest│  │ unsloth_trainer.py (Celery Task) │  │
│  │ VLTrainingJobReq  │  │ vl_trainer.py      (Celery Task) │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│           Persistence Layer (app/persistence/)               │
│  storage.py → MinioManager (upload, download, presigned URL, │
│               list_objects)                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Flujos de Integración Principales (Cloud-To-Edge)

### 3.1 Ingesta de Datos — Pipeline de Texto
El endpoint `POST /api/v1/datasets/upload` en `datasets.py` atiende a los Nodos Edge.
1. Recibe un archivo `.jsonl` y el `tenant_id`.
2. **Genera un nombre único con UUID** (`{tenant_id}_{tool_name}_{uuid8}.jsonl`) — cada upload es una partición independiente. No hay descarga ni reescritura del histórico.
3. Guarda localmente en `/tmp/datalake/` y lo sube al bucket `datalake` en MinIO.
4. Al momento de entrenar, el worker agrega todos los archivos del tenant por prefijo.

### 3.2 Ingesta de Datos — Pipeline VL (Vision-Language)
El endpoint `POST /api/v1/vl/upload` en `vl_datasets.py` maneja datos de Computer Use.
- Formato JSONL esperado: `{"messages": [...], "images": ["<base64_png>"]}` compatible con `FastVisionModel`.
- Descarga el objeto existente del bucket `datalake-vl`, le hace append del nuevo chunk, y lo re-sube como objeto canónico `{tenant_id}_vl_{tool_name}.jsonl`.
- **Nota**: Este patrón tiene una race condition si hay uploads concurrentes del mismo tenant.

### 3.3 Fine-Tuning — Dos Pipelines Independientes (Anti Catastrophic Forgetting)

**Pipeline de Texto** (`unsloth_trainer.py`):
- Base model: configurable por request (ej: `unsloth/qwen2.5-7b-bnb-4bit`)
- Usa `FastLanguageModel` con QLoRA 4-bit (r=16, alpha=32)
- `DataCollatorForCompletionOnlyLM`: entrena solo sobre las respuestas del assistant (SOTA)
- `standardize_sharegpt`: normaliza inconsistencias de formato entre distintos Edge nodes
- Exporta **safetensors LoRA** → comprime en `.tar.gz` → sube a bucket `models` como `{tenant_id}-v2-lora.tar.gz`

**Pipeline VL** (`vl_trainer.py`):
- Base model: `unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit`
- Usa `FastVisionModel` con fine-tuning de capas de visión Y lenguaje
- `UnslothVisionDataCollator`: maneja imágenes PIL + acciones JSON
- `dataloader_num_workers=0`: obligatorio, PIL no es serializable entre workers
- Exporta **safetensors LoRA VL** → comprime en `.tar.gz` → sube a `models` como `{tenant_id}-vl-lora.tar.gz`

**Webhook OTA** al finalizar:
```python
requests.post(webhook_url, json={"model_tag": f"{tenant_id}-v2"}, headers={"x-api-key": settings.API_KEY})
# VL:
requests.post(webhook_url, json={"model_tag": f"{tenant_id}-vl", "model_type": "vision"}, ...)
```

### 3.4 Registry de Modelos y Presigned URLs
El Edge consulta el registry para obtener la URL de descarga del artefacto:
- Texto: `GET /api/v1/models/{tenant_id}/latest/config` → retorna `{"lora_url": "<presigned_url>", "format": "Safetensors-LoRA"}`
- VL: `GET /api/v1/vl/models/{tenant_id}/vl/config` → retorna `{"lora_url": "<presigned_url>", "format": "safetensors"}`

El Edge descarga el `.tar.gz` directamente desde MinIO via la presigned URL, sin pasar por la API.

---

## 4. Unsloth Trainer Pipeline — Análisis Profundo

La pieza más valiosa de este repositorio es `start_finetuning_task` en `unsloth_trainer.py`. Es un pipeline MLOps State of the Art.

**Fortalezas detectadas:**
- **Uso de Unsloth:** Minimiza el footprint de VRAM dramáticamente y acelera el entrenamiento (x2 a x5) frente a HuggingFace TRL estándar. Permite correr el hub con GPUs de gama de entrada (RTX 3090/4090 o A10).
- **Format ChatML Normalizado:** `standardize_sharegpt` resuelve inconsistencias entre distintos recolectores de datos de los Edge.
- **DataCollatorForCompletionOnlyLM:** SOTA multi-turno. Solo penaliza el loss sobre las respuestas del assistant, no sobre los prompts del usuario.
- **`--max-tasks-per-child=1` en Dockerfile.worker:** Fuerza al SO a matar el proceso hijo y liberar el 100% de la VRAM GPU tras cada job. Previene memory fragmentation acumulada de PyTorch en jobs iterativos.
- **Pool `solo` (`-P solo`):** Correcto para PyTorch — evita el uso de `fork()` que corrompe el estado CUDA.
- **`HF_HUB_ENABLE_HF_TRANSFER=1`:** Acelera la descarga de pesos base de HuggingFace significativamente.
- **`hf_cache` volume compartido:** Evita re-descargar modelos base entre runs consecutivos.
- **`finally` block para VRAM:** `del model; gc.collect(); torch.cuda.empty_cache()` garantiza liberación de GPU pase lo que pase.

---

## 5. Bugs Críticos y Problemas Detectados

### 🔴 BUG CRÍTICO 1: `start_vl_finetuning_task` no registrada en el Worker Celery
```python
# app/core/celery_app.py:21
imports=["app.domain.services.unsloth_trainer"],  # vl_trainer NO está aquí
```
El worker solo importa `unsloth_trainer`. La tarea `start_vl_finetuning_task` de `vl_trainer.py` **nunca se registra**. Cualquier job VL encolado falla con `celery.exceptions.NotRegistered`. El pipeline VL completo no funciona en producción.

**Fix:** Agregar `"app.domain.services.vl_trainer"` a la lista `imports`.

---

### 🔴 BUG CRÍTICO 2: Bucket `datalake-vl` nunca se crea
```yaml
# docker-compose.yml — minio-init solo crea:
/usr/bin/mc mb myminio/datalake || true;
/usr/bin/mc mb myminio/models || true;
# datalake-vl NO se crea
```
`MinioManager.init_buckets()` tampoco lo crea (solo `datalake` y `models`), y además ese método **nunca se llama** en el startup de la aplicación. Todo upload VL (`POST /api/v1/vl/upload`) falla con `NoSuchBucket`.

**Fix:** Agregar `datalake-vl` al `minio-init` y llamar `storage.init_buckets()` en el startup de la API.

---

### 🔴 BUG CRÍTICO 3: Race condition en upload VL (pérdida de datos silenciosa)
El patrón de `vl_datasets.py` (download → append local → re-upload) es destructivo bajo concurrencia:
- Si dos requests llegan simultáneamente para el mismo `tenant_id + tool_name`, ambas descargan el estado actual, ambas hacen append, y la segunda sobrescribe la data appended de la primera.

**Fix:** Adoptar el mismo patrón que el pipeline de texto — UUID por upload, y agregar en el trainer.

---

### 🟠 PROBLEMA: Sin retry en webhook OTA post-training
```python
requests.post(webhook_url, json={"model_tag": ...}, timeout=10)
# Si falla → except Exception as e: logger.error(...)
```
Si el Edge está ocupado o hay un problema de red transitorio, el webhook falla silenciosamente. El training se completó exitosamente pero el Edge nunca recibe la señal OTA. No hay reintento.

**Fix:** Implementar retry con backoff exponencial (mínimo 3 intentos).

---

### 🟠 PROBLEMA: Limpieza de `/tmp/` incompleta en caso de error de training
La limpieza del directorio de export y el `.tar.gz` local solo ocurre dentro del bloque `try` (al final del Paso 5). Si el upload a S3 falla después de un training de 4 horas, los artefactos de `/tmp/models/` (~100MB–2GB) quedan en disco indefinidamente. Múltiples fallos llenan el disco del worker.

**Fix:** Mover la limpieza de archivos temporales al bloque `finally`.

---

### 🟡 RIESGO DE SEGURIDAD: `tenant_id` sin sanitización en paths de archivo
```python
local_dataset_path = f"/tmp/{object_name}"       # unsloth_trainer.py:59
local_vl_path = f"/tmp/{tenant_id}_vl_master.jsonl"  # vl_trainer.py:76
```
Un `tenant_id` malicioso con `../` podría apuntar a rutas del sistema (path traversal). El contenedor Docker mitiga el impacto, pero no elimina el riesgo.

**Fix:** Validar `tenant_id` con regex `^[a-zA-Z0-9_-]+$` en los endpoints.

---

### 🟡 RIESGO DE SEGURIDAD: API Key con comparación vulnerable a timing attack
```python
# security.py:8
if api_key == settings.API_KEY:
```
La comparación con `==` es susceptible a timing attacks. Usar `hmac.compare_digest(api_key, settings.API_KEY)`.

---

### ℹ️ Sin versionado dinámico de modelos
El output está hardcodeado como `{tenant_id}-v2-lora.tar.gz`. Cada training run sobrescribe el artefacto anterior en MinIO. No hay historial, fallback ni A/B testing posible.

---

## 6. Evaluación End-To-End: Test Suite

El script `test_e2e_mlops.py` valida correctamente la sanitización de `model_tag` (Fase 5) y el rechazo sin API key (Fase 4). Sin embargo:
- La Fase 2 usa `http://host.docker.internal:8000` como webhook URL — esto funciona en Docker Desktop en macOS/Windows pero **falla en Linux puro** sin configuración adicional.
- `MOTHERSHIP_API_KEY = "default-mothership-secret-key"` está hardcodeado en el test — debe leerse de variable de entorno.

---

## 7. Recomendaciones Prioritarias (Action Items)

### P0 — Crítico (rompen funcionalidad core)
1. **Registrar `vl_trainer` en Celery:** Agregar `"app.domain.services.vl_trainer"` a `celery_app.conf.imports`.
2. **Crear bucket `datalake-vl`:** Agregar al `minio-init` de docker-compose y llamar `storage.init_buckets()` en startup.
3. **Corregir race condition VL:** Adoptar UUID-per-shard en `vl_datasets.py` como en el pipeline de texto.

### P1 — Importante (robusteza y seguridad)
4. **Retry en webhook OTA:** 3 intentos con backoff exponencial en `unsloth_trainer.py` y `vl_trainer.py`.
5. **Cleanup en `finally`:** Mover `shutil.rmtree(export_dir)` y `os.remove(tar_path)` al bloque `finally`.
6. **Validar `tenant_id`:** Regex `^[a-zA-Z0-9_-]+$` en ambos trainers.
7. **Timing-safe API key:** Reemplazar `==` con `hmac.compare_digest`.

### P2 — Observabilidad MLOps
8. **Versiones dinámicas de modelo:** Usar `{tenant_id}-v{job_id[:8]}-lora.tar.gz` y exponer historial en el registry.
9. **Monitoreo Celery:** Agregar Flower (`celery flower`) para visibilidad del worker GPU.
10. **Conectar loss metrics:** Enviar `trainer_stats` a un sistema de tracking (MLFlow, LangSmith) para visualizar curvas de aprendizaje.
