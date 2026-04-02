# 🏗️ Auditoría de Arquitectura — ApiLLMOps (Mothership)

> **Rol:** Arquitecto de Software Senior  
> **Proyecto:** `ApiLLMOps` — Hub Central (Mothership) para Fine-Tuning y MLOps

---

## 1. Visión General del Sistema

El sistema `ApiLLMOps` actúa como el **Mothership (Hub Central)** en una arquitectura Cloud-to-Edge. Su responsabilidad principal es recibir datos telemétricos o históricos desde múltiples nodos Edge (ej. `IndustrialBackend`), consolidarlos en un Data Lake, orquestar procesos pesados de re-entrenamiento de LLMs (Fine-Tuning) utilizando aceleración GPU, y coordinar el despliegue automático (OTA - Over-The-Air) de los nuevos modelos hacia los nodos mediante webhooks y presigned URLs.

### Stack Técnico

| Componente | Tecnología |
|------------|-----------|
| **API** | FastAPI + Uvicorn |
| **Task Queue** | Celery + Redis (Broker & Result Backend) |
| **Object Storage** | MinIO (S3-compatible) para Data Lake y Modelos |
| **ML/Training** | PyTorch (cu128), Unsloth (QLoRA), Transformers, TRL |
| **Formato Exportación** | GGUF nativo + Ollama Modelfile |
| **Infraestructura** | Docker Compose (API container + GPU Worker separado) |

---

## 2. Arquitectura por Capas y Componentes

La estructura de carpetas sigue un patrón de diseño por capas (Domain-Driven Design simplificado) muy similar al del Edge Node, asegurando coherencia mental entre ambos proyectos.

```text
┌─────────────────────────────────────────────────────┐
│                 Infraestructura Docker               │
│  mops-api | mops-gpu-worker | mops-redis | mops-minio│
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   FastAPI (app/main.py)              │
│    Entrypoint, Auth, Middlewares, Healthchecks       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              API Layer (app/api/)                    │
│  datasets.py (Ingesta de datos .jsonl)               │
│  training.py (Dispara y consulta jobs de Celery)     │
│  models.py   (Registry: emite Presigned URLs S3)     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│          Domain Layer (app/domain/)                  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │   schemas/   │  │  services/   │                 │
│  │ Pydantic DTOs│  │unsloth_trainer (Celery Task)   │
│  └──────────────┘  └──────────────┘                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│        Persistence Layer (app/persistence/)          │
│  storage.py (MinioManager: S3 abstraction layer)     │
└─────────────────────────────────────────────────────┘
```

---

## 3. Flujos de Integración Principales (Cloud-To-Edge)

### 3.1 Ingesta de Datos (Data Lake Append)
El Endpoint `POST /upload` en `datasets.py` atiende a los Nodos Edge.
1. Recibe un archivo `.jsonl` de telemetría y el `tenant_id`.
2. Descarga el "master file" actual del bucket MinIO al `/tmp/` local.
3. Le anexa asincrónicamente el nuevo chunk (`aiofiles`).
4. Reemplaza el master file viejo en S3 subiendo el nuevo.

### 3.2 Fine-Tuning & Exportación GGUF
El Endpoint `POST /job` en `training.py` encola una tarea en Celery. El Worker GPU la consume:
1. **Download:** Descarga todos los `.jsonl` correspondientes al `tenant_id`.
2. **Pre-processing:** Carga los datos, formatea a ChatML usando HuggingFace `trl`.
3. **Training:** Carga el `base_model` usando `Unsloth` (load_in_4bit) + QLoRA target modules. Entrena usando `SFTTrainer`.
4. **Compile & Quantize:** Exporta directamente desde el estado LoRA activo a formato GGUF (int4/`q4_k_m`) nativo para llama.cpp/Ollama.
5. **Storage:** Sube el `.gguf` y el `.Modelfile` al bucket `models` de S3.
6. **Webook (OTA Trigger):** Si se proveyó un `webhook_url`, postea un JSON al Edge Target reportando el fin del entrenamiento con el label del modelo (ej. `aura_tenant_01-v2`).

### 3.3 Consumo OTA (Model Pull)
Días o semanas después del webhoook, cuando un Nodo Edge arranca u obedece a un reinicio, llama a:
`GET /{tenant_id}/latest/config` en `models.py`.
1. Emite URLs presignadas y efímeras de S3 para el GGUF y Modelfile.
2. El Edge los descarga directamente saltándose la API limitando cuellos de botella en la red de la Mothership.

---

## 4. Unsloth Trainer Pipeline — Análisis Profundo

La pieza más valiosa de este repositorio es `start_finetuning_task` en `unsloth_trainer.py`. Es un pipeline MLOps State of the Art.

**Fortalezas detectadas:**
- **Uso de Unsloth:** Minimiza el footprint de VRAM dramáticamente y acelera el entrenamiento (x2 a x5) en comparación con un stack tradicional de HuggingFace TRL estandar. Permitirá correr el hub con GPUs de gama de entrada (RTX 3090 / 4090 o A10).
- **Format ChatML Normalizado:** El código hace `standardize_sharegpt` lo cual soluciona muchas inconsistencias entre diferentes recolectores de datos de los Edge.
- **DataCollatorForCompletionOnlyLM:** Gran detalle SOTA. El agente no penaliza el loss function sobre el texto de los *prompts* del usuario, optimizando exclusivamente qué tan bien responde el *assistant*.
- **`MAX_JOBS=1` en llama.cpp:** Evita crasheos "Out of Memory" misteriosos en workers Docker que ocurren cuando llama.cpp intenta compilar C++ con docenas de hilos (paralelismo agresivo detectado de un bug clásico).

---

## 5. Problemas y Observaciones Críticas (Riesgos a Escala)

A pesar de ser un diseño moderno excelente, existen vulnerabilidades operativas y cuellos de botella en el uso de datos.

### ❌ CUELLO DE BOTELLA CRÍTICO: I/O Append en Memoria/Disco (`datasets.py`)
El código en `datasets.py` (`POST /upload`):
```python
storage.download_file(bucket, object_name, local_master_file) # Descarga un dataset maestro temporal entero
# Append local
storage.upload_file(bucket, object_name, local_master_file)   # Lo sube entero otra vez
```
**Impacto:** Cuando el Data Lake de un `tenant_id` llegue, digamos, a 50GB en `.jsonl` generado con el tiempo: 
- Cada nodo Edge subiendo apenas un chunk de 1MB forzará que la API descargue 50GB, le pegue 1MB, y resuba 50.001GB. Encolará tiempos de respuesta HTTP horribles y consumirá IOPS, banda ancha interna y disco temporal de Docker hasta colapsar.
- **Solución P0:** Migrar a la API multipart append si soportado por MinIO, cambiar el patrón a `Partitioning` (subir `sensor_data_2026-04-01T12:00.jsonl` independientemente) y luego que el Worker consolide/cargue a demanda usando DuckDB u Objeto S3 Spark al momento de entrenar.

### ⚠️ RIESGO LATENTE DE OOM (Out Of Memory) en Celery GPU Worker (`unsloth_trainer.py`)
Al final de la tarea de entrenamiento (línea 286), se utiliza:
```python
del trainer, model, tokenizer
gc.collect()
torch.cuda.empty_cache()
```
Aunque esto funciona en notebooks de Colab, usar Celery como *Long-running worker de PyTorch* casi siempre da por resultado una degradación de memoria GPU (memory fragmentation o leaks de C++) resultando en **OOM después de 3 o 4 tareas iterativas seguidas**.
- **Solución P1:** Configurar Celery con el flag `--max-tasks-per-child=1`. Esto obligará al SO a matar el proceso hijo y liberar el 100% real de la memoria GPU, lanzando un proceso hijo fresco a la siguiente tarea de la cola. Ya está la queue de Redis para mantener la persistencia. Para ajustar esto, modificar en `Dockerfile.worker` la linea extra: `CMD ["celery", "-A", "app.core.celery_app", "worker", ... "--max-tasks-per-child=1"]`

### ⚠️ INYECCIÓN DE DEPENDENCIA DE WEBHOOK INSEGURA
El Payload original enviado del Celery:
```python
requests.post(webhook_url, json={"model_tag": f"{tenant_id}-v2"}, ...)
```
1. Confía ciegamente en `req.webhook_url` ingresado en el `POST /job`. Si un atacante malicioso interno proveé un Webhook hacia una IP perimetral (SSRF - Server Side Request Forgery) el *GPU Worker* atacará su objetivo o filtrará datos.
2. Debería haber una validación de Tenant vs Endpoints Registrados en lugar de aceptar `webhook_url` ciegamente en la cabecera.

### ℹ️ NO HAY VERSIÓN DINÁMICA DE MODELOS
Para el output y tags, está hardcodeado como `aura_tenant_01-v2`. Cada vez que corras una corrida, el output machacará los bucket items de `-v2.gguf`. Una rotación por `epoch_{timestamp}` o el ID del job de celery como tag sería fundamental en una pipeline formal para permitir fallbacks/A-B Testing.

---

## 6. Integración Seguridad: End-To-End Test Evaluation

El script `test_e2e_mlops.py` suministrado expone que ya se validan correctamente ciertos supuestos de inyección en shell (Phase 4 y 5 de la suite) del lado de `IndustrialBackend`. Pero el test de la fase 2 invoca `http://host.docker.internal:8000/mlops/webhook/model-ready`. Note que `host.docker.internal` romperá resoluciones DNS de Linux si el orquestador no lo declara explícitamente y en Linux puro en prod (sin Docker Desktop) esto falla. 

---

## 7. Recomendaciones Prioritarias (Action Items)

### P0 — Crítico (Evitan Caídas del Sistema)
1. **Modificar el Append Pattern** en el Endpoint de Upload. Acumular el `append` como particiones individuales diarias/por envío del MinIO, que luego el Trainer descargue iterativamente. No descargar/resubir todo el monolito en cada HTTP de 1MB.
2. **Forzar Reciclaje del Worker:** Agregar flag `--max-tasks-per-child=1` en el CMD del `Dockerfile.worker` o correr en OOM a la terca o cuarta hora de operación continua.

### P1 — Importante (Robusteza)
3. **Versiones Dinámicas de Modelo:** Sustituir los strings estáticos (ej: `tenant_id-v2`) con hashes combinados de `job_id` y `tenant_id` para historizar artefactos (`aura-01-abcdef.gguf`). Modificar el API para pedir `get_latest` basados en orden del bucket.
4. Protección SSRF: Limitar dominios/IPs a la cual Celery tiene permiso de hacer webhook alerts. O manejar el registry desde la BD.

### P2 — Observables MLOps
5. Conectar Loguru directamente al sistema de Tracking (e.g. LangSmith or MLFlow) para el Loss Rate ya que entrenar en Celery pierde stdout/stderr muy comúnmente y el cliente final no verá curvas de aprendizaje de forma directa, impidiendo curar su dataset correctamente.
