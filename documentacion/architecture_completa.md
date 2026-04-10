# 🏗️ Auditoría de Arquitectura — IndustrialBackend (Aura AI)

> **Rol:** Arquitecto de Software Senior
> **Proyecto:** `IndustrialBackend` — Edge AI para Seguridad Industrial  
> **Última revisión:** Abril 2026

---

## 1. Visión General del Sistema

El sistema es un **backend Edge AI** construido sobre FastAPI, que implementa un agente de IA multi-nivel jerárquico para análisis de documentos industriales, datos SCADA en tiempo real, y automatización visual de interfaces GUI (Computer Use). La arquitectura es un sistema **Agentic RAG + MCP + MLOps** de borde (edge) con soporte multi-modelo y loop de aprendizaje continuo hacia la nube.

### Stack Técnico

| Capa | Tecnología |
|------|-----------|
| API | FastAPI 0.115 + uvicorn |
| Agentes | `deepagents` + LangGraph + LangChain |
| LLMs (local) | vLLM — `Qwen/Qwen3.5-2B` como base + adaptadores LoRA dinámicos |
| LLMs (cloud) | OpenRouter (fallback / modelos premium) |
| Persistencia SQL | PostgreSQL + SQLModel + asyncpg |
| Checkpoints de Agente | LangGraph `AsyncPostgresSaver` (memoria por thread) |
| Store Long-Term | LangGraph `AsyncPostgresStore` (memoria por usuario, cross-thread) |
| Vector Store | Qdrant (RAG — documentos técnicos e industriales) |
| Blobs | MinIO (documentos, archivos de usuario) |
| Scheduler | APScheduler (cron jobs para DB Collector) |
| MLOps | OTA via ApiLLMOps (Mothership) — descarga safetensors LoRA, inyecta en vLLM sin reiniciar |

---

## 2. Arquitectura por Capas

```
+-------------------------------------------------+
|              FastAPI (app/main.py)              |
|     CORS + Lifecycle (checkpointer, store)      |
+--------------------+----------------------------+
                     |
+--------------------v----------------------------+
|           API Layer (app/api/)                  |
|   router.py -> 14 endpoints                     |
|   deps.py   -> JWT auth dependency              |
+--------------------+----------------------------+
                     |
+--------------------v----------------------------+
|         Domain Layer (app/domain/)              |
|                                                 |
|  +-------------+   +--------------+            |
|  |  agent/     |   |  services/   |            |
|  |  deep_agent |   |  agent_svc   |            |
|  |  gen_agent  |   |  mcp_svc     |            |
|  |  satellite  |   |  mlops_svc   |            |
|  |  subagents  |   |  doc_svc     |            |
|  |  tools/     |   |  ...         |            |
|  |  memory/    |   +--------------+            |
|  |  prompts/   |                               |
|  |  skills/    |   +--------------+            |
|  |  middleware |   | db_collector |            |
|  +-------------+   |  (scheduler) |            |
|                    +--------------+            |
+--------------------+----------------------------+
                     |
+--------------------v----------------------------+
|       Persistence Layer (app/persistence/)      |
|  db.py (SQLAlchemy async) | vector.py (Qdrant)  |
|  memoryAI/ (checkpointer + store -> Postgres)   |
|  repositories/ (10 repos, 1 por entidad)        |
|  blob.py (MinIO) | replay_buffer.py             |
+-------------------------------------------------+
```

---

## 3. Analisis Profundo — Sistema de Agentes

Esta es la parte mas compleja y el corazon del sistema. Tiene una **arquitectura jerarquica de agentes** con dos modos de operacion.

### 3.1 Jerarquia de Agentes

```
Modo 1: Expert Direct (default, use_generalist=False)
-----------------------------------------------------
 User Query
     |
     v
 [IndustrialAgent] (vLLM Qwen3.5-2B + LoRA alias: aura_expert=/loras/aura_tenant_01-v2)
     +-- [knowledge-researcher] SubAgent
     |       +-- ask_knowledge_agent -> Qdrant RAG (documentos técnicos)
     +-- [mcp-orchestrator] SubAgent
     |       +-- call_dynamic_mcp -> APIs REST / MCP SSE / stdio
     +-- [general-assistant] SubAgent (fallback sin herramientas)


Modo 2: Macrohard Generalist Orchestrator (use_generalist=True)
---------------------------------------------------------------
 User Query
     |
     v
 [Generalist Orchestrator] (vLLM Qwen3.5-2B — "Unified multimodal director")
     +-- [Sistema 1 Subagent] Tool -> VL model (alias: aura_system1=/loras/aura_tenant_01-vl)
     |       +-- (Observe-Think-Act loop con screenshots de la GUI)
     |       +-- Feature flag: system1_enabled (default: True)
     +-- [Computer Use Subagent] Tool -> vLLM Qwen3.5-2B
     |       +-- (automatización GUI — demo mode: True por defecto)
     |       +-- Feature flag: computer_use_enabled (default: True)
     +-- [Industrial Expert] Tool -> IndustrialAgent (mismo que Modo 1)
             +-- (acceso en tiempo real a SCADA, RAG y APIs vía MCP)
```

La arquitectura "Macrohard" mantiene la separación: el Orchestrator de texto y razonamiento nunca ejecuta tareas visuales directamente. Las tareas GUI son exclusivas del Sistema 1 VL aislado (Computer Use subagent).

### 3.2 Flujo de Creacion del Agente (AgentService)

Cada request de chat **construye el agente desde cero**. El flujo es:

```
invoke() / stream()
    |
    +-- 1. Resolver model_id -> DB -> LLMProvider + model_name
    +-- 2. Merge params (DB model defaults + UI overrides)
    +-- 3. Crear LLM (ui_generalist_llm)
    +-- 4. Aplicar retry/timeout settings
    +-- 4.3 [stream only] Temporal Router -> es query historica?
    |         Si si -> deshabilitar MCP + Knowledge (ahorra tokens)
    +-- 4.4 Crear expert_llm_factory (lambda lazy)
    +-- 5. Merge system prompt (UI params + DB model)
    +-- 6. Cargar ToolConfigs desde DB -> construir tools_context string
    +-- 7. Crear agente (generalist or industrial)
    +-- 8. Invoke / Stream con config {thread_id, user_id, kb_id, session}
```

### 3.3 Deep Agent Factory (deep_agent.py) — Fortalezas

El factory sigue principios SOLID correctamente:
- **SRP**: Una sola responsabilidad: ensamblar el agente.
- **OCP**: Se pueden agregar subagentes sin modificar el factory (via `get_all_subagents()`).
- **DIP**: Recibe modelos y backends como argumentos (no los crea internamente).

El manejo de subagents es inteligente: filtra condicionalmente `knowledge-researcher` y `mcp-orchestrator` segun `enable_knowledge` y `enable_mcp`.

### 3.4 Sistema de Memoria

```
VFS (Virtual File System del DeepAgent)
+-- /AGENTS.md          -> StateBackend (ephemeral, contexto del dominio)
+-- /memories/{user_id} -> UserScopedStoreBackend -> PostgresStore
                           PERSISTENTE cross-thread, aprende preferencias

Checkpointer -> AsyncPostgresSaver -> PostgreSQL
(Guarda estado del grafo LangGraph por thread_id)

VL Replay Buffer -> JSONL Local -> Push a Nube
(Traza cada frame + instrucción + acción del Sistema 1 Visual y lo prepara para fine-tuning continuo)
```

La `UserScopedStoreBackend` es una decision muy elegante: al usar `user_id` como namespace (en vez de `thread_id`), las memorias se comparten entre TODAS las conversaciones del mismo usuario.

### 3.5 Temporal Router

El stream tiene un router que analiza la query antes de ejecutar el agente:
- Si la query es "puramente historica" (>6 meses) -> desactiva MCP + Knowledge
- Ahorra tokens y reduce latencia en queries de datos antiguos
- Usa el propio LLM generalist para clasificar (zero-shot JSON output)

---

## 4. Sistema de Herramientas (Tools)

### 4.1 ask_knowledge_agent — RAG Tool

- Extrae `user_id`, `knowledge_base_id`, `session` desde el `RunnableConfig`
- Patron lazy singleton del `SemanticSearcher`
- Manejo de Llama 3.1 hallucinations (nested `parameters` key)
- Trunca chunks a 800 chars para ahorrar tokens
- **Bien implementado**: context-aware via config injection

### 4.2 call_dynamic_mcp — MCP Tool

- Resuelve la configuracion del tool desde DB por nombre
- Soporta 3 transportes: `rest`, `sse` (MCP nativo), `stdio`
- **Smart Filtering**: Pre-filtra datos via `key_values` y `key_figures` **antes** de mapear -> ahorra tokens dramaticamente
- Resolucion robusta de URLs relativas/absolutas con deduplicacion de slashes
- Heuristica de deteccion REST: si la URL contiene `api.` o `/api/` -> fuerza `rest`

---

## 5. MCP Service — Motor de Datos en Tiempo Real

El `MCPService` es la pieza mas compleja y sofisticada del sistema.

### 5.1 Schema Discovery (Zero-Config)

```
_discover_rest_bridge()
    |
    +-- 1. Fetch muestra de respuesta (GET al endpoint)
    +-- 2. _extract_filterable_schema() -> detecta campos numericos vs categoricos
    +-- 3. LLM analiza el endpoint -> genera description, params, response_fields
    +-- 4. Construye parameter_schema completo con filterable_schema embebido
    +-- 5. Retorna tool_def para registro en DB
```

Esto significa que el sistema puede **auto-descubrir** cualquier REST API y volverlo un tool del agente, sin tocar codigo.

### 5.2 Filtrado Pre-LLM

La aplicacion de filtros se hace **antes** del mapeo KeyFigures/KeyValues:
```
Raw API data -> _apply_filters() -> data filtrada -> _auto_map_response()
```
Esto es arquitectonicamente correcto: filtra en Python O(n), no con LLM.

---

## 6. MLOps — Sistema OTA (Over-The-Air con vLLM)

El sistema OTA ha migrado de Ollama+GGUF a **vLLM + safetensors LoRA**. Hay dos servicios OTA paralelos:

**`MLOpsService.process_ota_webhook()`** (modelos de texto):
1. Recibe webhook de ApiLLMOps con `model_tag` (ej: `aura_tenant_01-v2`)
2. Consulta `GET /api/v1/models/{tenant_id}/latest/config` en la Mothership → obtiene `lora_url` (presigned URL del `.tar.gz`)
3. Descarga el `.tar.gz` en streaming con chunks de 1MB (evita saturar RAM)
4. Extrae los safetensors a `./loras/{model_tag}/` usando `tarfile` con `filter='data'` (previene path traversal)
5. Notifica a vLLM via `POST {vllm_host}/v1/load_lora_adapter` para carga dinámica sin reiniciar
6. Limpia archivos temporales en el bloque `finally`

**`VLMLOpsService.process_vl_ota_webhook()`** (modelos Vision-Language):
- Mismo flujo, pero consulta `GET /api/v1/vl/models/{tenant_id}/vl/config`
- Extrae a `./loras/{model_tag}/` e inyecta en vLLM como adaptador VL

**Requisito de vLLM:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` debe estar activo en el contenedor vLLM para que la carga dinámica funcione sin reinicio.

### 6.1 vLLM Multi-LoRA — Configuración y Gestión de Adaptadores

vLLM soporta servir **múltiples adaptadores LoRA simultáneamente** sobre un único modelo base en GPU. La configuración actual del Edge:

```yaml
# docker-compose.yml — servicio vllm
--model Qwen/Qwen3.5-2B     # base model en GPU
--enable-lora               # habilita soporte LoRA
--max-loras 4               # máximo de LoRAs en VRAM simultáneamente
--max-lora-rank 16          # debe coincidir con r=16 del training (Unsloth)
--gpu-memory-utilization 0.85
VLLM_ALLOW_RUNTIME_LORA_UPDATING=true  # permite actualización sin reinicio
volumes:
  - ./loras:/loras          # directorio compartido Edge ↔ vLLM container
```

**Slots ocupados actualmente (2 de 4 disponibles):**

| Alias | Path en container | Uso |
|-------|------------------|-----|
| `aura_expert` | `/loras/aura_tenant_01-v2` | IndustrialAgent (texto, SCADA, RAG) |
| `aura_system1` | `/loras/aura_tenant_01-vl` | Sistema 1 VL (análisis de screenshots) |

Los aliases se definen en `config.py`:
```python
default_llm_model = "aura_expert=/loras/aura_tenant_01-v2"
system1_model     = "aura_system1=/loras/aura_tenant_01-vl"
```

**Ciclo de vida de un adaptador LoRA en vLLM:**

```
Al iniciar vLLM:
  --lora-modules alias=path  ← pre-carga (opcional, no configurado actualmente)

En runtime (OTA update):
  POST /v1/load_lora_adapter  {"lora_name": "aura_expert", "lora_path": "/loras/...", "load_inplace": true}
  └→ load_inplace=true: actualiza el slot existente sin liberar VRAM (zero-downtime)
  └→ load_inplace=false: requeriría unload previo y dejaría un gap de servicio

Para eliminar un adaptador:
  POST /v1/unload_lora_adapter  {"lora_name": "aura_expert"}

Para listar adaptadores cargados:
  GET /v1/models  ← devuelve el base model + todos los LoRA aliases registrados
```

**Cómo el agente selecciona el LoRA en inferencia:**
```python
# LLMFactory genera el cliente apuntando a vLLM con el alias como model name
ChatOpenAI(
    base_url="http://vllm:8000/v1",
    model="aura_expert",        # ← vLLM enruta al LoRA correcto
    api_key="not-needed",
)
```
vLLM recibe la request con `model="aura_expert"`, aplica el adaptador correspondiente sobre `Qwen3.5-2B`, y retorna la respuesta. El base model en GPU se reutiliza para todos los LoRAs — solo los pesos delta del adaptador se suman en cada forward pass.

**Restricción crítica:** `--max-lora-rank 16` es un techo global. Si en el futuro se entrena con `r=32`, el adaptador no cargará. El `r` del training en Unsloth y el `--max-lora-rank` de vLLM deben estar alineados.

---

## 7. DB Collector — Ingesta de Datos

Scheduler APScheduler que ejecuta cron jobs para:
- Conectarse a fuentes de BD externas (MySQL, SQLite, PostgreSQL via connectors/)
- Extraer datos de sensores/telemetria segun intervalo configurado
- Formatearlos y almacenarlos para contexto del agente

---

## 8. Problemas Criticos Encontrados

### 🔴 BUG CRÍTICO: `LLMProvider.OLLAMA` no existe — AttributeError en tool discovery

En `mcp_service.py`, el código hace referencia a `LLMProvider.OLLAMA` que **no existe en el enum `LLMProvider`** del `LLMFactory`. El factory solo tiene `vllm` y `openrouter`. Esto produce un `AttributeError` al intentar usar el AI REST Bridge del MCPService, rompiendo el auto-descubrimiento de APIs REST.

**Fix:** Reemplazar `LLMProvider.OLLAMA` por `LLMProvider.VLLM` en `mcp_service.py`.

---

### 🔴 BUG CRÍTICO: `trigger_training_job` no existe en MothershipClient

`mlops.py` llama `await mothership_client.trigger_training_job(...)` pero ese método **no está definido** en `MothershipClient`. El endpoint `POST /mlops/training/launch` lanza `AttributeError` en producción. Solo existe `trigger_vl_training_job`.

**Fix:** Implementar `trigger_training_job()` en `mothership_client.py` que llame a `POST /api/v1/training/job`.

---

### 🟠 APScheduler captura objeto DbSource stale (datos desactualizados)

El scheduler registra el job con una referencia al objeto `DbSource` en el momento del registro. Si la fuente se actualiza (nuevo cron, nueva query), el job ejecutará con los datos viejos hasta que el scheduler se recargue.

**Fix:** Pasar solo `source.id` al job y hacer un fetch fresh de la BD al inicio de cada ejecución.

---

### 🟠 I/O síncrono en ReplayBuffers y streaming OTA bloquea el event loop

- `replay_buffer.py` y `vl_replay_buffer.py` usan `open()` síncrono para escribir experiencias en JSONL.
- `mlops_service.py` y `vl_mlops_service.py` usan `open()` síncrono para escribir chunks del streaming OTA.

Cada write bloquea el event loop de asyncio, afectando todas las requests concurrentes durante la descarga del modelo (~100–500MB).

**Fix:** Envolver con `asyncio.to_thread()` o usar `aiofiles`.

---

### 🟠 BUG: Duplicacion de código invoke() / stream() (~350 líneas duplicadas)

En `agent_service.py`, los métodos `invoke()` y `stream()` duplican prácticamente la misma lógica de preparación del agente (resolución de modelo, creación de LLM, construcción de tools, merge de prompts). Cualquier cambio debe hacerse en dos lugares.

**Fix:** Extraer `_prepare_agent_context()` con la lógica compartida.

---

### 🟡 AGENTE RECONSTRUIDO EN CADA REQUEST (latencia acumulada)

El grafo LangGraph se compila from scratch en cada `invoke()` y `stream()`. El sistema ya tiene `_GRAPH_CACHE` implementado con hash de configuración, pero solo aplica en algunos paths. Con concurrencia alta puede generar latencia acumulada.

**Verificar:** Que `_GRAPH_CACHE` cubra todos los casos de uso (generalist + industrial agent).

---

### 🟡 SATELLITE AGENTS retornan datos ficticios sin advertencia

Los agentes SAP, Google y Office retornan datos hardcodeados. Con `enable_satellite=True`, el Generalist puede enrutar a SAP y el usuario recibirá datos falsos sin ninguna advertencia.

**Fix:** Agregar flag `coming_soon=True` y bloquear el routing a satellites en producción.

---

### 🟡 Doble conexion DB en call_dynamic_mcp

`call_dynamic_mcp` en `mcp_tool.py` abre su propia sesión de DB en vez de reutilizar la del request. Crea una conexión extra al pool en cada invocación de herramienta.

---

## 9. Fortalezas de la Arquitectura

| Aspecto | Detalle |
|---------|---------|
| **Principios SOLID** | SRP en cada modulo, OCP en subagentes y provider registry |
| **Lazy Loading** | Expert LLM se crea solo si se usa (factory lambda) |
| **Think Block Filtering** | Filtrado de `<think>` en streaming muy robusto y eficiente |
| **Temporal Router** | Evita llamadas costosas para queries historicas |
| **Smart Filtering** | Pre-filtrado Python antes del mapeo KV/KF ahorra tokens |
| **Schema Discovery** | Auto-descubrimiento de APIs REST via LLM + heurisiticas |
| **Dual Memory** | Ephemeral (thread) + Persistent (user) con routing correcto |
| **MLOps OTA** | Actualizacion de modelo sin reiniciar el servidor |
| **Multi-transport MCP** | Soporta REST, SSE, stdio con fallback automatico |
| **Variable Interpolation** | `{{tool.key_values}}` en prompts dinamicos es elegante |
| **LLM Registry (OCP)** | Agregar proveedor = 1 entrada en el dict, sin if/elif |

---

## 10. Diagrama de Flujo — Request Completo (Streaming)

```
Frontend
    | POST /chat/chat/stream {query, thread_id, model_id, kb_id, use_generalist}
    v
chat.py::chat_stream_endpoint()
    | JWT auth -> user_id
    | Guardar ChatMessage (user) en Postgres
    v
AgentService.stream()
    | 1. Resolver model -> LLMProvider + model_name
    | 2. Crear ui_generalist_llm + settings
    | 3. Temporal Router (historica?) -> pode herramientas
    | 4. Expert factory (lambda lazy)
    | 5. Cargar ToolConfigs del usuario de DB
    | 6. Build tools_context string
    | 7. Crear agente (GeneralistOrchestrator o IndustrialAgent)
    v
agent.astream_events(messages, files={AGENTS.md}, config=...)
    | on_chat_model_stream     -> yield token (filtrar <think>)
    | on_tool_start            -> yield {type: subagent, status: running}
    | on_tool_end              -> yield {type: subagent, status: complete}
    | on_chat_model_end        -> yield fallback si no hubo stream
    v
chat.py::event_generator()
    | Acumula full_content
    | Guarda ChatMessage (assistant) en Postgres
    | yield SSE events: meta -> tokens -> done
    v
Frontend
```

---

## 11. Recomendaciones Prioritarias

### P0 — Crítico (bugs reales que rompen funcionalidad)

| # | Problema | Archivo | Acción |
|---|---------|---------|--------|
| 1 | `LLMProvider.OLLAMA` no existe — AI REST Bridge roto | `mcp_service.py` | Cambiar a `LLMProvider.VLLM` |
| 2 | `trigger_training_job` no existe en MothershipClient | `mlops.py`, `mothership_client.py` | Implementar método |
| 3 | APScheduler captura DbSource stale | `scheduler.py` | Pasar `source.id`, fetch fresh en runtime |

### P1 — Importante (estabilidad, calidad y seguridad)

| # | Problema | Archivo | Acción |
|---|---------|---------|--------|
| 4 | I/O síncrono en ReplayBuffers | `replay_buffer.py`, `vl_replay_buffer.py` | `asyncio.to_thread` o `aiofiles` |
| 5 | I/O síncrono en streaming OTA | `mlops_service.py`, `vl_mlops_service.py` | `aiofiles` para escritura por chunks |
| 6 | Satellite agents retornan datos ficticios | `satellite_agents.py` | Flag `coming_soon`, deshabilitar en prod |
| 7 | Doble conexión DB en MCP tool | `tools/mcp_tool.py` | Reutilizar session del config |

### P2 — Mejoras de Performance y Mantenibilidad

| # | Problema | Acción |
|---|---------|--------|
| 8 | invoke() y stream() duplican ~350 líneas | Extraer `_prepare_agent_context()` |
| 9 | Verificar cobertura del `_GRAPH_CACHE` | Asegurar cache en todos los paths de creación de agente |
