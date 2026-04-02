# 🏗️ Auditoría de Arquitectura — IndustrialBackend (Aura AI)

> **Rol:** Arquitecto de Software Senior
> **Proyecto:** `IndustrialBackend` — Edge AI para Seguridad Industrial

---

## 1. Visión General del Sistema

El sistema es un **backend Edge AI** construido sobre FastAPI, que implementa un agente de IA multi-nivel para análisis de documentos industriales y datos en tiempo real. La arquitectura es un sistema **Agentic RAG + MCP** de borde (edge) con soporte multi-modelo.

### Stack Técnico

| Capa | Tecnología |
|------|-----------|
| API | FastAPI 0.115 + uvicorn |
| Agentes | `deepagents` + LangGraph + LangChain |
| LLMs | Ollama (local) + OpenRouter (cloud) |
| RAG | Qdrant (similarity) + nomic-embed-text |
| Persistencia SQL | PostgreSQL + SQLModel + asyncpg |
| Checkpoints de Agente | LangGraph AsyncPostgresSaver |
| Store Long-Term | LangGraph AsyncPostgresStore |
| Blobs | MinIO |
| Scheduler | APScheduler (cron jobs) |
| MLOps | OTA via ApiLLMOps (Mothership) |

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
Modo 1: Expert Direct (default)
-----------------------------------------
 User Query
     |
     v
 [IndustrialAgent] (aura_tenant_01-v2 fine-tuned)
     +-- [knowledge-researcher] SubAgent
     |       +-- ask_knowledge_agent -> Qdrant RAG
     +-- [mcp-orchestrator] SubAgent
     |       +-- call_dynamic_mcp -> APIs REST / MCP SSE / stdio
     +-- [general-assistant] SubAgent (fallback)


Modo 2: Generalist Orchestrator (use_generalist=True)
------------------------------------------------------
 User Query
     |
     v
 [GeneralistOrchestrator] (llama3.1:8b / user-selected model)
     +-- [industrial-expert] Tool -> IndustrialAgent (lazy, on demand)
     |       +-- (identico al Modo 1)
     +-- [sap-agent] Tool -> check_inventory (PLACEHOLDER)
     +-- [google-agent] Tool -> google_search (PLACEHOLDER)
     +-- [office-agent] Tool -> read_outlook_email (PLACEHOLDER)
```

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

## 6. MLOps — Sistema OTA

El `MLOpsService.process_ota_webhook()` implementa:

1. Obtiene presigned URLs del Mothership (ApiLLMOps)
2. Descarga .gguf en streaming (evita saturar RAM con modelos de 3GB+)
3. Calcula SHA256 y sube blob a Ollama
4. Registra el modelo con tool-calling template (Qwen2.5 format)
5. Limpia archivos temporales en el `finally` block

Integracion clara entre el edge (este backend) y el MLOps cloud (ApiLLMOps).

---

## 7. DB Collector — Ingesta de Datos

Scheduler APScheduler que ejecuta cron jobs para:
- Conectarse a fuentes de BD externas (MySQL, SQLite, PostgreSQL via connectors/)
- Extraer datos de sensores/telemetria segun intervalo configurado
- Formatearlos y almacenarlos para contexto del agente

---

## 8. Problemas Criticos Encontrados

### BUG MAYOR: Duplicacion de LLMs en invoke()

En `agent_service.py` lineas 294-356, se crea `ui_generalist_llm` y `worker_llm` **dos veces**. El segundo bloque sobreescribe el primero silenciosamente, desperdiciando 2 llamadas al factory y una query a DB.

```python
# Primera creacion (lines 294-320) - DESPERDICIADA:
ui_generalist_llm = await LLMFactory.get_llm(...)
worker_llm = await LLMFactory.get_llm(...)

# Segunda creacion (lines 350-356) - la que realmente se usa:
ui_generalist_llm = await LLMFactory.get_llm(...)
worker_llm = ui_generalist_llm
```

En `stream()` no hay duplicacion (esta limpio).

### BUG EN PROMPT: AGENTS_MD_CONTENT Corrupto

En `app/domain/agent/prompts/industrial.py` lineas 36-41, el `AGENTS_MD_CONTENT` tiene **codigo Python embebido accidentalmente**. Esto se carga en el VFS del agente como memoria de dominio y contamina el contexto:

```python
AGENTS_MD_CONTENT = """...
##    "system_prompt": (    # <-- codigo Python en el markdown!
        "Industrial Data Orchestrator. "
        ...
"""
```

### PROMPT INCOMPLETO: GENERALIST_SYSTEM_PROMPT

El prompt generalist comienza en la **mitad de una oracion** (linea 12 de generalist.py):
```
GENERALIST_SYSTEM_PROMPT = """\
   - **M365/Office data?** ...   # <- le falta todo el encabezado
```
Le falta el bloque de decision (Step 1, Step 2...). El agente generalist no tiene contexto completo para enrutar correctamente.

### PROMPT MCP_SUBAGENT TRUNCADO

En `subagents.py`, el `MCP_SUBAGENT.system_prompt` comienza con:
```python
"in your call to `call_dynamic_mcp`. Only include filters..."
```
Le falta el contexto inicial ("You are an..."). La cadena fue truncada.

### AGENTE RECONSTRUIDO EN CADA REQUEST

El grafo LangGraph se compila from scratch en cada `invoke()` y `stream()`. Compilar tiene overhead de CPU. Con concurrencia alta puede generar latencia acumulada.

**Solucion sugerida**: Cache de agentes compilados keyed por `(model_id, kb_id, source_id)` + TTL.

### CORS NO CONFIGURABLE

```python
allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"]
```
Hardcoded, bloqueara el frontend en produccion. Deberia leerse del `.env`.

### SESSION PASADA DENTRO DEL CONFIG DEL AGENTE

```python
config = {"configurable": {"session": session, ...}}
```
La sesion SQLAlchemy (no serializable) se pasa dentro del config del grafo. Funcionara en el modo actual, pero si el grafo se cachea o serializa, causara errores.

### SATELLITE AGENTS SON PLACEHOLDERS EN PRODUCCION

Los agentes SAP, Google y Office retornan datos ficticios hardcodeados. Con `enable_satellite=True` por defecto en el generalist, puede enrutar al SAP agent y el usuario recibira datos falsos sin advertencia.

### DOBLE CONEXION DB EN call_dynamic_mcp

`call_dynamic_mcp` en `mcp_tool.py` abre su propia sesion de DB (`async with async_session_factory() as session`) en vez de reutilizar la sesion del request que viene por config. Crea una conexion extra innecesaria al pool.

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

### P0 — Critico (bugs reales que afectan comportamiento)

| # | Problema | Archivo | Accion |
|---|---------|---------|--------|
| 1 | Doble creacion de LLMs en invoke() | agent_service.py:294-320 | Eliminar el primer bloque LLM creation |
| 2 | AGENTS_MD_CONTENT tiene codigo Python embebido | prompts/industrial.py:36-41 | Limpiar el markdown |
| 3 | GENERALIST_SYSTEM_PROMPT incompleto | prompts/generalist.py | Agregar encabezado de routing |
| 4 | MCP_SUBAGENT.system_prompt truncado | subagents.py:58 | Completar el contexto inicial |

### P1 — Importante (calidad y seguridad)

| # | Problema | Archivo | Accion |
|---|---------|---------|--------|
| 5 | Satellite agents retornan datos ficticios | satellite_agents.py | Flag "coming_soon", disable en prod |
| 6 | CORS origins hardcodeados | main.py:79 | Leer de settings/env var |
| 7 | Doble conexion DB en MCP tool | tools/mcp_tool.py | Pasar session via config |

### P2 — Mejoras de Performance

| # | Problema | Accion |
|---|---------|--------|
| 8 | Agente compilado cada request | Cache con key (model_id, kb_id, source_id) + TTL |
| 9 | Pool compartido checkpointer+store | Considerar pools independientes |
