# 🌌 Aura AI Ecosystem: Master Architecture Report
## "Industrial Edge-to-Cloud Intelligence"

> **Última revisión:** Abril 2026

Este reporte técnico consolida la auditoría completa de los proyectos `IndustrialBackend` (Nodo Edge) y `ApiLLMOps` (Mothership Cloud), analizando los patrones de diseño, flujos lógicos y la arquitectura de agentes industriales.

---

## 1. Filosofía de Arquitectura: "The Replay Loop"

El ecosistema Aura AI está diseñado bajo la filosofía de **Continuous Edge Learning** + **Macrohard Architecture**:

- **Edge (IndustrialBackend)**: Ejecuta inferencia de baja latencia con un **Sistema 2** (Razonamiento texto/SCADA) y un **Sistema 1** (Ejecución Visual Digital Optimus), recolectando datos operativos y trayectorias de Computer Use.
- **Cloud (ApiLLMOps)**: Consolida datos, realiza Fine-Tuning de texto Y de visión (Vision-Language), y devuelve el conocimiento dual al borde vía OTA (Over-The-Air).
- **El Loop**: Edge aprende de su propio uso → Mothership lo re-entrena → Edge se actualiza sin downtime. Ciclo continuo y autónomo por diseño.

---

## 2. Patrones de Diseño Utilizados (System-Wide)

### 2.1 Factory Pattern (Creación Dinámica)
- **`LLMFactory` (Edge)**: Crea instancias de `ChatOpenAI` apuntando a vLLM local o OpenRouter según el `LLMProvider` configurado en BD. Registry pattern: agregar proveedor = una entrada en el dict (OCP).
- **`DeepAgent Factory` (Edge)**: Ensambla el grafo LangGraph inyectando subagentes, herramientas y memoria según la configuración del request.
- **`ConnectorRegistry` (Edge)**: Instancia el conector de BD correcto (MySQL, Postgres, SQLite, etc.) dinámicamente para el colector.

### 2.2 Bridge & Adapter (Abstracción de Herramientas)
- **`MCPService` (Edge)**: Puente entre el agente y APIs heterogéneas. Transforma protocolos REST, SSE y STDIO a un formato unificado `MCPResponse` (key_figures + key_values).
- **`MinioManager` (Mothership)**: Adapta la API de MinIO como sistema de archivos persistente y registry de modelos.

### 2.3 Registry Pattern (Extensibilidad)
- **Provider Registry**: `_PROVIDER_REGISTRY` dict — agregar Anthropic/Groq = una entrada, cumpliendo OCP.
- **Subagent Registry**: `get_all_subagents()` en `definitions.py` — lista central de subagentes, el factory los itera automáticamente.

### 2.4 Strategy Pattern (Routing de Ejecución)
- **Temporal Router**: Clasificador zero-shot que detecta si una query requiere datos históricos (>6 meses) → cortocircuita RAG y MCP, redirige a `sistema1-experto`. Activo solo en el path de streaming.
- **Generalist Orchestrator**: Router de intents que decide qué especialista invocar. Referencia `sap-agent`, `google-agent` y `office-agent` en su prompt — estos **existen en `satellite.py` en modo DEMO** (retornan placeholders) pero **no están registrados en el orchestrator** aún.

---

## 3. Jerarquía de Agentes (3 Niveles)

```
Generalist Orchestrator   (generalist_model — vLLM Qwen3.5-2B)
  │
  ├── sistema1-experto     (vision_model — vLLM + LoRA aura_system1)
  │     └── sin herramientas — conocimiento baked in fine-tuned weights
  │
  ├── computer-use-agent   (vision_model — misma instancia que sistema1)
  │     └── tools: take_screenshot, execute_action, task_complete
  │
  └── industrial-expert    (expert_model — vLLM + LoRA aura_expert)   ← lazy-loaded
        ├── knowledge-researcher   (RAG sobre Qdrant)
        │     └── tool: ask_knowledge_agent
        ├── mcp-orchestrator       (datos en tiempo real)
        │     └── tool: call_dynamic_mcp
        └── general-assistant      (fallback sin herramientas)
```

**Selección de ruta por query:**

| Query | Agente destino |
|---|---|
| Datos históricos > 6 meses | `sistema1-experto` |
| Screenshot SAP/SCADA/HMI | `sistema1-experto` |
| Acción en pantalla (SAP GUI, formulario) | `computer-use-agent` |
| Sensores en tiempo real, KPIs actuales | `industrial-expert` → `mcp-orchestrator` |
| Manuales, normativas ISO/OSHA/NOM | `industrial-expert` → `knowledge-researcher` |
| Pregunta general | `industrial-expert` → `general-assistant` |

El `AgentService` cachea instancias compiladas del grafo en `_GRAPH_CACHE` (LRU, hasta 100 entradas) con clave `(user_id, model_id, tools_hash, custom_prompt_hash)` — evita recompilar el grafo en cada request del mismo usuario.

---

## 4. Computer Use Agent (Sistema 1 / Digital Optimus Local)

El Computer Use Agent implementa el loop **Observe → Think → Act** de la arquitectura Macrohard:

```
Orchestrator emite instrucción de alto nivel
  ↓
[observe node]  take_screenshot() → imagen base64
  ↓
[think_act node]  VL model recibe imagen + instrucción → tool_call JSON
  ↓  
  ├── execute_action(action_json)   → pyautogui ejecuta click/type/press/scroll
  ├── VLReplayBuffer.append_step()  → guarda (screenshot, instrucción, acción) para training
  └── Si task_complete() → FIN
  ↓
Regresa a [observe] hasta max_steps o task_complete
```

**Modos de operación:**

| Parámetro | DEMO (`computer_use_demo_mode=True`) | Producción |
|---|---|---|
| `take_screenshot` | Imagen estática de `/static/demo/screens/` | Captura real con `mss` |
| `execute_action` | Loguea la acción, no la ejecuta | `pyautogui` ejecuta en pantalla real |
| Requisito | Ninguno | SAP GUI abierto en el edge node |

**Parámetros clave** (`config.py`): `computer_use_max_steps` (máximo steps por tarea), `computer_use_demo_mode` (toggle), `computer_use_enabled` (activa/desactiva el subagente).

---

## 5. VL Replay Buffer — Aprendizaje desde Computer Use

Cada step del Computer Use Agent genera un dato de entrenamiento que se acumula en `data/vl_replay.jsonl`:

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "instrucción"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "{\"type\":\"click\",\"x\":450}"}]}
  ],
  "images": ["<base64_png>"],
  "metadata": {"tool": "computer_use", "timestamp": "..."}
}
```

- **Capacidad**: 2000 steps máximo (FIFO + retención del 20% más reciente).
- **Upload**: `MothershipClient.upload_vl_dataset()` envía el buffer a ApiLLMOps bucket `datalake-vl`.
- **Cierre del loop**: El Celery VL Worker entrena sobre este buffer — el modelo aprende a ver y actuar en pantallas industriales desde sus propias sesiones de uso.

---

## 6. Flujo End-to-End: Ciclo de Datos

### Fase 1: Recolección
1. `db_collector/scheduler.py` — APScheduler cron por fuente, dispara `CollectorService.run_source()`
2. `db_collector/collector_service.py` — desencripta conexión, ejecuta query, aplica delta slicing por posición acumulada, formatea con `rows_to_sharegpt()`
3. `mothership_client.py` — `upload_dataset()` (texto SCADA) y `upload_vl_dataset()` (trayectorias Computer Use)

### Fase 2: Fine-Tuning en la Nube (ApiLLMOps)
4. `datasets.py` — recibe `.jsonl`, guarda como partición UUID independiente en bucket `datalake`
5. `unsloth_trainer.py` / `vl_trainer.py` (Celery GPU) — consolida por prefijo tenant, entrena QLoRA 4-bit (r=16), exporta safetensors → `.tar.gz` → bucket `models`
6. Worker dispara webhook OTA al Edge: `POST {edge_public_url}/mlops/webhook/model-ready`

### Fase 3: Despliegue OTA
7. `mlops.py` (endpoint) — valida `model_tag` con regex, despacha a `BackgroundTask`
8. `mlops_service.py` / `vl_mlops_service.py` — obtiene presigned URL de la Mothership, descarga `.tar.gz` en streaming, extrae a `./loras/{model_tag}/`, notifica vLLM: `POST /v1/load_lora_adapter` (carga dinámica, sin reiniciar)
9. Próximas inferencias usan el LoRA actualizado via alias (`aura_expert` o `aura_system1`)

---

## 7. Arquitectura de Memoria Jerárquica

| Tipo | Tecnología | Archivo | Alcance |
|---|---|---|---|
| **Episódica (corto plazo)** | `AsyncPostgresSaver` (LangGraph) | `memoryAI/checkpointer.py` | Por `thread_id` — historial exacto del chat |
| **Semántica (conocimiento)** | Qdrant (vector search) | `persistence/vector.py` | Por usuario — manuales ISO/OSHA vectorizados |
| **Dominio (VFS)** | DeepAgents VFS | `agent/prompts/industrial.py` | Global — `AGENTS.md` inyectado en cada turno |
| **Usuario (long-term)** | `AsyncPostgresStore` (LangGraph) | `agent/memory/backends.py` | Por `user_id` — cross-thread, persistente |

La memoria de usuario es implementada por `UserScopedStoreBackend` en `backends.py`, que sobreescribe el namespace del `StoreBackend` de DeepAgents usando `user_id` en lugar de `thread_id`. Archivos guardados en `/memories/` son accesibles en cualquier conversación futura del mismo usuario.

---

## 8. Multi-LoRA vLLM + AI REST Bridge

### 8.1 Multi-LoRA
vLLM sirve múltiples LoRAs sobre un único base model (`Qwen3.5-2B`). Los dos slots activos:

| Alias | Ruta | Usado por |
|---|---|---|
| `aura_expert` | `/loras/aura_tenant_01-v2/` | IndustrialExpert (SCADA, RAG) |
| `aura_system1` | `/loras/aura_tenant_01-vl/` | Sistema1 + ComputerUse |

Config: `--max-loras 4`, `--max-lora-rank 16`. OTA usa `load_inplace=True` para swap atómico sin downtime.

### 8.2 AI REST Bridge (Auto-Discovery)
`MCPService._discover_rest_bridge()` permite registrar cualquier API REST sin configuración manual de schema:
1. Hace GET a la URL para obtener una muestra de respuesta
2. Extrae campos filtrables (`key_figures` numéricos, `key_values` categóricos)
3. Invoca un LLM para inferir descripción, parámetros y `response_fields`
4. Persiste el schema generado en `ToolConfig.parameter_schema` (BD)
5. El agente usa ese schema para saber qué filtros aplicar en cada llamada

---

## 9. Auditoría de Seguridad y Token Management

### 9.1 Token Pruning (Ahorro de Contexto)
`MCPService` aplica filtros `key_figures` / `key_values` en Python **antes** de que el LLM vea los datos crudos. Impacto típico: JSON crudo (20k tokens) → resumen estructurado (<500 tokens).

### 9.2 JWT + API Key Combo
- **Usuario final**: JWT HS256 + `HTTPBearer` en `deps.py`
- **Comunicación Edge → Mothership**: Header `x-api-key`
- **Comunicación Mothership → Edge (webhook OTA)**: mismo `mothership_api_key`

> [!WARNING]
> **Riesgo activo:** El mismo secreto se usa **bidireccionalmente**. Un atacante que lo obtenga puede subir datasets falsos Y disparar actualizaciones OTA maliciosas. Solución: separar en `edge_upload_key` + `cloud_webhook_key`.

---

## 10. Bugs Críticos Identificados (9 activos)

| ID | Ubicación | Descripción | Impacto |
|---|---|---|---|
| **P0-1** | `agent_service.py:312` | `LLMProvider.OLLAMA` no existe en el enum → `AttributeError` en `invoke()` | Non-streaming completamente roto |
| **P0-2** | `mothership_client.py` | `trigger_training_job()` no implementado → `AttributeError` en `POST /mlops/training/launch` | Disparo manual de training roto |
| **P0-3** | `mcp_service.py:479` | `LLMProvider.OLLAMA` en REST Bridge → `AttributeError` en auto-discovery | Registro de APIs REST vía AI Discovery roto |
| **P0-4** | `agent_service.py:176` | `lines.append` fuera del `for kv_field` loop → solo el último campo categorical visible en el contexto del agente | Routing MCP degradado silenciosamente |
| **P1-5** | `agent_service.py:507` | Temporal Router solo en `stream()`, no en `invoke()` | Queries históricas en modo non-streaming nunca llegan a Sistema1 |
| **P1-6** | `docker-compose.yml:56` | `--max-model-len 2048` — demasiado restrictivo; conversaciones con historial + tools superan el límite | Context overflow frecuente en producción |
| **P1-7** | `collector_service.py:151` | `open()` síncrono en función async → bloquea event loop | Degrada latencia de todas las requests concurrent |
| **P1-8** | `vl_replay_buffer.py` | `open()` síncrono con screenshots base64 → bloquea event loop | Especialmente grave al guardar imágenes grandes |
| **P2-9** | `mlops_service.py` + `vl_mlops_service.py` | Sin flag de idempotencia OTA — doble webhook corrompe `/loras/` | Corrupción silenciosa de adaptador |

---

## 11. Visión y Roadmap del Sistema

### Estado actual
- ✅ IndustrialExpert (RAG + MCP tiempo real) — operativo
- ✅ Sistema1 (histórico + VL) — operativo con vLLM LoRA
- ✅ Computer Use Agent — operativo en DEMO mode
- ✅ Loop MLOps texto + VL — arquitectura completa, bugs P0 pendientes
- 🚧 Satellite Agents (SAP, Google, Office) — scaffolding en `satellite.py`, connectors no configurados

### Objetivos de corto plazo (resolver P0/P1)
1. Corregir `LLMProvider.OLLAMA` → `LLMProvider.VLLM` en `agent_service.py` y `mcp_service.py`
2. Implementar `trigger_training_job()` en `MothershipClient`
3. Subir `--max-model-len` a 8192+ en docker-compose
4. Envolver I/O síncrono en `asyncio.to_thread()`

### Objetivos de mediano plazo
- **Computer Use en producción**: activar `mss` + `pyautogui` sobre SAP GUI real del edge node
- **Loop MLOps autónomo**: disparo automático de training al acumular N nuevas filas (sin intervención manual)
- **Satellite agents activados**: configurar conectores reales para SAP (RFC/REST), Google Workspace API, Microsoft Graph API

### Objetivos de largo plazo
- **Multi-tenant**: múltiples instancias edge por Mothership, gestión de LoRA slots por tenant
- **OTA idempotente**: flag de estado por `model_tag`, reintentos seguros con ETag de MinIO
- **Secretos separados**: `edge_upload_key` vs `cloud_webhook_key` para eliminar el riesgo bidireccional
- **Temporal Router en `invoke()`**: paridad completa entre modo streaming y non-streaming

---

## 12. Conclusiones y Evaluación del Arquitecto

El ecosistema Aura AI es **robusto, desacoplado y de grado industrial**. La combinación Unsloth (cloud GPU) + vLLM multi-LoRA (edge) permite actualizar modelos en producción sin downtime — solo el adaptador (~100MB) se descarga, no los pesos base (~2GB).

| Categoría | Evaluación | Nota |
|---|---|---|
| **Escalabilidad** | Alta — Celery + Docker + Edge/Cloud separado | 9/10 |
| **Separación de Concerns** | Excelente — Domain / Persistence / API correctamente separados | 10/10 |
| **Inteligencia de Herramientas** | Superior — Smart Filter + AI Schema Discovery | 10/10 |
| **Bugs / Estabilidad** | 4 bugs P0 críticos rompen flujos core (training, non-streaming, AI discovery) | 5/10 |
| **Visión arquitectónica** | Completa — VL Replay Loop, Computer Use, multi-LoRA, satélites planificados | 9/10 |

> [!IMPORTANT]
> ### Recomendación Final
> El sistema está a **4 fixes de ser operacional en producción**. Los P0 son correcciones de 1-5 líneas de código cada uno. Una vez resueltos, el loop completo (SCADA → training → OTA → inferencia mejorada) funciona de forma autónoma. La arquitectura subyacente es sólida y escala sin cambios estructurales.
