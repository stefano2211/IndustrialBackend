# 🌌 Aura AI Ecosystem: Master Architecture Report
## "Industrial Edge-to-Cloud Intelligence"

> **Última revisión:** Abril 2026

Este reporte técnico consolida la auditoría completa de los proyectos `IndustrialBackend` (Nodo Edge) y `ApiLLMOps` (Mothership Cloud), analizando los patrones de diseño, flujos lógicos y la arquitectura de agentes industriales.

---

## 1. Filosofía de Arquitectura: "The Replay Loop"

El ecosistema Aura AI está diseñado bajo la filosofía de **Continuous Edge Learning** + **Macrohard Architecture**.
- **Edge (IndustrialBackend)**: Ejecuta inferencia de baja latencia con un Sistema 2 (Razonamiento texto/SCADA) y un Sistema 1 (Ejecución Visual Digital Optimus), recolectando datos operativos.
- **Cloud (ApiLLMOps)**: Consolida datos, realiza Fine-Tuning de texto Y de visión (Vision-Language), y devuelve el conocimiento dual al borde vía OTA (Over-The-Air).

---

## 2. Patrones de Diseño Utilizados (System-Wide)

El sistema destaca por un uso riguroso de patrones de software modernos que aseguran escalabilidad y desacoplamiento.

### 2.1 Factory Pattern (Creación Dinámica)
Utilizado en todo el stack para instanciar componentes pesados sin acoplamiento:
- **`LLMFactory` (Edge)**: Crea instancias de `ChatOpenAI` (apuntando a vLLM local) o `ChatOpenAI` (apuntando a OpenRouter) basándose en el `LLMProvider` configurado en BD. Sigue el patrón Registry: agregar un proveedor nuevo = una entrada en el dict (OCP).
- **`DeepAgent Factory` (Edge)**: Ensambla el grafo de LangGraph, inyectando subagentes, herramientas y memoria según la configuración del request.
- **`ConnectorRegistry` (Edge)**: Instancia el conector de DB correcto (MySQL, Postgres, etc.) dinámicamente para el colector.

### 2.2 Bridge & Adapter (Abstracción de Herramientas)
- **`MCPService` (Edge)**: Actúa como un puente (Bridge) entre el agente y APIs heterogéneas. Transforma protocolos REST, SSE y STDIO a un formato unificado que el LLM entiende.
- **`MinioManager` (Ambos)**: Adapta la API de MinIO para ser tratada como un sistema de archivos persistente o un registry de modelos.

### 2.3 Registry Pattern (Extensibilidad)
- **Provider Registry**: Permite agregar nuevos proveedores de LLM (e.g., Anthropic, Groq) simplemente agregando una entrada en un diccionario, cumpliendo con el principio **Open/Closed (OCP)**.
- **Subagent Registry**: El `DeepAgent` carga subagentes de una lista centralizada, facilitando la adición de agentes especializados (Mantenimiento, Legal, etc.).

### 2.4 Strategy Pattern (Routing de Ejecución)
- **Temporal Router**: Decide la estrategia de búsqueda Basándose en la antigüedad de la query (Query histórica -> No buscar en tiempo real).
- **Generalist Orchestrator**: Una estrategia de routing de intents que decide si el experto industrial debe ser invocado o si basta con un agente satélite (Google/SAP).

---

## 3. Auditoría Lógica End-to-End: Flujo de Archivos

A continuación, el recorrido lógico de una pieza de información a través de los archivos del sistema.

### Fase 1: Recolección (Ingesta de Conocimiento + Visual)
1.  `app/persistence/db_source_repository.py` (Edge): Datos SQL de telemetría, y `vl_replay_buffer.py` para recolección de capturas SAP.
2.  `app/domain/db_collector/scheduler.py` (Edge): Dispara el job de base de datos.
3.  `app/core/mothership_client.py` (Edge): Realiza `upload_dataset` (texto) y `upload_vl_dataset` (capturas visuales) a la nube.

### Fase 2: Fine-Tuning en la Nube
5.  `app/api/endpoints/datasets.py` (Mops): Recibe el `.jsonl` y lo guarda como una **partición UUID independiente** en MinIO bucket `datalake`. No hay append de master file.
6.  `app/api/endpoints/training.py` / `vl_training.py` (Mops): Encola el job en Celery y retorna un `job_id`.
7.  `app/domain/services/unsloth_trainer.py` / `vl_trainer.py` (Mops - Celery Worker GPU):
    - Agrega todos los archivos del tenant por prefijo desde MinIO.
    - Carga pesos base desde cache (`HF_HOME`) con `HF_HUB_ENABLE_HF_TRANSFER=1`.
    - Entrena con **QLoRA 4-bit** (r=16, alpha=32) via Unsloth.
    - Exporta **safetensors LoRA** (no GGUF) → comprime en `.tar.gz` → sube a bucket `models`.
    - Lanza Webhook OTA de retorno al Edge con `x-api-key`.

### Fase 3: Despliegue OTA y Ejecución
8.  `app/api/endpoints/mlops.py` (Edge): Capta el webhook con `model_type` (`text` / `vision`). Valida el `model_tag` con regex. Despacha a `BackgroundTask` según el tipo.
9.  `app/domain/services/mlops_service.py` y `vl_mlops_service.py` (Edge):
    - Consultan al registry de la Mothership para obtener la **presigned URL** del `.tar.gz` de safetensors.
    - Descargan el tarball en streaming (chunks 1MB) a `/tmp/`.
    - Extraen los pesos a `./loras/{model_tag}/` con `filter='data'` (previene path traversal).
    - Notifican a vLLM: `POST /v1/load_lora_adapter` — **carga dinámica sin reiniciar**.
    - Limpian archivos temporales en el bloque `finally`.
10. `app/domain/services/agent_service.py` (Edge): Las próximas conversaciones usan el alias LoRA actualizado (`aura_expert=/loras/aura_tenant_01-v2` o `aura_system1=/loras/aura_tenant_01-vl`).

---

## 4. El Cerebro: Arquitectura de Memoria Jerárquica

Aura AI implementa una de las arquitecturas de memoria más completas en proyectos Python:

1.  **Memoria de Corto Plazo (Episódica)**:
    - **Archivo**: `app/persistence/memoryAI/checkpointer.py`
    - **Tecnología**: `AsyncPostgresSaver` (LangGraph).
    - **Logica**: Mantiene el historial exacto del chat por `thread_id`.

2.  **Memoria Semántica (Conocimiento)**:
    - **Archivo**: `app/persistence/vector.py`
    - **Tecnología**: Qdrant.
    - **Logica**: Vectoriza manuales de seguridad (OSHA/ISO) para RAG.

3.  **Memoria de Dominio (VFS)**:
    - **Archivo**: `app/domain/agent/middleware/memory_manager.py`
    - **Logica**: Inyecta un archivo virtual `AGENTS.md` con reglas del sistema que el agente lee antes de cada turno.

4.  **Memoria Persistente de Usuario (Long-Term)**:
    - **Archivo**: `app/domain/agent/memory/user_profile.py`
    - **Tecnología**: `AsyncPostgresStore`.
    - **Logica**: Guarda preferencias transversales del usuario (ej: "Prefiero respuestas técnicas") que persisten entre diferentes chats.

---

## 5. Auditoría de Seguridad y Token Management

### 5.1 Token Pruning (Ahorro de Costos)
- En `mcp_service.py`, el sistema aplica filtros por `key_figures` en Python **antes** de que el LLM vea los datos crudos. 
- **Impacto**: Reduce el contexto de 20,000 tokens (JSON crudo de API) a < 500 tokens (Resumen estructurado).

### 5.2 JWT + API Key Combo
- El sistema utiliza **JWT** para el usuario final (`app/api/deps.py`) pero usa **MOTHERSHIP_API_KEY** para la comunicación entre servicios. Un atacante con el JWT del usuario no puede disparar un entrenamiento; solo la Mothership autenticada puede hacerlo.
- **Riesgo identificado:** El mismo secreto (`mothership_api_key`) se usa **bidirecccionalmente** — el Edge lo envía en uploads y el Mothership lo envía en webhooks. Si se compromete, un atacante puede tanto subir datos falsos como enviar webhooks OTA maliciosos. Se recomienda separar en dos secretos distintos.

---

## 6. Conclusiones y Evaluación del Arquitecto

El ecosistema Aura AI es **robusto, desacoplado y profesional**. No es un simple MVP; es una infraestructura de IA de grado industrial.

| Categoría | Evaluación | Nota |
|-----------|------------|------|
| **Escalabilidad** | Alta (Celery + Docker + arquitectura Edge/Cloud separada) | 9/10 |
| **Separación de Concernos** | Excelente (Domain logic vs Persistence) | 10/10 |
| **Inteligencia de Herramientas** | Superior (Smart Filter + Schema Discovery) | 10/10 |
| **Puntos de Falla** | Loop MLOps roto (4 bugs críticos), I/O síncrono en OTA | 6/10 |

> [!IMPORTANT]
> ### Recomendación Final de Negocio
> El proyecto está listo para escalonamiento masivo (Multi-tenant) una vez resueltos los 4 bugs críticos del loop MLOps. La decisión de usar **Unsloth** en la nube y **vLLM con adaptadores LoRA dinámicos** en el borde es la combinación correcta: permite actualizar el modelo en producción sin downtime y sin re-descargar pesos base (~2GB), solo el adaptador (~100MB).
