# Auditoria Cross-Domain de la Capa de Persistencia

## Resumen Ejecutivo

La capa de persistencia (`app/persistence/`) mantiene un **aislamiento limpio** entre dominios proactivo y reactivo: no existen imports cruzados indebidos entre `persistence/proactiva/` y `persistence/reactiva/`. La infraestructura base (PostgreSQL, Qdrant, MinIO) es compartida y correctamente configurada con namespaces aislados (colecciones, buckets, tablas separadas). Sin embargo, se detectan **tres hallazgos de deuda tecnica**: duplicacion exacta del `UserRepository`, una fuga cross-domain en `SettingsRepository` (shared depende de proactiva), y una inconsistencia de naming en la tabla proactiva (`mcpsource` vs convencion `mcp_source`).

---

## 1. Infraestructura Base (Shared)

### 1.1 PostgreSQL (`app/persistence/db.py`)

| Aspecto | Estado | Detalle |
|---------|--------|---------|
| Engine | Singleton | `create_async_engine` una sola vez a nivel de modulo |
| Session Factory | Singleton | `async_session_factory` con `expire_on_commit=False` |
| Tablas registradas | Completo | Importa TODOS los SQLModel de proactiva, reactiva y shared para `metadata.create_all` |
| Aislamiento de schemas | Correcto | `Event` (reactiva) y `Conversation` (proactiva) comparten el mismo engine pero tablas distintas |

**Observacion**: El `db.py` importa explicitamente todos los modelos (lineas 10-23) para que `SQLModel.metadata.create_all()` cree las tablas de ambos dominios. Esto es el patron correcto en SQLModel.

### 1.2 Qdrant (`app/persistence/vector.py` + `app/persistence/reactiva/reactive_vector.py`)

| Componente | Coleccion | Aislamiento |
|------------|-----------|-------------|
| `QdrantManager` | `settings.qdrant_collection` (proactiva) | Filtro por `metadata.user_id` |
| `ReactiveQdrantManager` | `settings.reactive_qdrant_collection` | Hereda de `QdrantManager`, solo overridea `collection_name` |

**Estado**: Aislamiento correcto. `ReactiveQdrantManager` no duplica logica de conexion; solo redefine la coleccion y fuerza re-check de existencia (`_initialized = False`).

### 1.3 MinIO (`app/persistence/blob.py` + `app/persistence/reactiva/reactive_blob.py`)

| Componente | Bucket | Aislamiento |
|------------|--------|-------------|
| `MinIOClient` | `settings.minio_bucket` (proactiva) | Singleton `minio_client` |
| `ReactiveMinIOClient` | `settings.reactive_minio_bucket` | Hereda de `MinIOClient`, overridea `bucket` |

**Estado**: Aislamiento correcto. Mismo patron que Qdrant.

---

## 2. Repositorios por Dominio

### 2.1 Dominio Proactivo (`app/persistence/proactiva/repositories/`)

| Repositorio | Modelo(s) | Scope | Estado |
|-------------|-----------|-------|--------|
| `ConversationRepository` | `Conversation`, `ChatMessage` | `user_id` | Correcto |
| `DbSourceRepository` | `DbSource` | Global (no user filter) | Correcto — schema en `domain/schemas/db_source` (shared) |
| `KnowledgeRepository` | `KnowledgeBase`, `KnowledgeDocument` | `user_id` | Correcto |
| `LLMConfigRepository` | `LLMConfig` | Global (role-based) | Correcto |
| `MCPSourceRepository` | `MCPSource` | `user_id` | Correcto |
| `ModelRepository` | `Model` | Global | Correcto |
| `PromptRepository` | `Prompt` | Global (con `is_enabled`) | Correcto |
| `ToolConfigRepository` | `ToolConfig` | `user_id` (via join con `MCPSource`) | Correcto |
| `UserRepository` | `User` | Global | **DUPLICADO** (ver seccion 4) |

### 2.2 Dominio Reactivo (`app/persistence/reactiva/repositories/`)

| Repositorio | Modelo(s) | Scope | Estado |
|-------------|-----------|-------|--------|
| `EventRepository` | `Event` | `tenant_id` (via param) | Correcto |
| `ReactiveKnowledgeRepository` | `ReactiveKnowledgeBase`, `ReactiveKnowledgeDocument` | `tenant_id` | Correcto |
| `ReactiveMCPSourceRepository` | `ReactiveMCPSource` | `tenant_id` | Correcto |
| `ReactiveToolConfigRepository` | `ReactiveToolConfig` | System-scoped (no user/tenant join, solo `get_all()`) | Correcto — notar que NO filtra por tenant a diferencia de los otros reactivos |

### 2.3 Infraestructura Compartida (`app/persistence/shared/`)

| Repositorio | Modelo(s) | Scope | Estado |
|-------------|-----------|-------|--------|
| `SettingsRepository` | `SystemSettings` | Global (single row, id=1) | **FUGA CROSS-DOMAIN** (ver seccion 4) |
| `UserRepository` | `User` | Global | Correcto en teoria, pero **DUPLICADO** con proactiva |

---

## 3. Schemas y Tablas SQLModel

### 3.1 Tablas Proactivas

| Tabla | Archivo Schema | Observacion |
|-------|---------------|-------------|
| `conversation` | `domain/proactiva/schemas/conversation.py` | Correcto |
| `knowledge_base` | `domain/proactiva/schemas/knowledge.py` | Correcto |
| `knowledge_document` | `domain/proactiva/schemas/knowledge.py` | Correcto |
| `prompt` | `domain/proactiva/schemas/prompt.py` | Correcto |
| `llm_config` | `domain/proactiva/schemas/llm_config.py` | Correcto |
| `mcpsource` | `domain/proactiva/schemas/mcp_source.py` | **INCONSISTENCIA NAMING**: tabla se llama `mcpsource` (sin underscore) a diferencia de `reactive_mcp_source` |
| `model` | `domain/proactiva/schemas/model.py` | Correcto |
| `tool_config` | `domain/proactiva/schemas/tool_config.py` | Correcto |

### 3.2 Tablas Reactivas

| Tabla | Archivo Schema | Observacion |
|-------|---------------|-------------|
| `event` | `domain/reactiva/schemas/event.py` | Correcto |
| `reactive_knowledge_base` | `domain/reactiva/schemas/reactive_knowledge.py` | Correcto |
| `reactive_knowledge_document` | `domain/reactiva/schemas/reactive_knowledge.py` | Correcto |
| `reactive_mcp_source` | `domain/reactiva/schemas/reactive_mcp_source.py` | Correcto — con `tenant_id` |
| `reactive_tool_config` | `domain/reactiva/schemas/reactive_tool_config.py` | Correcto — con `source_id` FK a `reactive_mcp_source.id` |

### 3.3 Tablas Compartidas

| Tabla | Archivo Schema | Usado por |
|-------|---------------|-----------|
| `user` | `domain/shared/schemas/user.py` | Ambos dominios (auth) |
| `db_source` | `domain/schemas/db_source.py` | Ambos dominios (DB Collector hibrido) |

**Estado**: Aislamiento de tablas es correcto. No hay colisiones de nombres. La unica inconsistencia es `mcpsource` (proactiva) vs `reactive_mcp_source` (reactiva) — el primero omite el underscore.

---

## 4. Hallazgos de Deuda Tecnica

### P1 — `UserRepository` Duplicado (100% identico)

**Archivos**:
- `app/persistence/proactiva/repositories/user_repository.py`
- `app/persistence/shared/user_repository.py`

**Evidencia**: Ambos archivos son identicos byte-a-byte (importan `User` desde `app.domain.shared.schemas.user`, mismos metodos `get_by_email`, `get_by_id`, `list_all`, `create`, `update`, `delete`).

**Impacto**: Cualquier cambio en `UserRepository` debe aplicarse en dos lugares. Riesgo de divergencia silenciosa.

**Recomendacion**: Eliminar `app/persistence/proactiva/repositories/user_repository.py` y hacer que todo el codigo proactivo importe desde `app.persistence.shared.user_repository`.

### P2 — `SettingsRepository` (shared) depende de `domain.proactiva`

**Archivo**: `app/persistence/shared/settings_repository.py:5-8`

```python
from app.domain.proactiva.schemas.settings import (
    SystemSettings,
    SystemSettingsGeneralUpdate,
    SystemSettingsDocumentsUpdate
)
```

**Impacto**: La capa `shared` tiene una dependencia hacia `domain.proactiva`, rompiendo la regla de que `shared` no debe conocer dominios especificos. Si en el futuro se requiere settings especificos para el dominio reactivo, se complica la refactorizacion.

**Recomendacion**: Migrar `SystemSettings` y sus DTOs a `app.domain.shared.schemas.settings`. Actualizar `SettingsRepository` para importar desde alli.

### P3 — Inconsistencia de Naming: `mcpsource` vs `reactive_mcp_source`

**Archivo**: `app/domain/proactiva/schemas/mcp_source.py:24`

```python
__tablename__ = "reactive_mcp_source"  # Reactivo
__tablename__ = "mcpsource"            # Proactivo
```

**Impacto**: Baja — solo afecta legibilidad y consistencia en la base de datos. No hay bug funcional.

**Recomendacion**: Renombrar a `mcp_source` (con underscore) para alinearse con la convencion de las demas tablas. Requiere migration de base de datos.

### P4 — `ReactiveToolConfigRepository` no filtra por `tenant_id`

**Archivo**: `app/persistence/reactiva/repositories/reactive_tool_config_repository.py:30-34`

```python
async def get_all(self) -> List[ReactiveToolConfig]:
    """Get all reactive tool configs (system-scoped, no user filter)."""
    stmt = select(ReactiveToolConfig)
    result = await self.session.execute(stmt)
    return list(result.scalars().all())
```

**Impacto**: `EventProcessor`/`ReactiveAgentService` cargan TODOS los tool configs reactivos sin filtrar por tenant. En un entorno multi-tenant, un tenant podria ver (o usar via MCP) herramientas de otro tenant.

**Recomendacion**: Agregar parametro `tenant_id` a `get_all()` y join/filter con `ReactiveMCPSource.tenant_id`, o agregar un campo `tenant_id` directamente a `ReactiveToolConfig`.

### P5 — `db.py` importa `DbSource` desde ruta inconsistente

**Archivo**: `app/persistence/db.py:18`

```python
from app.domain.schemas.db_source import DbSource  # noqa: F401
```

**Impacto**: Baja. La ruta `app.domain.schemas.db_source` no sigue la convencion de `app.domain.shared.schemas` ni `app.domain.proactiva.schemas`. Esto sugiere que `db_source.py` esta en un directorio intermedio.

**Recomendacion**: Mover `DbSource` a `app.domain.shared.schemas.db_source` para clarificar que es un modelo compartido.

---

## 5. Validacion Post-Cambios en `reactive_service.py`

Los cambios recientes (cache reactivo rehabilitado, session removida de `configurable`, parser regex) fueron verificados contra la capa de persistencia:

| Cambio | Persistencia Impactada | Estado |
|--------|----------------------|--------|
| Cache reactivo con `_stable_hash(mcp_tools_context)` | `_build_mcp_context(session)` usa `ReactiveToolConfigRepository(session)` | Correcto — la sesion se pasa directamente al repo, no al grafo LangGraph |
| `session` removida de `configurable` | `call_reactive_mcp` y `ask_reactive_knowledge` usan `async_session_factory()` como fallback | Correcto — las tools reactivas abren su propia sesion cuando no reciben una via `configurable` |
| Parser regex multiline | Ninguna | No aplica |

**Verificacion de `_build_mcp_context`**:

```python
# app/domain/reactiva/agent/reactive_service.py:43-58
async def _build_mcp_context(self, session) -> str:
    repo = ReactiveToolConfigRepository(session)  # OK: session viva aqui
    tools = await repo.get_all()
    ...
```

La sesion se usa directamente dentro del metodo `_build_mcp_context` (que corre antes de invocar el grafo), y el resultado (string `context`) se pasa como parametro a `create_reactive_orchestrator`. No hay referencia a la sesion dentro del grafo cacheado.

---

## 6. Comparativa: Duplicacion Justificada vs No Justificada

| Par de Repositorios | Duplicacion | Justificada? | Razon |
|---------------------|-------------|--------------|-------|
| `ToolConfigRepository` / `ReactiveToolConfigRepository` | ~60% (CRUD identico, queries distintas) | **SI** | Tablas y schemas distintos; scope diferente (user vs tenant/system) |
| `MCPSourceRepository` / `ReactiveMCPSourceRepository` | ~50% (CRUD identico, schemas distintos) | **SI** | Tablas distintas; scope user vs tenant |
| `KnowledgeRepository` / `ReactiveKnowledgeRepository` | ~60% (CRUD identico, schemas distintos) | **SI** | Tablas distintas; scope user vs tenant |
| `UserRepository` (proactiva) / `UserRepository` (shared) | **100%** | **NO** | Mismo modelo (`User` en shared), mismo codigo. El proactivo deberia usar el shared. |
| `QdrantManager` / `ReactiveQdrantManager` | ~5% (solo collection name) | **SI** | Herencia correcta, zero duplicacion de logica |
| `MinIOClient` / `ReactiveMinIOClient` | ~5% (solo bucket name) | **SI** | Herencia correcta |

---

## 7. Recomendaciones de Refactorizacion

### Inmediato (esta semana)

1. **P1 — Eliminar `UserRepository` duplicado**:
   - Borrar `app/persistence/proactiva/repositories/user_repository.py`
   - Actualizar todos los imports en el dominio proactivo para usar `app.persistence.shared.user_repository`

2. **P4 — Agregar filtro `tenant_id` a `ReactiveToolConfigRepository.get_all()`**:
   - Agregar parametro opcional `tenant_id: str = None`
   - Si se provee, hacer join con `ReactiveMCPSource` y filtrar por `tenant_id`

### Corto Plazo (1-2 semanas)

3. **P2 — Mover `SystemSettings` a shared**:
   - Crear `app/domain/shared/schemas/settings.py`
   - Migrar `SystemSettings`, `SystemSettingsGeneralUpdate`, `SystemSettingsDocumentsUpdate`
   - Actualizar `SettingsRepository` y todos los endpoints que usan estos schemas

4. **P3 — Renombrar tabla `mcpsource` a `mcp_source`**:
   - Cambiar `__tablename__` en `app/domain/proactiva/schemas/mcp_source.py`
   - Crear migration de Alembic (o script SQL) para renombrar la tabla

### Mediano Plazo (1 mes)

5. **P5 — Mover `DbSource` a `domain/shared/schemas`**:
   - Mover archivo y actualizar imports en `db.py`, `db_source_repository.py`, y DB Collector

6. **Consolidar base CRUD opcional**:
   - Considerar una clase base `BaseRepository[T]` con metodos `get_by_id`, `create`, `update`, `delete` para reducir el ~60% de duplicacion en CRUDs de repositorios proactivos/reactivos. Esto es un trade-off: gana mantenibilidad pero pierde flexibilidad para queries custom.

---

## 8. Cambios Aplicados (2026-04-30)

| Hallazgo | Estado | Archivos Modificados | Detalle del Cambio |
|----------|--------|---------------------|-------------------|
| **P1** `UserRepository` duplicado | **RESUELTO** | `persistence/proactiva/repositories/user_repository.py` (eliminado) | Dead code — ningun archivo lo importaba. El unico `UserRepository` activo es `persistence/shared/user_repository.py`. |
| **P2** `SettingsRepository` depende de `domain.proactiva` | **RESUELTO** | `domain/shared/schemas/settings.py` (creado), `domain/proactiva/schemas/settings.py` (eliminado), `persistence/shared/settings_repository.py`, `api/proactiva/endpoints/admin.py`, `core/llm.py`, `domain/proactiva/agent/retrieval/searcher.py`, `domain/proactiva/ingestion/pipeline.py`, `api/proactiva/endpoints/auth.py`, `api/proactiva/endpoints/models.py` | `SystemSettings` migrado a `domain/shared/schemas/settings.py`. Todos los imports actualizados. |
| **P3** Tabla `mcpsource` sin underscore | **RESUELTO** (schema) | `domain/proactiva/schemas/mcp_source.py`, `domain/proactiva/schemas/tool_config.py` | `__tablename__` cambiado a `"mcp_source"`. FK en `ToolConfig.source_id` actualizada a `"mcp_source.id"`. **Pendiente**: migration de base de datos (Alembic/SQL) para renombrar tabla existente. |
| **P4** `ReactiveToolConfigRepository.get_all()` sin tenant | **RESUELTO** | `persistence/reactiva/repositories/reactive_tool_config_repository.py`, `domain/reactiva/agent/reactive_service.py` | `get_all(tenant_id=None)` agregado con join a `ReactiveMCPSource`. `_build_mcp_context` pasa `tenant_id` al repo. Cache key del grafo reactivo ya incluye tenant. |
| **P5** `DbSource` en ruta inconsistente | **RESUELTO** | `domain/shared/schemas/db_source.py` (creado), `domain/schemas/db_source.py` (eliminado), `persistence/db.py`, `persistence/proactiva/repositories/db_source_repository.py`, `domain/proactiva/db_collector/collector_service.py`, `domain/proactiva/db_collector/connectors/registry.py`, `api/proactiva/endpoints/db_collector.py` | Archivo movido a `domain/shared/schemas/db_source.py`. Todos los imports migrados. |
| — `SystemSettings` no registrado en `db.py` | **RESUELTO** | `persistence/db.py` | Agregado import `SystemSettings` para que `init_db()` cree la tabla `systemsetting`. |
| — Bug de imports `settings_repository` desde ruta inexistente | **RESUELTO** | `core/llm.py`, `api/proactiva/endpoints/auth.py`, `api/proactiva/endpoints/models.py`, `api/proactiva/endpoints/admin.py`, `domain/proactiva/ingestion/pipeline.py`, `domain/proactiva/agent/retrieval/searcher.py` | Se importaba desde `persistence.proactiva.repositories.settings_repository` (archivo inexistente). Corregido a `persistence.shared.settings_repository`. |

---

*Documento generado el 2026-04-30. Stack: FastAPI + SQLModel + PostgreSQL + Qdrant + MinIO.*
