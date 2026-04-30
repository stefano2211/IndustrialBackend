# Auditoria Completa de Endpoints: Proactivo vs Reactivo

## Resumen Ejecutivo

**Dominio Proactivo**: 14 modulos, **66 endpoints** — cobertura completa de auth, usuarios, chat, conversaciones, conocimiento, documentos, prompts, modelos, tools, MCP sources, admin, MLOps, DB Collector y system health.

**Dominio Reactivo**: 3 modulos, **24 endpoints** (post-fix) — cobertura completa de eventos, conocimiento reactivo y fuentes MCP reactivas.

**Cambios aplicados**: Se resolvieron **7 problemas criticos de seguridad** (endpoints reactivos sin autenticacion) y se agregaron **7 endpoints faltantes** para alcanzar paridad funcional con los repositories.

---

## 1. Dominio Proactivo — Endpoints (66 rutas)

| Modulo | Rutas | Auth | Observacion |
|--------|-------|------|-------------|
| `auth.py` | 2 (login, signup) | No / JWT | Correcto |
| `users.py` | 2 (me, update me) | Bearer JWT | Correcto |
| `chat.py` | 2 (chat, stream) | Bearer JWT | Correcto |
| `conversations.py` | 5 (CRUD + messages) | Bearer JWT | Correcto |
| `knowledge.py` | 6 (KB CRUD + docs) | Bearer JWT | Correcto |
| `documents.py` | 4 (upload, list, get, delete) | Bearer JWT | Correcto |
| `tools.py` | 6 (CRUD + MCP discover) | Bearer JWT | Correcto |
| `mcp_sources.py` | 6 (CRUD + discover + tools) | Bearer JWT | Correcto |
| `prompts.py` | 6 (CRUD + set active) | Bearer JWT | Correcto |
| `models.py` | 7 (CRUD + discovery) | Bearer JWT | Correcto |
| `admin.py` | 8 (users, settings, stats) | Bearer + is_superuser | Correcto |
| `mlops.py` | 3 (webhook, training launch x2) | API Key / JWT | Correcto |
| `db_collector.py` | 7 (CRUD + run + preview) | Bearer JWT | Correcto |
| `system.py` | 2 (health, stats) | None / Bearer | Correcto |

**Estado**: Cobertura completa. Todos los repositorios proactivos tienen endpoints correspondientes.

---

## 2. Dominio Reactivo — Endpoints (24 rutas post-fix)

### 2.1 `events.py` (8 rutas)

| Metodo | Ruta | Auth | Estado |
|--------|------|------|--------|
| POST | `/events/ingest` | X-API-Key | Correcto (sensores externos) |
| POST | `/events/manual` | Bearer JWT | Correcto |
| GET | `/events` | Bearer JWT + tenant filter | **FIXED** (antes sin filtro) |
| GET | `/events/stream` | Bearer o `?token=` + tenant filter SSE | **FIXED** (antes sin filtro) |
| GET | `/events/{id}` | Bearer JWT + ownership | **FIXED** (antes sin auth check) |
| POST | `/events/{id}/approve` | Bearer JWT + ownership | **FIXED** (antes sin auth check) |
| POST | `/events/{id}/reject` | Bearer JWT + ownership | **FIXED** (antes sin auth check) |
| GET | `/events/stats` | Bearer JWT + tenant filter | **AGREGADO** (nuevo) |

### 2.2 `reactive_knowledge.py` (8 rutas post-fix)

| Metodo | Ruta | Auth | Estado |
|--------|------|------|--------|
| POST | `/events/knowledge/` | Bearer JWT + tenant | **FIXED** (antes sin auth) |
| GET | `/events/knowledge/` | Bearer JWT + tenant filter | **FIXED** (antes sin auth) |
| GET | `/events/knowledge/{id}` | Bearer JWT + ownership | **FIXED** (antes sin auth) |
| PUT | `/events/knowledge/{id}` | Bearer JWT + ownership | **AGREGADO** (nuevo) |
| DELETE | `/events/knowledge/{id}` | Bearer JWT + ownership | **FIXED** (antes sin auth) |
| POST | `/events/knowledge/{id}/documents` | Bearer JWT + ownership | **FIXED** (antes sin auth) |
| GET | `/events/knowledge/{id}/documents` | Bearer JWT + ownership | **AGREGADO** (nuevo) |
| DELETE | `/events/knowledge/{id}/documents/{doc_id}` | Bearer JWT + ownership | **AGREGADO** (nuevo) |

### 2.3 `reactive_mcp_sources.py` (8 rutas post-fix)

| Metodo | Ruta | Auth | Estado |
|--------|------|------|--------|
| POST | `/events/mcp/` | Bearer JWT + tenant | **FIXED** (antes sin auth) |
| GET | `/events/mcp/` | Bearer JWT + tenant filter | **FIXED** (antes sin auth) |
| GET | `/events/mcp/{id}` | Bearer JWT + ownership | **AGREGADO** (nuevo) |
| PUT | `/events/mcp/{id}` | Bearer JWT + ownership | **AGREGADO** (nuevo) |
| DELETE | `/events/mcp/{id}` | Bearer JWT + ownership | **FIXED** (antes sin auth) |
| POST | `/events/mcp/{id}/sync` | Bearer JWT + ownership | **FIXED** (antes sin auth) |
| GET | `/events/mcp/{id}/tools` | Bearer JWT + ownership | **FIXED** (antes sin auth) |
| DELETE | `/events/mcp/{id}/tools/{tool_id}` | Bearer JWT + ownership | **AGREGADO** (nuevo) |

---

## 3. Bugs de Seguridad Corregidos

### S1 — Endpoints reactivos sin autenticacion (CRITICO)

**Archivos afectados** (pre-fix):
- `api/reactiva/endpoints/reactive_knowledge.py`: create, list, get, upload, delete — **sin JWT**
- `api/reactiva/endpoints/reactive_mcp_sources.py`: create, list, delete, sync, list_tools — **sin JWT**

**Fix aplicado**: Todos los endpoints reactivos ahora reciben `current_user: Annotated[User, Depends(get_current_user)]`. Se filtra por `tenant_id` del usuario; superusers ven todos los tenants.

### S2 — Listado global de eventos (CRITICO)

**Archivo**: `api/reactiva/endpoints/events.py`

**Pre-fix**: `GET /events` listaba **todos los eventos de todos los tenants**.
**Fix**: `EventRepository.list_all()` ahora acepta `tenant_id` opcional; el endpoint filtra por `current_user.tenant_id` (o `None` para superuser).

### S3 — Aprobacion/rechazo sin verificacion de ownership (CRITICO)

**Archivo**: `api/reactiva/endpoints/events.py`

**Pre-fix**: Cualquier usuario autenticado podia aprobar/rechazar eventos de otros tenants.
**Fix**: Se agrego verificacion `event.tenant_id != current_user.tenant_id` (403) en `approve_event` y `reject_event`.

### S4 — SSE stream sin filtrado de tenant (MEDIO)

**Archivo**: `api/reactiva/endpoints/events.py`

**Pre-fix**: El stream SSE enviaba todos los eventos a todos los clientes conectados.
**Fix**: El generator del SSE ahora descarta payloads cuyo `tenant_id` no coincide con el del usuario (a menos que sea superuser).

### S5 — `triggered_by_user_id` como `str` (MEDIO)

**Archivo**: `domain/reactiva/schemas/event.py`

**Pre-fix**: Campo tipo `str` sin foreign key, imposible JOIN eficiente.
**Fix**: Cambiado a `Optional[uuid.UUID]` con `foreign_key="user.id"`. Agregados `approved_by_user_id` y `rejected_by_user_id` para trazabilidad completa.

---

## 4. Endpoints Agregados (Faltantes)

| Modulo | Endpoint | Descripcion | Motivo |
|--------|----------|-------------|--------|
| `events.py` | `GET /events/stats` | Stats agregadas por severity/status | Dashboard necesita metricas |
| `reactive_knowledge.py` | `PUT /{kb_id}` | Update reactive knowledge base | Paridad con repo `update_kb()` |
| `reactive_knowledge.py` | `GET /{kb_id}/documents` | List documents for KB | Paridad con repo `list_documents()` |
| `reactive_knowledge.py` | `DELETE /{kb_id}/documents/{doc_id}` | Delete document | Paridad con repo `delete_document()` |
| `reactive_mcp_sources.py` | `GET /{source_id}` | Get MCP source by ID | Paridad con repo `get_by_id()` |
| `reactive_mcp_sources.py` | `PUT /{source_id}` | Update MCP source | Paridad con repo `update()` |
| `reactive_mcp_sources.py` | `DELETE /{source_id}/tools/{tool_id}` | Delete reactive tool config | Paridad con repo `delete()` |

---

## 5. Matriz de Cobertura: Repositorio vs Endpoint

### Reactivo — Cobertura completa post-fix

| Repositorio | Metodos | Endpoints que lo consumen | Estado |
|-------------|---------|--------------------------|--------|
| `EventRepository` | create, get_by_id, list_all, update_status, update_analysis, save | events.py (8 rutas) | Completo |
| `ReactiveKnowledgeRepository` | create_kb, get_kb_by_id, list_kbs, update_kb, delete_kb, add_document, list_documents, delete_document | reactive_knowledge.py (8 rutas) | Completo |
| `ReactiveMCPSourceRepository` | get_by_id, list_all, create, update, delete | reactive_mcp_sources.py (6 rutas) | Completo |
| `ReactiveToolConfigRepository` | get_by_name, get_by_source, get_all, create, update, delete | reactive_mcp_sources.py (2 rutas) + service interno | Completo |

### Proactivo — Cobertura completa

| Repositorio | Modulo endpoint | Estado |
|-------------|----------------|--------|
| `ConversationRepository` | conversations.py | Completo |
| `KnowledgeRepository` | knowledge.py, documents.py | Completo |
| `ToolConfigRepository` | tools.py | Completo |
| `MCPSourceRepository` | mcp_sources.py | Completo |
| `PromptRepository` | prompts.py | Completo |
| `ModelRepository` | models.py | Completo |
| `LLMConfigRepository` | admin.py, core/llm.py | Completo |
| `SettingsRepository` | admin.py | Completo |
| `UserRepository` | users.py, auth.py, admin.py | Completo |
| `DbSourceRepository` | db_collector.py | Completo |

---

## 6. Notas sobre Autenticacion Multi-Tenant

- Todos los endpoints reactivos ahora usan `get_current_user` (JWT Bearer) o `get_current_user_flexible` (SSE con query param).
- El `User` model tiene `tenant_id: str` con default `"default"`.
- Superusers (`is_superuser=True`) pueden ver todos los tenants.
- Usuarios regulares solo ven recursos de su `tenant_id`.
- Los endpoints de ingest con `X-API-Key` continuan sin autenticacion de usuario (correcto para sensores IoT).

---

## 7. Pendientes

1. **Migracion DB**: `User.tenant_id` y `Event.triggered_by_user_id` (str->UUID) requieren migracion.
2. **Rol `operator`**: Aun no hay RBAC granular mas alla de `is_superuser`. Considerar tabla `user_tenant_membership` a futuro.
3. **Reactive Document Processor**: El endpoint `upload_reactive_document` usa `ReactiveDocumentProcessor` que procesa con Qdrant; no hay endpoint para re-procesar o re-indexar documentos existentes.

---

*Auditoria y fixes aplicados el 2026-04-30. Stack: FastAPI + SQLModel + PostgreSQL + JWT + SSE.*
