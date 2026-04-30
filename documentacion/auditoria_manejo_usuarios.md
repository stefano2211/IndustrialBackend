# Auditoria: Manejo de Usuarios — Proactivo vs Reactivo

## Resumen Ejecutivo

El manejo de usuarios está **correctamente centralizado en la capa compartida** (`domain/shared/`, `persistence/shared/`, `api/deps.py`). Tanto el dominio proactivo como el reactivo utilizan el mismo modelo `User`, repositorio `UserRepository`, servicio `UserService` y dependencia de autenticación `get_current_user`. La autenticación JWT (HS256 + bcrypt) es uniforme en ambos dominios. Sin embargo, se detectan **tres problemas de seguridad y diseño** que afectan principalmente al dominio reactivo: falta de filtro por tenant en listado de eventos, ausencia de RBAC granular más allá de `is_superuser`, e inconsistencia de tipo en `triggered_by_user_id`.

---

## 1. Arquitectura de Usuarios (Capa Compartida)

### 1.1 Modelo de Datos

```
┌─────────────────────────────────────┐
│         domain/shared/schemas       │
│            user.py                  │
├─────────────────────────────────────┤
│  User (table=True)                  │
│  ├── id: UUID (PK)                  │
│  ├── username: str                  │
│  ├── email: str (unique)            │
│  ├── hashed_password: str           │
│  ├── is_active: bool                │
│  ├── is_superuser: bool             │
│  ├── created_at: datetime           │
│  └── updated_at: datetime           │
└─────────────────────────────────────┘
```

**Estado**: ✅ Correctamente ubicado en capa compartida. No hay duplicación.

### 1.2 Repositorio y Servicio

| Componente | Ubicación | Uso Proactivo | Uso Reactivo |
|------------|-----------|---------------|--------------|
| `UserRepository` | `persistence/shared/` | ✅ Login, perfil, admin | ✅ Auth SSE stream, manual events |
| `UserService` | `domain/shared/services/` | ✅ Registro, auth, CRUD | ✅ Decode JWT → User |
| `get_current_user` | `api/deps.py` | ✅ Bearer token | ✅ Bearer token + query param (SSE) |

**Estado**: ✅ Ambos dominios consumen las mismas instancias compartidas. Sin fugas cross-domain.

### 1.3 Seguridad (core/security.py)

| Aspecto | Implementación | Estado |
|---------|---------------|--------|
| Password hashing | `bcrypt` via `passlib.CryptContext` | ✅ Correcto |
| JWT algoritmo | `HS256` | ✅ Estándar |
| JWT expiración | `settings.access_token_expire_minutes` | ✅ Configurable |
| JWT payload | `{"exp": ..., "sub": email}` | ✅ Correcto |

---

## 2. Uso en Dominio Proactivo

| Endpoint | Auth | User Context | Observación |
|----------|------|-------------|-------------|
| `POST /auth/login` | No (credenciales) | Crea token JWT | ✅ Correcto |
| `POST /auth/signup` | No | Verifica `settings.auth_enable_sign_ups` | ✅ Correcto |
| `GET /users/me` | Bearer | Devuelve current user | ✅ Correcto |
| `PUT /users/me` | Bearer | Actualiza own user | ✅ Correcto |
| `GET /admin/users` | Bearer + `is_superuser` | Lista usuarios | ✅ RBAC básico |
| `PUT /admin/users/{id}/role` | Bearer + `is_superuser` | Cambia rol | ✅ RBAC básico |
| `DELETE /admin/users/{id}` | Bearer + `is_superuser` | Elimina usuario | ✅ RBAC básico |

**Estado**: ✅ El dominio proactivo tiene control de acceso basado en autenticación + flag `is_superuser`.

---

## 3. Uso en Dominio Reactivo

| Endpoint | Auth | User Context | Observación |
|----------|------|-------------|-------------|
| `POST /events/ingest` | `X-API-Key` | Ninguno (sensor externo) | ✅ Correcto para IoT |
| `POST /events/manual` | Bearer JWT | `triggered_by_user_id` | ✅ Traza de operador |
| `GET /events` | Bearer JWT | current_user disponible | ⚠️ **NO filtra por tenant/user** |
| `GET /events/stream` | Bearer o `?token=` | current_user disponible | ⚠️ **NO filtra por tenant/user** |
| `GET /events/{id}` | Bearer JWT | current_user disponible | ⚠️ **NO verifica ownership** |
| `POST /events/{id}/approve` | Bearer JWT | current_user disponible | ⚠️ **NO verifica permisos** |
| `POST /events/{id}/reject` | Bearer JWT | current_user disponible | ⚠️ **NO verifica permisos** |

### 3.1 Problema Crítico: Listado Global de Eventos

```python
# app/api/reactiva/endpoints/events.py:139-161
@router.get("")
async def list_events(
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_user)],  # ← Se recibe pero NO se usa
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    ...
):
    repo = EventRepository(session)
    items, total = await repo.list_all(
        severity=severity,
        status=status,
        source_type=source_type,
        limit=limit,
        offset=offset,
    )  # ← Sin filtro por tenant ni por user
```

**Impacto**: Cualquier usuario autenticado puede ver **todos los eventos de todos los tenants**. En un entorno multi-tenant industrial, un operador de la planta A puede ver alarmas críticas de la planta B.

### 3.2 Problema: Inconsistencia de Tipo

```python
# app/domain/reactiva/schemas/event.py:61-64
triggered_by_user_id: Optional[str] = Field(
    default=None,
    description="User UUID who created a manual event",
)
```

**Impacto**: El `User.id` es `uuid.UUID`, pero `triggered_by_user_id` es `str`. Imposible hacer JOIN nativo sin cast. Además, `EventRepository` no expone método para listar eventos por operador.

---

## 4. Hallazgos de Deuda Técnica

| ID | Hallazgo | Severidad | Categoría |
|----|----------|-----------|-----------|
| **U1** | `list_events` no filtra por `tenant_id` del usuario | **Alta** | Seguridad / Multi-tenant |
| **U2** | `get_event`/`approve`/`reject` no verifican ownership ni permisos | **Alta** | Seguridad / Autorización |
| **U3** | `triggered_by_user_id` es `str` en vez de `uuid.UUID` | **Media** | Consistencia de tipos |
| **U4** | No existe RBAC granular (solo `is_superuser`) | **Media** | Seguridad / Roles |
| **U5** | `User` model no tiene `tenant_id` ni relación multi-tenant | **Baja** | Diseño / Escalabilidad |

### U1 — Listado Global de Eventos (Alta)

**Recomendación inmediata**: Agregar filtro `tenant_id` a `list_events`. Si el `User` no tiene campo `tenant_id`, se debe usar el `tenant_id` del request payload o inferirlo de alguna forma (ej. header `X-Tenant-ID`).

```python
# Opción A: Si User tiene tenant_id
items, total = await repo.list_all(
    tenant_id=current_user.tenant_id,  # ← Filtrar por tenant del usuario
    severity=severity,
    ...
)

# Opción B: Si se mantiene User global, usar header/query param tenant
# (requiere que el frontend envíe el tenant activo)
```

### U2 — Falta de Autorización en Operaciones Sensibles (Alta)

**Recomendación inmediata**: Agregar verificaciones antes de `approve`/`reject`:

1. Verificar que el evento pertenezca al tenant del usuario (o esté en whitelist)
2. Opcional: verificar que el usuario tenga rol `operator` o `admin`
3. Registrar `approved_by_user_id` / `rejected_by_user_id` en el evento

### U3 — `triggered_by_user_id` como `str` (Media)

**Recomendación**:

```python
# Cambiar en event.py
triggered_by_user_id: Optional[uuid.UUID] = Field(
    default=None,
    foreign_key="user.id",  # ← Referencia explícita
    description="User UUID who created a manual event",
)
```

Esto permite:
- JOIN eficiente entre `event` y `user`
- Integridad referencial (si se desea)
- Type safety en todo el pipeline

### U4 / U5 — RBAC Granular y Multi-tenancy (Media/Baja)

**Recomendación a largo plazo**:

Considerar extender el modelo `User` o crear una tabla de membresía:

```python
class UserTenantMembership(SQLModel, table=True):
    __tablename__ = "user_tenant_membership"
    
    user_id: uuid.UUID = Field(foreign_key="user.id", primary_key=True)
    tenant_id: str = Field(primary_key=True)
    role: str = Field(default="operator")  # operator | admin | viewer
    is_default: bool = Field(default=False)
```

Esto permite:
- Un usuario pertenecer a múltiples tenants
- Roles diferenciados por tenant
- Tenant default para el login

---

## 5. Comparativa: Proactivo vs Reactivo

| Aspecto | Proactivo | Reactivo | Estado |
|---------|-----------|----------|--------|
| **Autenticación** | Bearer JWT | Bearer JWT + SSE token query | ✅ Correcto |
| **User model** | Compartido `User` | Compartido `User` | ✅ Correcto |
| **Registro de actor** | `user_id` en Conversation | `triggered_by_user_id` en Event | ⚠️ Tipo inconsistente |
| **Filtrado por usuario** | `list_by_user` en Conversation | **NO existe** en Event | ❌ Faltante |
| **RBAC** | `is_superuser` (admin CRUD) | **NO existe** | ❌ Faltante |
| **Multi-tenant** | N/A (user-scoped) | `tenant_id` en Event | ⚠️ User no tiene tenant |

---

## 6. Recomendaciones Priorizadas

### Inmediato (esta semana)

1. **U1**: Agregar filtro `tenant_id` a `EventRepository.list_all()` y usarlo en `list_events` endpoint
2. **U2**: Agregar verificación de tenant/permisos en `approve`, `reject`, `get_event`

### Corto plazo (1-2 semanas)

3. **U3**: Cambiar `triggered_by_user_id` de `str` a `uuid.UUID` con `foreign_key`
4. **U4**: Considerar agregar `approved_by_user_id` / `rejected_by_user_id` a `Event` para trazabilidad completa

### Mediano plazo (1 mes)

5. **U5**: Evaluar tabla `UserTenantMembership` para soportar multi-tenancy real con roles por tenant

---

## 7. Cambios Aplicados (2026-04-30)

| Hallazgo | Estado | Archivos Modificados | Detalle del Cambio |
|----------|--------|---------------------|-------------------|
| **U1** `list_events` no filtra por tenant | **RESUELTO** | `domain/shared/schemas/user.py`, `persistence/reactiva/repositories/event_repository.py`, `api/reactiva/endpoints/events.py` | Agregado `tenant_id` a `User` model. `list_all()` filtra por `tenant_id`. Endpoints `list_events` y `events_stream` (SSE) filtran por tenant del current_user. Superusers ven todos los tenants. |
| **U2** `approve`/`reject`/`get_event` sin autorizacion | **RESUELTO** | `api/reactiva/endpoints/events.py` | Agregadas verificaciones `event.tenant_id != current_user.tenant_id` (403) en `get_event`, `approve_event`, `reject_event`. Superusers exentos. |
| **U3** `triggered_by_user_id` como `str` | **RESUELTO** | `domain/reactiva/schemas/event.py`, `domain/reactiva/events/schemas.py`, `domain/shared/events/publisher.py`, `api/reactiva/endpoints/events.py` | Cambiado a `Optional[uuid.UUID]` con `foreign_key="user.id"`. Actualizado `EventPublisher.publish()` y endpoints. |
| **U4** Sin audit trail de aprobacion/rechazo | **RESUELTO** | `domain/reactiva/schemas/event.py`, `domain/reactiva/events/schemas.py`, `api/reactiva/endpoints/events.py` | Agregados `approved_by_user_id` y `rejected_by_user_id` al modelo `Event` y a `EventResponse`. Se popula en `approve_event` y `reject_event`. |
| **U5** `User` sin `tenant_id` | **RESUELTO** | `domain/shared/schemas/user.py` | Agregado `tenant_id: str = Field(default="default", index=True)` a `User`. `UserRead` expone el campo. `UserUpdate` permite modificarlo (admin). |

**Nota sobre migracion de base de datos**: Los cambios a `User.tenant_id` y `Event.triggered_by_user_id` (str → UUID) requieren una migracion de Alembic o script SQL para actualizar las tablas existentes. El campo `tenant_id` en usuarios existentes se poblará con `"default"`. Los valores existentes de `triggered_by_user_id` (strings con UUIDs) deben castearse a tipo UUID en PostgreSQL.

---

*Documento generado el 2026-04-30. Stack: FastAPI + SQLModel + PostgreSQL + JWT (HS256) + bcrypt.*
