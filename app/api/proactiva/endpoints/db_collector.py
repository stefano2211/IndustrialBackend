"""
DbCollector API Endpoints.

Provides CRUD for DbSource (database collection sources) and
a manual trigger endpoint for immediate on-demand runs.
All write operations require superuser access.
"""

import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.persistence.db import get_session
from app.domain.proactiva.db_collector.collector_service import collector_service
from app.domain.proactiva.db_collector.encryption import encrypt
from app.domain.proactiva.db_collector.scheduler import collector_scheduler
from app.domain.shared.schemas.db_source import (
    DbSource,
    DbSourceCreate,
    DbSourceRead,
    DbSourceUpdate,
)
from app.domain.shared.schemas.user import User
from app.persistence.proactiva.repositories.db_source_repository import DbSourceRepository

router = APIRouter()


def _require_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser access required.")
    return current_user


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.get("/sources", response_model=List[DbSourceRead])
async def list_sources(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
):
    """List all registered database collection sources."""
    repo = DbSourceRepository(session)
    return await repo.get_all()


@router.post("/sources", response_model=DbSourceRead, status_code=201)
async def create_source(
    payload: DbSourceCreate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(_require_superuser),
):
    """
    Register a new database source.
    The connection_string is encrypted with Fernet before storage
    and is never returned in API responses.
    """
    encrypted = encrypt(payload.connection_string)

    source = DbSource(
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        db_type=payload.db_type,
        connection_string_enc=encrypted,
        query=payload.query,
        cron_expression=payload.cron_expression,
        is_enabled=payload.is_enabled,
        tenant_id=payload.tenant_id,
        sector=payload.sector,
        domain=payload.domain,
    )

    repo = DbSourceRepository(session)
    created = await repo.create(source)

    # Reload scheduler so the new job is registered immediately
    await collector_scheduler.reload()

    logger.info(f"[DbCollector] Source created: {created.name} ({created.db_type})")
    return created


@router.put("/sources/{source_id}", response_model=DbSourceRead)
async def update_source(
    source_id: uuid.UUID,
    payload: DbSourceUpdate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(_require_superuser),
):
    """Update an existing source. Reloads the scheduler automatically."""
    repo = DbSourceRepository(session)
    source = await repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="DbSource not found.")

    update_data = payload.model_dump(exclude_unset=True)

    # Re-encrypt if connection_string is being updated
    if "connection_string" in update_data:
        update_data["connection_string_enc"] = encrypt(update_data.pop("connection_string"))

    for field, value in update_data.items():
        setattr(source, field, value)

    updated = await repo.save(source)
    await collector_scheduler.reload()

    logger.info(f"[DbCollector] Source updated: {updated.name}")
    return updated


@router.delete("/sources/{source_id}", status_code=204)
async def delete_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(_require_superuser),
):
    """Delete a source and remove its scheduled job."""
    repo = DbSourceRepository(session)
    source = await repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="DbSource not found.")

    await repo.delete(source)
    await collector_scheduler.reload()

    logger.info(f"[DbCollector] Source deleted: {source.name}")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

@router.post("/sources/{source_id}/run")
async def run_source_now(
    source_id: uuid.UUID,
    bg_tasks: BackgroundTasks,
    _: User = Depends(_require_superuser),
):
    """
    Trigger an immediate on-demand collection run for a specific source.
    Runs in the background  poll /sources/{id}/status to check the result.
    """
    bg_tasks.add_task(collector_service.run_source_by_id, source_id)
    return {
        "status": "accepted",
        "message": f"Manual collection triggered for source {source_id}. Check /status for result.",
    }


@router.get("/sources/{source_id}/status", response_model=DbSourceRead)
async def get_source_status(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
):
    """Return the latest run metadata (last_run_at, status, rows, error) for a source."""
    repo = DbSourceRepository(session)
    source = await repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="DbSource not found.")
    return source


@router.post("/run-all")
async def run_all_now(
    bg_tasks: BackgroundTasks,
    _: User = Depends(_require_superuser),
):
    """Trigger an immediate collection run for ALL enabled sources."""
    bg_tasks.add_task(collector_service.run_all_enabled)
    return {
        "status": "accepted",
        "message": "Manual collection triggered for all enabled sources.",
    }
