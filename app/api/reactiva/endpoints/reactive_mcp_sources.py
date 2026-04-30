"""
Reactive MCP Sources API endpoints.

System-scoped endpoints for managing MCP tools available to the Reactive Agent.
Tools configured here will be automatically registered when the reactive
orchestrator executes.
"""

import uuid
from typing import List

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.deps import get_current_user
from app.persistence.db import get_session
from app.domain.shared.schemas.user import User
from app.domain.reactiva.schemas.reactive_mcp_source import (
    ReactiveMCPSourceRead,
    ReactiveMCPSourceCreate,
    ReactiveMCPSource,
)
from app.domain.reactiva.schemas.reactive_tool_config import ReactiveToolConfigRead
from app.persistence.reactiva.repositories.reactive_mcp_source_repository import ReactiveMCPSourceRepository
from app.persistence.reactiva.repositories.reactive_tool_config_repository import ReactiveToolConfigRepository
from app.domain.shared.services.mcp_service import MCPService


router = APIRouter()


@router.post("/", response_model=ReactiveMCPSourceRead)
async def create_reactive_source(
    source_in: ReactiveMCPSourceCreate,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveMCPSourceRepository(session)
    tenant_id = current_user.tenant_id if current_user else "default"
    # Check if a source with this URL already exists for this tenant
    existing = await repo.list_all(tenant_id)
    for src in existing:
        if src.url == source_in.url:
            raise HTTPException(status_code=400, detail="Reactive MCP source URL already exists")

    source = ReactiveMCPSource(
        **source_in.model_dump(),
        tenant_id=tenant_id
    )
    return await repo.create(source)


@router.get("/", response_model=List[ReactiveMCPSourceRead])
async def list_reactive_sources(
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveMCPSourceRepository(session)
    tenant_filter = None if (current_user and current_user.is_superuser) else (current_user.tenant_id if current_user else "default")
    if tenant_filter is not None:
        return await repo.list_all(tenant_filter)
    return await repo.list_all()


@router.get("/{source_id}", response_model=ReactiveMCPSourceRead)
async def get_reactive_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveMCPSourceRepository(session)
    source = await repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if current_user and not current_user.is_superuser and source.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this source")
    return source


@router.put("/{source_id}", response_model=ReactiveMCPSourceRead)
async def update_reactive_source(
    source_id: uuid.UUID,
    source_in: ReactiveMCPSourceCreate,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveMCPSourceRepository(session)
    source = await repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if current_user and not current_user.is_superuser and source.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this source")

    update_data = source_in.model_dump()
    updated = await repo.update(source_id, update_data)
    if not updated:
        raise HTTPException(status_code=500, detail="Update failed")
    return updated


@router.delete("/{source_id}")
async def delete_reactive_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveMCPSourceRepository(session)
    source = await repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if current_user and not current_user.is_superuser and source.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this source")

    success = await repo.delete(source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "ok"}


@router.post("/{source_id}/sync")
async def sync_reactive_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    """
    Connect to the MCP source, discover available tools, and auto-register them
    into the reactive_tool_config table.
    """
    source_repo = ReactiveMCPSourceRepository(session)
    source = await source_repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if current_user and not current_user.is_superuser and source.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to sync this source")

    mcp_service = MCPService()
    try:
        discovered_tools = await mcp_service.discover_tools(
            base_url=source.url,
            is_stdio=(source.type == "stdio")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {e}")

    tool_repo = ReactiveToolConfigRepository(session)
    existing_tools = await tool_repo.get_by_source(source.id)
    existing_names = {t.name for t in existing_tools}

    added = 0
    from app.domain.reactiva.schemas.reactive_tool_config import ReactiveToolConfig
    for tool_def in discovered_tools:
        if tool_def.name not in existing_names:
            new_tool = ReactiveToolConfig(
                name=tool_def.name,
                description=tool_def.description or "",
                api_url=source.url,
                system_prompt="",
                parameter_schema={"schema": getattr(tool_def, 'inputSchema', {})},
                config={"transport": source.type},
                source_id=source.id,
            )
            await tool_repo.create(new_tool)
            added += 1

    return {"status": "ok", "tools_discovered": len(discovered_tools), "tools_added": added}


@router.get("/{source_id}/tools", response_model=List[ReactiveToolConfigRead])
async def list_reactive_tools_for_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    source_repo = ReactiveMCPSourceRepository(session)
    source = await source_repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if current_user and not current_user.is_superuser and source.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this source")

    tool_repo = ReactiveToolConfigRepository(session)
    return await tool_repo.get_by_source(source_id)


@router.delete("/{source_id}/tools/{tool_id}")
async def delete_reactive_tool(
    source_id: uuid.UUID,
    tool_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    source_repo = ReactiveMCPSourceRepository(session)
    source = await source_repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if current_user and not current_user.is_superuser and source.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify tools for this source")

    tool_repo = ReactiveToolConfigRepository(session)
    success = await tool_repo.delete(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "ok"}
