"""
Reactive MCP Sources API endpoints.

System-scoped endpoints for managing MCP tools available to the Reactive Agent.
Tools configured here will be automatically registered when the reactive
orchestrator executes.
"""

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession

from app.persistence.db import get_session
from app.domain.schemas.reactive_mcp_source import (
    ReactiveMCPSourceRead,
    ReactiveMCPSourceCreate,
    ReactiveMCPSource,
)
from app.domain.schemas.reactive_tool_config import ReactiveToolConfigRead
from app.persistence.reactiva.repositories.reactive_mcp_source_repository import ReactiveMCPSourceRepository
from app.persistence.reactiva.repositories.reactive_tool_config_repository import ReactiveToolConfigRepository
from app.domain.shared.services.mcp_service import MCPService


router = APIRouter()


@router.post("/", response_model=ReactiveMCPSourceRead)
async def create_reactive_source(
    source_in: ReactiveMCPSourceCreate,
    tenant_id: str = "default",
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveMCPSourceRepository(session)
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
    tenant_id: str = "default",
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveMCPSourceRepository(session)
    return await repo.list_all(tenant_id)


@router.delete("/{source_id}")
async def delete_reactive_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveMCPSourceRepository(session)
    success = await repo.delete(source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "ok"}


@router.post("/{source_id}/sync")
async def sync_reactive_source(
    source_id: uuid.UUID,
    session: AsyncSession = Depends(get_session)
):
    """
    Connect to the MCP source, discover available tools, and auto-register them
    into the reactive_tool_config table.
    """
    source_repo = ReactiveMCPSourceRepository(session)
    source = await source_repo.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

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
    from app.domain.schemas.reactive_tool_config import ReactiveToolConfig
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
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveToolConfigRepository(session)
    return await repo.get_by_source(source_id)
