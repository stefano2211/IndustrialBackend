"""Tools endpoints — thin HTTP handlers for ToolConfig CRUD."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.db import get_session
from app.domain.proactiva.schemas.tool_config import ToolConfigCreate, ToolConfigRead, ToolConfigUpdate
from app.domain.proactiva.services.tool_config_service import ToolConfigService
from app.domain.shared.services.mcp_service import MCPService
from app.domain.exceptions import NotFoundError

router = APIRouter()


async def get_tool_service(
    session: AsyncSession = Depends(get_session),
) -> ToolConfigService:
    return ToolConfigService(session)


@router.post("/", response_model=ToolConfigRead)
async def create_tool(
    tool_in: ToolConfigCreate,
    service: ToolConfigService = Depends(get_tool_service),
):
    existing = await service.get_by_name(tool_in.name)
    if existing:
        raise HTTPException(status_code=400, detail="Tool with this name already exists")
    return await service.create(tool_in)


@router.get("/", response_model=List[ToolConfigRead])
async def list_tools(
    service: ToolConfigService = Depends(get_tool_service),
):
    return await service.get_all()


@router.get("/{tool_id}", response_model=ToolConfigRead)
async def get_tool(
    tool_id: int,
    service: ToolConfigService = Depends(get_tool_service),
):
    try:
        return await service.get_by_id(tool_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{tool_id}", response_model=ToolConfigRead)
async def update_tool(
    tool_id: int,
    tool_in: ToolConfigUpdate,
    service: ToolConfigService = Depends(get_tool_service),
):
    try:
        return await service.update(tool_id, tool_in)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{tool_id}")
async def delete_tool(
    tool_id: int,
    service: ToolConfigService = Depends(get_tool_service),
):
    success = await service.delete(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "success"}


@router.get("/mcp/discover")
async def discover_mcp_tools(
    url: str,
    is_stdio: bool = False,
    is_resource: bool = False,
    method: str = "GET",
):
    """Dynamically discover tools from an MCP server or REST API endpoint."""
    service = MCPService()
    try:
        return await service.discover_tools(url, is_stdio=is_stdio, is_resource=is_resource, method=method)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
