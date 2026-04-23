from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.api import deps
from app.domain.schemas.user import User
from app.persistence.db import get_session
from app.domain.schemas.mcp_source import MCPSourceCreate, MCPSourceRead, MCPSourceUpdate
from app.domain.proactiva.services.mcp_source_service import MCPSourceService
from app.domain.proactiva.services.mcp_service import MCPService
from app.domain.exceptions import NotFoundError

router = APIRouter(dependencies=[Depends(deps.get_current_user)])

async def get_source_service(session: AsyncSession = Depends(get_session)) -> MCPSourceService:
    return MCPSourceService(session)

@router.post("/", response_model=MCPSourceRead)
async def create_source(
    source_in: MCPSourceCreate,
    current_user: User = Depends(deps.get_current_user),
    service: MCPSourceService = Depends(get_source_service),
):
    return await service.create(user_id=current_user.id, source_in=source_in)

@router.get("/", response_model=List[MCPSourceRead])
async def list_sources(
    current_user: User = Depends(deps.get_current_user),
    service: MCPSourceService = Depends(get_source_service),
):
    return await service.get_all(user_id=current_user.id)

@router.get("/{source_id}", response_model=MCPSourceRead)
async def get_source(
    source_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: MCPSourceService = Depends(get_source_service),
):
    try:
        return await service.get_by_id(source_id=source_id, user_id=current_user.id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{source_id}", response_model=MCPSourceRead)
async def update_source(
    source_id: uuid.UUID,
    source_in: MCPSourceUpdate,
    current_user: User = Depends(deps.get_current_user),
    service: MCPSourceService = Depends(get_source_service),
):
    try:
        return await service.update(source_id=source_id, user_id=current_user.id, source_in=source_in)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{source_id}")
async def delete_source(
    source_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: MCPSourceService = Depends(get_source_service),
):
    try:
        await service.delete(source_id=source_id, user_id=current_user.id)
        return {"status": "success"}
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{source_id}/discover")
async def discover_source_tools(
    source_id: uuid.UUID,
    method: str = "GET",
    current_user: User = Depends(deps.get_current_user),
    service: MCPSourceService = Depends(get_source_service),
):
    source = await service.get_by_id(source_id=source_id, user_id=current_user.id)
    mcp_service = MCPService()
    try:
        return await mcp_service.discover_tools(source.url, is_stdio=(source.type == "stdio"), method=method)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
