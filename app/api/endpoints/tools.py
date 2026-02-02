from fastapi import APIRouter, Depends, HTTPException
from app.domain.schemas.tool_config import ToolConfigCreate, ToolConfigRead, ToolConfigUpdate
from app.domain.services.tool_config_service import ToolConfigService
from app.persistence.db import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

router = APIRouter()

@router.post("/", response_model=ToolConfigRead)
async def create_tool(
    tool_in: ToolConfigCreate, 
    session: AsyncSession = Depends(get_session)
):
    service = ToolConfigService(session)
    existing = await service.get_by_name(tool_in.name)
    if existing:
        raise HTTPException(status_code=400, detail="Tool with this name already exists")
    return await service.create(tool_in)

@router.get("/", response_model=List[ToolConfigRead])
async def list_tools(session: AsyncSession = Depends(get_session)):
    service = ToolConfigService(session)
    return await service.get_all()

@router.get("/{tool_id}", response_model=ToolConfigRead)
async def get_tool(tool_id: int, session: AsyncSession = Depends(get_session)):
    service = ToolConfigService(session)
    tool = await service.get_by_id(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool

@router.put("/{tool_id}", response_model=ToolConfigRead)
async def update_tool(
    tool_id: int, 
    tool_in: ToolConfigUpdate, 
    session: AsyncSession = Depends(get_session)
):
    service = ToolConfigService(session)
    tool = await service.update(tool_id, tool_in)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool

@router.delete("/{tool_id}")
async def delete_tool(tool_id: int, session: AsyncSession = Depends(get_session)):
    service = ToolConfigService(session)
    success = await service.delete(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"status": "success"}
