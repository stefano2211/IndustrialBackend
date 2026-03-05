from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.persistence.db import get_session
from app.persistence.repositories.llm_config_repository import LLMConfigRepository
from app.domain.schemas.llm_config import LLMConfigRead, LLMConfigUpdate, LLMConfig

router = APIRouter()

@router.get("/", response_model=List[LLMConfigRead])
async def list_configs(session: AsyncSession = Depends(get_session)):
    repo = LLMConfigRepository(session)
    return await repo.list_configs()

@router.get("/{role}", response_model=LLMConfigRead)
async def get_config(role: str, session: AsyncSession = Depends(get_session)):
    repo = LLMConfigRepository(session)
    config = await repo.get_config(role)
    if not config:
        raise HTTPException(status_code=404, detail=f"No config found for role: {role}")
    return config

@router.put("/{role}", response_model=LLMConfigRead)
async def set_config(
    role: str, 
    config_in: LLMConfigUpdate, 
    session: AsyncSession = Depends(get_session)
):
    if not config_in.provider or not config_in.model_name:
        raise HTTPException(status_code=400, detail="Provider and model_name are required for initial config")
        
    repo = LLMConfigRepository(session)
    return await repo.set_config(role, config_in.provider, config_in.model_name)

@router.patch("/{role}", response_model=LLMConfigRead)
async def update_config(
    role: str, 
    config_in: LLMConfigUpdate, 
    session: AsyncSession = Depends(get_session)
):
    repo = LLMConfigRepository(session)
    config = await repo.update_config(role, config_in)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config
