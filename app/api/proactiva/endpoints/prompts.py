"""Prompt endpoints — thin HTTP handlers for Prompt CRUD."""

import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.db import get_session
from app.persistence.proactiva.repositories.prompt_repository import PromptRepository
from app.domain.proactiva.schemas.prompt import Prompt, PromptCreate, PromptRead, PromptUpdate

router = APIRouter()


async def get_prompt_repo(
    session: AsyncSession = Depends(get_session),
) -> PromptRepository:
    return PromptRepository(session)


@router.post("/", response_model=PromptRead)
async def create_prompt(
    prompt_in: PromptCreate,
    repo: PromptRepository = Depends(get_prompt_repo),
):
    prompt = Prompt.model_validate(prompt_in)
    return await repo.create_prompt(prompt)


@router.get("/", response_model=List[PromptRead])
async def list_prompts(
    only_enabled: bool = Query(False),
    repo: PromptRepository = Depends(get_prompt_repo),
):
    return await repo.list_prompts(only_enabled=only_enabled)


@router.get("/{prompt_id}", response_model=PromptRead)
async def get_prompt(
    prompt_id: uuid.UUID,
    repo: PromptRepository = Depends(get_prompt_repo),
):
    prompt = await repo.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.patch("/{prompt_id}", response_model=PromptRead)
async def update_prompt(
    prompt_id: uuid.UUID,
    prompt_update: PromptUpdate,
    repo: PromptRepository = Depends(get_prompt_repo),
):
    prompt = await repo.update_prompt(prompt_id, prompt_update)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.patch("/{prompt_id}/active")
async def set_active_prompt(
    prompt_id: uuid.UUID,
    repo: PromptRepository = Depends(get_prompt_repo),
):
    """Mark a specific prompt as active and others as inactive."""
    prompt = await repo.set_active(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.delete("/{prompt_id}")
async def delete_prompt(
    prompt_id: uuid.UUID,
    repo: PromptRepository = Depends(get_prompt_repo),
):
    success = await repo.delete_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"detail": "Prompt deleted"}
