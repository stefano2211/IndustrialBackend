"""Model management endpoints — handlers for custom Model configurations."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.db import get_session
from app.persistence.repositories.model_repository import ModelRepository
from app.domain.schemas.model import Model, ModelCreate, ModelRead, ModelUpdate

router = APIRouter()

async def get_model_repo(
    session: AsyncSession = Depends(get_session),
) -> ModelRepository:
    return ModelRepository(session)

@router.get("/", response_model=List[ModelRead])
async def list_models(
    repo: ModelRepository = Depends(get_model_repo),
):
    """List all configured models."""
    return await repo.list_all()

@router.get("/{model_id}", response_model=ModelRead)
async def get_model(
    model_id: str,
    repo: ModelRepository = Depends(get_model_repo),
):
    """Get details of a specific model by ID (slug)."""
    model = await repo.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.post("/", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_in: ModelCreate,
    repo: ModelRepository = Depends(get_model_repo),
):
    """Create a new model configuration."""
    existing = await repo.get_by_id(model_in.id)
    if existing:
        raise HTTPException(status_code=400, detail="Model with this ID already exists")
    
    new_model = Model.model_validate(model_in.model_dump())
    return await repo.create(new_model)

@router.put("/{model_id}", response_model=ModelRead)
async def update_model(
    model_id: str,
    model_in: ModelUpdate,
    repo: ModelRepository = Depends(get_model_repo),
):
    """Update an existing model configuration."""
    model = await repo.update(model_id, model_in)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    repo: ModelRepository = Depends(get_model_repo),
):
    """Delete a model configuration."""
    success = await repo.delete(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "success", "message": f"Model {model_id} deleted"}
