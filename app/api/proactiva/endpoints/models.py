"""Model management endpoints — handlers for custom Model configurations."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.persistence.db import get_session
from app.api import deps
from app.domain.shared.schemas.user import User
from app.persistence.proactiva.repositories.model_repository import ModelRepository
from app.persistence.proactiva.repositories.settings_repository import SettingsRepository
from app.domain.proactiva.schemas.model import Model, ModelCreate, ModelRead, ModelUpdate
from app.core.config import settings

router = APIRouter(dependencies=[Depends(deps.get_current_user)])

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
    _: User = Depends(deps.get_current_user),
    repo: ModelRepository = Depends(get_model_repo),
):
    """Create a new model configuration."""
    if not model_in.id:
        # Simple slug generation from name, fallback to UUID
        import re
        import uuid
        base_slug = re.sub(r'[^a-z0-9]', '-', model_in.name.lower()).strip('-')
        model_in.id = f"{base_slug}-{str(uuid.uuid4())[:8]}"
    
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
    _: User = Depends(deps.get_current_user),
    repo: ModelRepository = Depends(get_model_repo),
):
    """Delete a model configuration."""
    success = await repo.delete(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "success", "message": f"Model {model_id} deleted"}


# -- Discovery Endpoints --------------------------------


@router.get("/discovery/providers")
async def list_providers(
    session: AsyncSession = Depends(get_session),
):
    """Return a list of enabled providers based on system settings."""
    repo = SettingsRepository(session)
    sys_settings = await repo.get_settings()
    
    providers = []
    # Dejar solo vLLM, usando la URL del orquestador
    providers.append({
        "id": "vllm", 
        "name": "vLLM Orchestrator", 
        "base_url": getattr(settings, "vllm_orchestrator_url", settings.vllm_base_url)
    })
    
    return providers


@router.get("/discovery/models/{provider_id}")
async def list_provider_models(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Fetch available models for a specific provider."""
    if provider_id == "vllm":
        repo = SettingsRepository(session)
        sys_settings = await repo.get_settings()

        urls_to_query = []

        # Como corremos 2 vllm con diferentes modelos, solo mostramos el modelo orquestador
        orchestrator_url = getattr(settings, 'vllm_orchestrator_url', None)
        if orchestrator_url:
            urls_to_query.append(("Orchestrator", orchestrator_url))
        else:
            urls_to_query.append(("vLLM", settings.vllm_base_url))

        all_models = []
        seen_ids = set()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                for label, url in urls_to_query:
                    try:
                        # settings.vllm_base_url already includes /v1, so just append /models
                        response = await client.get(f"{url}/models")
                        if response.status_code == 200:
                            data = response.json()
                            for model in data.get("data", []):
                                model_id = model["id"]
                                # Solo mostrar modelos base (formato HuggingFace: "Org/Model")
                                # Los LoRA adapters son internos del sistema y no se exponen al usuario
                                if "/" not in model_id:
                                    continue
                                if model_id not in seen_ids:
                                    seen_ids.add(model["id"])
                                    all_models.append({
                                        "id": model["id"],
                                        "name": model["id"],
                                        "size": 0,
                                        "details": {"source": label}
                                    })
                    except Exception:
                        pass  # Un servidor caído no rompe el discovery del otro

            if all_models:
                return all_models
        except Exception:
            pass

        # Fallback: retornar el modelo default como Offline
        return [{"id": settings.default_llm_model, "name": f"{settings.default_llm_model} (Offline)", "error": "Could not connect to vLLM servers"}]

    return []

    return []
