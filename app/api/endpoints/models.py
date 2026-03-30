"""Model management endpoints — handlers for custom Model configurations."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.persistence.db import get_session
from app.api import deps
from app.domain.schemas.user import User
from app.persistence.repositories.model_repository import ModelRepository
from app.persistence.repositories.settings_repository import SettingsRepository
from app.domain.schemas.model import Model, ModelCreate, ModelRead, ModelUpdate
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


# ── Discovery Endpoints ────────────────────────────────


@router.get("/discovery/providers")
async def list_providers(
    session: AsyncSession = Depends(get_session),
):
    """Return a list of enabled providers based on system settings."""
    repo = SettingsRepository(session)
    sys_settings = await repo.get_settings()
    
    providers = []
    if sys_settings.ollama_enabled:
        providers.append({"id": "ollama", "name": "Ollama", "base_url": sys_settings.ollama_base_url})
    if sys_settings.openrouter_enabled or settings.openrouter_api_key:
        providers.append({
            "id": "openrouter", 
            "name": "OpenRouter", 
            "base_url": sys_settings.openrouter_base_url or settings.openrouter_base_url
        })
    
    # Always include OpenAI if it's the internal default or manually enabled (mocked for now)
    providers.append({"id": "openai", "name": "OpenAI (GPT-4o)", "base_url": "https://api.openai.com/v1"})
    
    return providers


@router.get("/discovery/models/{provider_id}")
async def list_provider_models(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Fetch available models for a specific provider."""
    if provider_id == "ollama":
        repo = SettingsRepository(session)
        sys_settings = await repo.get_settings()
        base_url = sys_settings.ollama_base_url
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        # Filter out embedding models
                        details = model.get("details", {})
                        families = details.get("families", [])
                        is_embedding = (
                            "embed" in model["name"].lower() or 
                            "bert" in model["name"].lower() or
                            (families and any("bert" in f.lower() for f in families))
                        )
                        
                        if not is_embedding:
                            models.append({
                                "id": model["name"],
                                "name": model["name"],
                                "size": model.get("size"),
                                "details": details
                            })
                    return models
        except Exception as e:
            return [{"id": settings.default_llm_model, "name": f"{settings.default_llm_model} (Offline)", "error": str(e)}]

    if provider_id == "openrouter":
        # Dynamic fetch would be better, but sticking to requested models
        return [
            {"id": "qwen/qwen3-vl-30b-a3b-thinking", "name": "Qwen3 VL 30B A3B Thinking"},
            {"id": "qwen/qwen3-vl-30b-a3b-instruct", "name": "Qwen3 VL 30B A3B Instruct"},

            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"},
            {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            {"id": "meta-llama/llama-3.1-405b", "name": "Llama 3.1 405B"},
            {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5"},
        ]


    if provider_id == "openai":
         return [
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
        ]

    return []
