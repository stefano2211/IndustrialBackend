"""
LLM Factory — Creates LangChain chat model instances from config.

Uses a provider registry (dict) instead of if/elif chains (OCP).
Adding a new provider = adding one entry to the registry.
"""

from enum import Enum
from typing import Optional, Any, Dict
from langchain_ollama import ChatOllama
from app.core.config import settings
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.repositories.llm_config_repository import LLMConfigRepository


class LLMProvider(str, Enum):
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Provider Registry — add new providers here (OCP)
# ---------------------------------------------------------------------------


def _create_ollama(model_name: str, temperature: float, **kwargs):
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name or "llama3.1:8b",
        temperature=temperature,
        **kwargs,
    )


_PROVIDER_REGISTRY: Dict[str, Any] = {
    LLMProvider.OLLAMA: _create_ollama,
}


# ---------------------------------------------------------------------------
# Role → default provider/model resolution
# ---------------------------------------------------------------------------

_ROLE_DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "orchestrator": {
        "provider": settings.orchestrator_llm_provider,
        "model": settings.orchestrator_llm_model,
    },
    "subagent": {
        "provider": settings.subagent_llm_provider,
        "model": settings.subagent_llm_model,
    },
    "extractor": {
        "provider": settings.extractor_llm_provider,
        "model": settings.extractor_llm_model,
    },
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMFactory:
    @staticmethod
    async def get_db_config(session: AsyncSession, role: str) -> Optional[Dict[str, str]]:
        repo = LLMConfigRepository(session)
        config = await repo.get_config(role)
        if config:
            return {"provider": config.provider, "model_name": config.model_name}
        return None

    @staticmethod
    async def get_llm(
        role: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0,
        session: Optional[AsyncSession] = None,
        **kwargs,
    ) -> Any:
        # 1. Try DB config first
        if session and role and not (provider and model_name):
            db_config = await LLMFactory.get_db_config(session, role)
            if db_config:
                provider = provider or db_config["provider"]
                model_name = model_name or db_config["model_name"]

        # 2. Resolve from role defaults
        if role and role in _ROLE_DEFAULTS:
            defaults = _ROLE_DEFAULTS[role]
            provider = provider or defaults.get("provider")
            model_name = model_name or defaults.get("model")

        # 3. Global fallback
        provider = provider or settings.default_llm_provider
        if not model_name:
            if provider == LLMProvider.OLLAMA:
                model_name = settings.default_llm_model or "llama3.1:8b"
            else:
                model_name = settings.default_llm_model

        logger.info(f"Initializing LLM: role={role}, provider={provider}, model={model_name}")

        # 4. Create via registry
        factory_fn = _PROVIDER_REGISTRY.get(provider)
        if not factory_fn:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return factory_fn(model_name=model_name, temperature=temperature, **kwargs)
