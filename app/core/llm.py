"""
LLM Factory — Creates LangChain chat model instances from config.

Uses a provider registry (dict) instead of if/elif chains (OCP).
Adding a new provider = adding one entry to the registry.
"""

from enum import Enum
from typing import Optional, Any, Dict
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from app.core.config import settings
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.repositories.llm_config_repository import LLMConfigRepository
from app.persistence.repositories.settings_repository import SettingsRepository


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


# ---------------------------------------------------------------------------
# Provider Registry — add new providers here (OCP)
# ---------------------------------------------------------------------------


def _create_ollama(model_name: str, temperature: float, base_url: Optional[str] = None, **kwargs):
    return ChatOllama(
        base_url=base_url or settings.ollama_base_url,
        model=model_name or "llama3.1:8b",
        temperature=temperature,
        **kwargs,
    )


def _create_openrouter(model_name: str, temperature: float, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    # Set a reasonable default max_tokens to avoid OpenRouter credit check failures (402)
    # If the user doesn't provide one, we default to 4096.
    if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
        kwargs["max_tokens"] = 4096
        
    return ChatOpenAI(
        openai_api_key=api_key or settings.openrouter_api_key,
        openai_api_base=base_url or settings.openrouter_base_url,
        model=model_name or "openai/gpt-4o",
        temperature=temperature,
        **kwargs,
    )


_PROVIDER_REGISTRY: Dict[str, Any] = {
    LLMProvider.OLLAMA: _create_ollama,
    LLMProvider.OPENROUTER: _create_openrouter,
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
        # 1. Try DB config first (optional, matches role if provided)
        if session and role and not (provider and model_name):
            db_config = await LLMFactory.get_db_config(session, role)
            if db_config:
                provider = provider or db_config["provider"]
                model_name = model_name or db_config["model_name"]

        # 2. Global fallback (No more role defaults)
        provider = provider or settings.default_llm_provider
        if not model_name:
            if provider == LLMProvider.OLLAMA:
                model_name = settings.default_llm_model or "llama3.1:8b"
            else:
                model_name = settings.default_llm_model

        # 3. Load provider settings from DB if possible
        factory_kwargs = {}
        if session:
            settings_repo = SettingsRepository(session)
            sys_settings = await settings_repo.get_settings()
            if provider == LLMProvider.OLLAMA:
                factory_kwargs["base_url"] = sys_settings.ollama_base_url
            elif provider == LLMProvider.OPENROUTER:
                factory_kwargs["api_key"] = sys_settings.openrouter_api_key
                factory_kwargs["base_url"] = sys_settings.openrouter_base_url

        logger.info(f"Initializing Unified LLM: provider={provider}, model={model_name} (requested role={role})")

        # 4. Create via registry
        factory_fn = _PROVIDER_REGISTRY.get(provider)
        if not factory_fn:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return factory_fn(model_name=model_name, temperature=temperature, **{**factory_kwargs, **kwargs})
