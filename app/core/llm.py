"""
LLM Factory — Creates LangChain chat model instances from config.

Uses a provider registry (dict) instead of if/elif chains (OCP).
Adding a new provider = adding one entry to the registry.
"""

from enum import Enum
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.core.config import settings
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.repositories.llm_config_repository import LLMConfigRepository


class LLMProvider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Provider Registry — add new providers here (OCP)
# ---------------------------------------------------------------------------

def _create_openrouter(model_name: str, temperature: float, **kwargs):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=model_name,
        temperature=temperature,
        **kwargs,
    )


def _create_openai(model_name: str, temperature: float, **kwargs):
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not found in settings")
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model_name or "gpt-4-turbo-preview",
        temperature=temperature,
        **kwargs,
    )


def _create_anthropic(model_name: str, temperature: float, **kwargs):
    return ChatAnthropic(
        anthropic_api_key=settings.anthropic_api_key,
        model_name=model_name or "claude-3-opus-20240229",
        temperature=temperature,
        **kwargs,
    )


def _create_gemini(model_name: str, temperature: float, **kwargs):
    return ChatGoogleGenerativeAI(
        google_api_key=settings.gemini_api_key,
        model=model_name or "gemini-pro",
        temperature=temperature,
        **kwargs,
    )


def _create_ollama(model_name: str, temperature: float, **kwargs):
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name or "llama3.2",
        temperature=temperature,
        **kwargs,
    )


_PROVIDER_REGISTRY: Dict[str, Any] = {
    LLMProvider.OPENROUTER: _create_openrouter,
    LLMProvider.OPENAI: _create_openai,
    LLMProvider.ANTHROPIC: _create_anthropic,
    LLMProvider.GEMINI: _create_gemini,
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
            if provider == LLMProvider.OPENROUTER:
                model_name = settings.openrouter_model
            elif provider == LLMProvider.OLLAMA:
                model_name = settings.default_llm_model or "llama3.2"
            else:
                model_name = settings.default_llm_model

        logger.info(f"Initializing LLM: role={role}, provider={provider}, model={model_name}")

        # 4. Create via registry
        factory_fn = _PROVIDER_REGISTRY.get(provider)
        if not factory_fn:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return factory_fn(model_name=model_name, temperature=temperature, **kwargs)
