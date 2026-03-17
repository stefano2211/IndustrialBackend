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
    # Set a larger context window (num_ctx) to avoid truncation (default is usually 4k)
    if "num_ctx" not in kwargs:
        kwargs["num_ctx"] = 50000 # Increased to 128k
    
    # Map max_tokens to num_predict for ChatOllama
    if "max_tokens" in kwargs:
        if kwargs["max_tokens"] is not None:
            kwargs["num_predict"] = kwargs.pop("max_tokens")
        else:
            kwargs.pop("max_tokens")
        
    streaming = kwargs.pop("streaming", True)
    
    return ChatOllama(
        base_url=base_url or settings.ollama_base_url,
        model=model_name or "qwen3.5:9b",
        temperature=temperature,
        streaming=streaming,
        **kwargs,
    )


def _create_openrouter(model_name: str, temperature: float, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    # Set a reasonable default max_tokens to avoid OpenRouter credit check failures (402)
    if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
        kwargs["max_tokens"] = 2048 # Reduced from 4096
        
    # OpenRouter/OpenAI-client compatibility: remove parameters not supported by ChatOpenAI
    # top_k is common in Ollama but not in OpenAI standard completions API
    top_k = kwargs.pop("top_k", None)
    
    logger.info(f"Creating OpenRouter chat model: {model_name} at {base_url or settings.openrouter_base_url}")
    
    # If top_k is provided, we can pass it in extra_body for OpenRouter to handle natively
    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k

    streaming = kwargs.pop("streaming", True)

    return ChatOpenAI(
        openai_api_key=api_key or settings.openrouter_api_key,
        openai_api_base=base_url or settings.openrouter_api_base_url if hasattr(settings, 'openrouter_api_base_url') else (base_url or settings.openrouter_base_url),
        model=model_name or "openai/gpt-4o",
        temperature=temperature,
        streaming=streaming,
        default_headers={
            "HTTP-Referer": "https://industrial-backend.ai",
            "X-Title": "Industrial Backend",
        },
        model_kwargs={"extra_body": extra_body} if extra_body else {},
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
        # Gracefully handle callers passing 'model' instead of 'model_name'
        model_from_kwargs = kwargs.pop("model", None)
        model_name = model_name or model_from_kwargs
        # 1. Try DB config first (optional, matches role if provided)
        if session and role and not (provider and model_name):
            db_config = await LLMFactory.get_db_config(session, role)
            if db_config:
                provider = provider or db_config["provider"]
                model_name = model_name or db_config["model_name"]

        # 2. Provider Prioritization from System Settings
        if not provider:
            # Auto-detect OpenRouter if model name contains a slash (e.g. "qwen/...")
            if model_name and "/" in str(model_name):
                provider = LLMProvider.OPENROUTER
            elif session:
                settings_repo = SettingsRepository(session)
                sys_settings = await settings_repo.get_settings()
                
                if sys_settings.ollama_enabled:
                    provider = LLMProvider.OLLAMA
                elif sys_settings.openrouter_enabled:
                    provider = LLMProvider.OPENROUTER
                else:
                    provider = settings.default_llm_provider
            else:
                provider = settings.default_llm_provider


        # 3. Model Name Resolution
        # If model_name is None OR matches a provider name, treat as "want default for this provider"
        if not model_name or model_name.lower() in ["ollama", "openrouter"]:
            if provider == LLMProvider.OLLAMA:
                # If the global default looks like an OpenRouter model (contains /), use a safe Ollama fallback
                if settings.default_llm_model and "/" in settings.default_llm_model:
                    model_name = "qwen3.5:9b"
                else:
                    model_name = settings.default_llm_model or "qwen3.5:9b"
            else:
                model_name = settings.default_llm_model

        # 4. Load provider configuration from DB
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

        # 5. Create via registry
        factory_fn = _PROVIDER_REGISTRY.get(provider)
        if not factory_fn:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return factory_fn(model_name=model_name, temperature=temperature, **{**factory_kwargs, **kwargs})
