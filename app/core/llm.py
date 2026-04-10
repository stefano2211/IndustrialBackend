"""
LLM Factory — Creates LangChain chat model instances from config.

Uses a provider registry (dict) instead of if/elif chains (OCP).
Adding a new provider = adding one entry to the registry.
"""

from enum import Enum
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from app.core.config import settings
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.repositories.llm_config_repository import LLMConfigRepository
from app.persistence.repositories.settings_repository import SettingsRepository


class LLMProvider(str, Enum):
    VLLM = "vllm"
    OPENROUTER = "openrouter"


# ---------------------------------------------------------------------------
# Provider Registry — add new providers here (OCP)
# ---------------------------------------------------------------------------


def _create_vllm(model_name: str, temperature: float, base_url: Optional[str] = None, **kwargs):
    # Map max_tokens appropriately
    streaming = kwargs.pop("streaming", True)
    
    # Ensure stop tokens are always present to avoid infinite loops
    if "stop" not in kwargs:
        kwargs["stop"] = ["<|im_end|>", "<|endoftext|>", "Assistant:", "User:"]

    return ChatOpenAI(
        openai_api_key="EMPTY",  # vLLM doesn't require an API key by default
        openai_api_base=base_url or settings.vllm_base_url,
        model=model_name or settings.default_llm_model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=kwargs.pop("max_tokens", 2048),
        **kwargs,
    )


def _create_openrouter(model_name: str, temperature: float, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
        kwargs["max_tokens"] = 2048 
        
    top_k = kwargs.pop("top_k", None)
    
    logger.info(f"Creating OpenRouter chat model: {model_name} at {base_url or settings.openrouter_base_url}")
    
    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k

    streaming = kwargs.pop("streaming", True)

    return ChatOpenAI(
        openai_api_key=api_key or settings.openrouter_api_key,
        openai_api_base=base_url or settings.openrouter_base_url,
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
    LLMProvider.VLLM: _create_vllm,
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
        model_from_kwargs = kwargs.pop("model", None)
        model_name = model_name or model_from_kwargs

        if session and role and not (provider and model_name):
            db_config = await LLMFactory.get_db_config(session, role)
            if db_config:
                provider = provider or db_config["provider"]
                model_name = model_name or db_config["model_name"]

        # Fetch DB settings once for both provider resolution and factory kwargs
        sys_settings = None
        if session:
            settings_repo = SettingsRepository(session)
            sys_settings = await settings_repo.get_settings()

        if not provider:
            if model_name and "/" in str(model_name):
                provider = LLMProvider.OPENROUTER
            elif sys_settings is not None:
                if hasattr(sys_settings, 'openrouter_enabled') and sys_settings.openrouter_enabled:
                    provider = LLMProvider.OPENROUTER
                else:
                    provider = LLMProvider.VLLM
            else:
                provider = settings.default_llm_provider

        if not model_name or model_name.lower() in ["vllm", "openrouter", "ollama"]:
            if provider == LLMProvider.VLLM:
                model_name = settings.default_llm_model
            else:
                model_name = "openai/gpt-4o"

        factory_kwargs = {}
        if sys_settings is not None:
            if provider == LLMProvider.VLLM:
                factory_kwargs["base_url"] = getattr(sys_settings, "vllm_base_url", settings.vllm_base_url)
            elif provider == LLMProvider.OPENROUTER:
                factory_kwargs["api_key"] = sys_settings.openrouter_api_key
                factory_kwargs["base_url"] = sys_settings.openrouter_base_url

        logger.info(f"Initializing Unified LLM: provider={provider}, model={model_name} (requested role={role})")

        # Fallback handling in case DB still contains "ollama" provider from migration
        if provider == "ollama":
            provider = LLMProvider.VLLM
            logger.info("Migrated legacy 'ollama' provider to 'vllm' dynamically")

        factory_fn = _PROVIDER_REGISTRY.get(provider)
        if not factory_fn:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return factory_fn(model_name=model_name, temperature=temperature, **{**factory_kwargs, **kwargs})
