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

from app.persistence.proactiva.repositories.llm_config_repository import LLMConfigRepository
from app.persistence.proactiva.repositories.settings_repository import SettingsRepository


class LLMProvider(str, Enum):
    VLLM = "vllm"


# ---------------------------------------------------------------------------
# Provider Registry — add new providers here (OCP)
# ---------------------------------------------------------------------------


def _create_vllm(model_name: str, temperature: float, base_url: Optional[str] = None, **kwargs):
    streaming = kwargs.get("streaming", True)
    kwargs.pop("streaming", None)
    kwargs.pop("top_k", None)

    if 'stop' not in kwargs:
        kwargs['stop'] = ['<turn|>']

    return ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=base_url or settings.vllm_base_url,
        model=model_name or settings.default_llm_model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=kwargs.pop("max_tokens", 2048),
        **kwargs,
    )


_PROVIDER_REGISTRY: Dict[str, Any] = {
    LLMProvider.VLLM: _create_vllm,
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
            provider = LLMProvider.VLLM

        if not model_name or model_name.lower() in ["vllm", "openrouter", "ollama"]:
            if provider == LLMProvider.VLLM:
                # Default to the generalist/base model when no specific model is requested.
                # Specific subagents (Expert, VL) will explicitly request their LoRA models.
                model_name = settings.generalist_llm_model
            else:
                model_name = "openai/gpt-4o"

        factory_kwargs = {}
        if provider == LLMProvider.VLLM:
            # Unified MoE endpoint — all agents share the same Gemma 4 backbone
            factory_kwargs["base_url"] = settings.vllm_base_url

        logger.info(f"Initializing Unified LLM: provider={provider}, model={model_name} (requested role={role})")

        # Fallback handling in case DB still contains "ollama" provider from migration
        if provider == "ollama":
            provider = LLMProvider.VLLM
            logger.info("Migrated legacy 'ollama' provider to 'vllm' dynamically")

        factory_fn = _PROVIDER_REGISTRY.get(provider)
        if not factory_fn:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return factory_fn(model_name=model_name, temperature=temperature, **{**factory_kwargs, **kwargs})


# ---------------------------------------------------------------------------
# Shared vLLM probe (LoRA adapter availability check)
# ---------------------------------------------------------------------------


async def _vllm_model_exists(base_url: str, model_name: str) -> bool:
    """
    Probe vLLM /v1/models to verify a model or LoRA adapter is loaded.

    LLMFactory.get_llm() never raises — it just creates a config object.
    The 404 only fires on the first actual request. This probe lets us check
    upfront and fall back to the base model when a LoRA hasn't been trained yet.
    """
    import httpx

    models_url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(models_url)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return any(m.get("id") == model_name for m in data)
    except Exception as exc:
        logger.debug(f"[LLMFactory] vLLM probe failed for '{model_name}': {exc}")
    return False
