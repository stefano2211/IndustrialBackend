from enum import Enum
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from loguru import logger

class LLMProvider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LLMFactory:
    @staticmethod
    def get_llm(
        role: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0,
        **kwargs
    ) -> Any:
        # 1. Determine Provider
        if not provider:
            if role == "orchestrator" and settings.orchestrator_llm_provider:
                provider = settings.orchestrator_llm_provider
            elif role == "subagent" and settings.subagent_llm_provider:
                provider = settings.subagent_llm_provider
            elif role == "extractor" and settings.extractor_llm_provider:
                provider = settings.extractor_llm_provider
            else:
                provider = settings.default_llm_provider
        
        # 2. Determine Model Name
        if not model_name:
            if role == "orchestrator" and settings.orchestrator_llm_model:
                model_name = settings.orchestrator_llm_model
            elif role == "subagent" and settings.subagent_llm_model:
                model_name = settings.subagent_llm_model
            elif role == "extractor" and settings.extractor_llm_model:
                model_name = settings.extractor_llm_model
            elif provider == LLMProvider.OPENROUTER:
                model_name = settings.openrouter_model
            else:
                model_name = settings.default_llm_model
                
        logger.info(f"Initializing LLM: role={role}, provider={provider}, model={model_name}")

        if provider == LLMProvider.OPENROUTER:
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                model=model_name,
                temperature=temperature,
                **kwargs
            )
        
        elif provider == LLMProvider.OPENAI:
            if not settings.openai_api_key:
                logger.warning("OPENAI_API_KEY not found in settings, using default")
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model=model_name or "gpt-4-turbo-preview",
                temperature=temperature,
                **kwargs
            )
            
        elif provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                anthropic_api_key=settings.anthropic_api_key,
                model_name=model_name or "claude-3-opus-20240229",
                temperature=temperature,
                **kwargs
            )
            
        elif provider == LLMProvider.GEMINI:
            return ChatGoogleGenerativeAI(
                google_api_key=settings.gemini_api_key,
                model=model_name or "gemini-pro",
                temperature=temperature,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
