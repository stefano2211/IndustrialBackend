"""
Agent Service — Orchestrates Deep Agent invocations.

Encapsulates the logic of:
  1. Creating the LLM with connection resilience
  2. Building the Deep Agent
  3. Invoking the agent with the correct config
  4. Extracting the final response

This isolates the chat endpoint from agent internals (Dependency Inversion).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from deepagents.backends.utils import create_file_data
from loguru import logger

from app.core.config import settings
from app.core.llm import LLMFactory
from app.domain.agent.deep_agent import create_industrial_agent
from app.domain.agent.prompts import AGENTS_MD_CONTENT
from app.persistence.repositories.model_repository import ModelRepository


class AgentService:
    """
    Stateless service that manages Deep Agent lifecycle per request.

    Usage:
        service = AgentService()
        answer = await service.invoke(
            user_id="...", thread_id="...", query="...",
            knowledge_base_id="...", session=session,
            checkpointer=checkpointer, store=store,
        )
    """

    def _apply_params(self, llm, params):
        """Apply user-specified model parameters to the LLM instance."""
        if not params:
            return
        if params.temperature is not None:
            llm.temperature = params.temperature
        if params.max_tokens is not None:
            llm.max_tokens = params.max_tokens
        if params.top_p is not None:
            llm.top_p = params.top_p
        # top_k and seed are provider-specific, set via model_kwargs
        kwargs = getattr(llm, 'model_kwargs', {}) or {}
        if params.top_k is not None:
            kwargs['top_k'] = params.top_k
        if params.seed is not None:
            kwargs['seed'] = params.seed
        if params.stop_sequence:
            llm.stop = [params.stop_sequence]
        if kwargs:
            llm.model_kwargs = kwargs

    def _apply_model_config(self, llm, model_config):
        """Apply model-specific configuration from the DB."""
        if not model_config:
            return
            
        # 1. Apply system prompt if present in model config
        # (This can be overridden by request params later)
        if hasattr(model_config, 'system_prompt') and model_config.system_prompt:
            # We don't set it directly on llm, but store it for message building
            pass
            
        # 2. Apply parameters from model config
        if hasattr(model_config, 'params') and model_config.params:
            # params in DB is a dict
            params_dict = model_config.params
            if 'temperature' in params_dict:
                llm.temperature = params_dict['temperature']
            if 'max_tokens' in params_dict:
                llm.max_tokens = params_dict['max_tokens']
            # etc. (could add more)

    def _build_messages(self, query: str, params=None):
        """Build the messages list, optionally prepending a system prompt."""
        messages = []
        if params and params.system_prompt:
            messages.append(SystemMessage(content=params.system_prompt))
        messages.append(HumanMessage(content=query))
        return messages

    async def invoke(
        self,
        *,
        user_id: str,
        thread_id: str,
        query: str,
        knowledge_base_id: str | None,
        session,
        checkpointer=None,
        store=None,
        params=None,
        model_id: str | None = None,
    ) -> str:
        """
        Invoke the Deep Agent and return the assistant's response text.
        """
        # 1. Resolve Provider and Model Name from DB if model_id is provided
        provider = None
        model_name = None
        db_model = None
        
        if model_id and session:
            model_repo = ModelRepository(session)
            db_model = await model_repo.get_model(model_id)
            if db_model:
                # base_model_id is usually "provider:model"
                if ":" in db_model.base_model_id:
                    provider, model_name = db_model.base_model_id.split(":", 1)
                else:
                    model_name = db_model.base_model_id

        # 2. Create LLM with resolved config
        llm = await LLMFactory.get_llm(
            provider=provider,
            model_name=model_name,
            temperature=0, 
            session=session,
        )
        
        # 3. Apply DB Model specifics (system prompt, params)
        if db_model:
            self._apply_model_config(llm, db_model)
            # If request params don't have a system prompt, use the model's one
            if params and not params.system_prompt and db_model.system_prompt:
                params.system_prompt = db_model.system_prompt
            elif not params and db_model.system_prompt:
                # Create a mini object/dict to hold the system prompt for build_messages
                from types import SimpleNamespace
                params = SimpleNamespace(system_prompt=db_model.system_prompt, temperature=None, max_tokens=None, top_p=None, top_k=None, seed=None, stop_sequence=None)
        if hasattr(llm, "max_retries"):
            llm.max_retries = settings.llm_max_retries
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = settings.llm_request_timeout
        self._apply_params(llm, params)

        # 2. Build agent
        agent = create_industrial_agent(
            model=llm, checkpointer=checkpointer, store=store,
        )

        # 3. Invoke with config
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "session": session,
            }
        }

        response = await agent.ainvoke(
            {
                "messages": self._build_messages(query, params),
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=config,
        )

        # 4. Extract final message
        return response["messages"][-1].content

    async def stream(
        self,
        *,
        user_id: str,
        thread_id: str,
        query: str,
        knowledge_base_id: str | None,
        session,
        checkpointer=None,
        store=None,
        params=None,
        model_id: str | None = None,
    ):
        """
        Stream the Deep Agent response, yielding text chunks as they arrive.
        Uses LangGraph's astream_events v2 API.
        """
        # 1. Resolve Provider and Model Name from DB if model_id is provided
        provider = None
        model_name = None
        db_model = None
        
        if model_id and session:
            model_repo = ModelRepository(session)
            db_model = await model_repo.get_model(model_id)
            if db_model:
                if ":" in db_model.base_model_id:
                    provider, model_name = db_model.base_model_id.split(":", 1)
                else:
                    model_name = db_model.base_model_id

        # 2. Create LLM with resolved config
        llm = await LLMFactory.get_llm(
            provider=provider,
            model_name=model_name,
            temperature=0, 
            session=session,
        )
        
        # 3. Apply DB Model specifics
        if db_model:
            self._apply_model_config(llm, db_model)
            if params and not params.system_prompt and db_model.system_prompt:
                params.system_prompt = db_model.system_prompt
            elif not params and db_model.system_prompt:
                from types import SimpleNamespace
                params = SimpleNamespace(system_prompt=db_model.system_prompt, temperature=None, max_tokens=None, top_p=None, top_k=None, seed=None, stop_sequence=None)
        if hasattr(llm, "max_retries"):
            llm.max_retries = settings.llm_max_retries
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = settings.llm_request_timeout
        self._apply_params(llm, params)

        agent = create_industrial_agent(
            model=llm, checkpointer=checkpointer, store=store,
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "session": session,
            }
        }

        async for event in agent.astream_events(
            {
                "messages": self._build_messages(query, params),
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            # Only yield actual text content from the LLM stream
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
