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
    ) -> str:
        """
        Invoke the Deep Agent and return the assistant's response text.
        """
        # 1. Create LLM with connection resilience from settings
        llm = await LLMFactory.get_llm(
            role="orchestrator", temperature=0, session=session,
        )
        llm.max_retries = settings.llm_max_retries
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
    ):
        """
        Stream the Deep Agent response, yielding text chunks as they arrive.
        Uses LangGraph's astream_events v2 API.
        """
        llm = await LLMFactory.get_llm(
            role="orchestrator", temperature=0, session=session,
        )
        llm.max_retries = settings.llm_max_retries
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
