"""
Agent Service — Orchestrates Deep Agent invocations.

Encapsulates the logic of:
  1. Creating the LLM with connection resilience
  2. Building the Deep Agent
  3. Invoking the agent with the correct config
  4. Extracting the final response

This isolates the chat endpoint from agent internals (Dependency Inversion).
"""

from langchain_core.messages import HumanMessage
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
    ) -> str:
        """
        Invoke the Deep Agent and return the assistant's response text.

        Args:
            user_id: Current authenticated user ID.
            thread_id: Conversation thread ID (for checkpointer).
            query: The user's message.
            knowledge_base_id: Optional KB to scope vector search.
            session: Async DB session (passed through config for tools).
            checkpointer: LangGraph checkpointer for state persistence.
            store: LangGraph store for long-term memories.

        Returns:
            The agent's final response as a plain string.

        Raises:
            Exception: Any error from LLM or agent invocation.
        """
        # 1. Create LLM with connection resilience from settings
        llm = await LLMFactory.get_llm(
            role="orchestrator", temperature=0, session=session,
        )
        llm.max_retries = settings.llm_max_retries
        llm.request_timeout = settings.llm_request_timeout

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
                "messages": [HumanMessage(content=query)],
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=config,
        )

        # 4. Extract final message
        return response["messages"][-1].content
