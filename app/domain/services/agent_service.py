"""
Agent Service — Orchestrates Deep Agent invocations.

Encapsulates the logic of:
  1. Creating the LLM with connection resilience
  2. Building the Deep Agent
  3. Invoking the agent with the correct config
  4. Extracting the final response

This isolates the chat endpoint from agent internals (Dependency Inversion).
"""

import uuid
from typing import Any, AsyncGenerator, Dict, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from deepagents.backends.utils import create_file_data
from loguru import logger

from app.core.config import settings
from app.core.llm import LLMFactory, LLMProvider
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

    def _extract_params(self, params_obj_or_dict) -> dict:
        """Extract parameters from a Pydantic model or a dictionary into a clean dict."""
        if not params_obj_or_dict:
            return {}
            
        if isinstance(params_obj_or_dict, dict):
            return {k: v for k, v in params_obj_or_dict.items() if v is not None}
            
        # Assume it's an object (SimpleNamespace or Pydantic model)
        extracted = {}
        for attr in ['temperature', 'max_tokens', 'top_p', 'top_k', 'seed', 'stop_sequence']:
            if hasattr(params_obj_or_dict, attr):
                val = getattr(params_obj_or_dict, attr)
                if val is not None:
                    extracted[attr] = val
        return extracted

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
        knowledge_base_id: str | None = None,
        mcp_source_id: str | None = None,
        session: Any = None,
        checkpointer=None,
        store=None,
        params=None,
        model_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Invoke the Deep Agent and return the assistant's response text.
        """
        # 1. Resolve Provider and Model Name from DB if model_id is provided
        provider = None
        model_name = None
        db_model = None
        
        if model_id and session:
            model_repo = ModelRepository(session)
            db_model = await model_repo.get_by_id(model_id)
            if db_model:
                # base_model_id is usually "provider:model"
                if ":" in db_model.base_model_id:
                    provider, model_name = db_model.base_model_id.split(":", 1)
                elif db_model.base_model_id in [p.value for p in LLMProvider]:
                    # If it's just a provider name, treat as default model for that provider
                    provider = db_model.base_model_id
                    model_name = None
                else:
                    model_name = db_model.base_model_id

        # 2. Collect and merge parameters
        merged_params = {}
        if db_model and hasattr(db_model, 'params') and db_model.params:
            merged_params.update(self._extract_params(db_model.params))
            
        if params:
            merged_params.update(self._extract_params(params))

        # Handle stop sequence mapping
        if "stop_sequence" in merged_params:
            stop_val = merged_params.pop("stop_sequence")
            if stop_val:
                merged_params["stop"] = [stop_val]

        # 3. Create LLM with resolved config and merged params
        llm = await LLMFactory.get_llm(
            provider=provider,
            model_name=model_name,
            session=session,
            **merged_params
        )
        
        # 4. Apply additional settings
        if hasattr(llm, "max_retries"):
            llm.max_retries = settings.llm_max_retries
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = settings.llm_request_timeout
        
        # 5. Handle system prompt composition
        if params and not params.system_prompt and db_model and db_model.system_prompt:
            params.system_prompt = db_model.system_prompt
        elif not params and db_model and db_model.system_prompt:
            from types import SimpleNamespace
            params = SimpleNamespace(system_prompt=db_model.system_prompt)

        # 2. Build agent with dynamic tools context
        from app.persistence.repositories.tool_config_repository import ToolConfigRepository
        tool_repo = ToolConfigRepository(session)
        
        # Filter by source_id if provided
        if mcp_source_id == "none":
             all_tools = []
        elif mcp_source_id:
             # Fetch tools for specific source (implicitly verified by source_id lookups usually, 
             # but here we should ideally double check ownership if we were strict)
             all_tools = await tool_repo.get_by_source(uuid.UUID(mcp_source_id))
        else:
             # Default: Use all tools OWNED by the user
             all_tools = await tool_repo.get_all_by_user(uuid.UUID(user_id))
        
        dynamic_tools_list = []
        import json
        for t in all_tools:
            # Include name, description, and schema to guide the agent dynamically
            schema_str = json.dumps(t.parameter_schema) if t.parameter_schema else "{}"
            dynamic_tools_list.append(f"- Name: {t.name}\n  Description: {t.description}\n  Parameters schema: {schema_str}")
        
        tools_context = "\n".join(dynamic_tools_list) if dynamic_tools_list else "No dynamic tools currently registered."
        
        custom_prompt = db_model.system_prompt if db_model else None
        agent = create_industrial_agent(
            model=llm, checkpointer=checkpointer, store=store,
            custom_system_prompt=custom_prompt,
            mcp_tools_context=tools_context
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

        logger.info(f"Invoking agent for thread {thread_id} with query: {query}")
        response = await agent.ainvoke(
            {
                "messages": self._build_messages(query, params),
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=config,
        )

        # 4. Extract final message and return with model info
        last_message = response["messages"][-1]
        content = last_message.content
        # Resolved model ID is either from db_model or defaults
        resolved_model_id = model_id or (db_model.id if db_model else (model_name or "default"))
        return content, resolved_model_id

    async def stream(
        self,
        *,
        user_id: str,
        thread_id: str,
        query: str,
        knowledge_base_id: str | None = None,
        mcp_source_id: str | None = None,
        session: Any = None,
        checkpointer=None,
        store=None,
        params=None,
        model_id: str | None = None,
    ):
        """
        Stream the Deep Agent response, yielding text chunks as they arrive, 
        plus a final metadata chunk with model info.
        Uses LangGraph's astream_events v2 API.
        """
        # 1. Resolve Provider and Model Name from DB if model_id is provided
        provider = None
        model_name = None
        db_model = None
        
        if model_id and session:
            model_repo = ModelRepository(session)
            db_model = await model_repo.get_by_id(model_id)
            if db_model:
                if ":" in db_model.base_model_id:
                    provider, model_name = db_model.base_model_id.split(":", 1)
                elif db_model.base_model_id in [p.value for p in LLMProvider]:
                    provider = db_model.base_model_id
                    model_name = None
                else:
                    model_name = db_model.base_model_id

        # 2. Collect and merge parameters
        merged_params = {}
        if db_model and hasattr(db_model, 'params') and db_model.params:
            merged_params.update(self._extract_params(db_model.params))
            
        if params:
            merged_params.update(self._extract_params(params))

        # Handle stop sequence mapping
        if "stop_sequence" in merged_params:
            stop_val = merged_params.pop("stop_sequence")
            if stop_val:
                merged_params["stop"] = [stop_val]

        # 3. Create LLM with resolved config and merged params
        llm = await LLMFactory.get_llm(
            provider=provider,
            model_name=model_name,
            session=session,
            **merged_params
        )
        
        # 4. Apply additional settings
        if hasattr(llm, "max_retries"):
            llm.max_retries = settings.llm_max_retries
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = settings.llm_request_timeout

        # 5. Handle system prompt composition
        if params and not params.system_prompt and db_model and db_model.system_prompt:
            params.system_prompt = db_model.system_prompt
        elif not params and db_model and db_model.system_prompt:
            from types import SimpleNamespace
            params = SimpleNamespace(system_prompt=db_model.system_prompt)

        # 2. Build agent with dynamic tools context
        from app.persistence.repositories.tool_config_repository import ToolConfigRepository
        tool_repo = ToolConfigRepository(session)
        
        # Filter by source_id if provided
        if mcp_source_id == "none":
             all_tools = []
        elif mcp_source_id:
             all_tools = await tool_repo.get_by_source(uuid.UUID(mcp_source_id))
        else:
             all_tools = await tool_repo.get_all()
        
        dynamic_tools_list = []
        import json
        for t in all_tools:
            # Include name, description, and schema to guide the agent dynamically
            schema_str = json.dumps(t.parameter_schema) if t.parameter_schema else "{}"
            dynamic_tools_list.append(f"- Name: {t.name}\n  Description: {t.description}\n  Parameters schema: {schema_str}")
        
        tools_context = "\n".join(dynamic_tools_list) if dynamic_tools_list else "No dynamic tools currently registered."
        
        custom_prompt = db_model.system_prompt if db_model else None
        agent = create_industrial_agent(
            model=llm, checkpointer=checkpointer, store=store,
            custom_system_prompt=custom_prompt,
            mcp_tools_context=tools_context
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "session": session,
            }
        }

        any_token_yielded = False
        async for event in agent.astream_events(
            {
                "messages": self._build_messages(query, params),
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            
            if kind == "on_chat_model_start":
                any_token_yielded = False

            if kind in ["on_chat_model_stream", "on_llm_stream"]:
                data = event.get("data", {})
                chunk = data.get("chunk")
                
                text = ""
                if hasattr(chunk, "content"): text = chunk.content
                elif hasattr(chunk, "text"): text = chunk.text
                elif isinstance(chunk, dict): text = chunk.get("content", "")
                
                if text:
                    any_token_yielded = True
                    yield text
                continue

            if kind == "on_chat_model_end":
                # If this turn finished and yielded no tokens (e.g. filtered, tool call preamble omitted),
                # we check if there's content we should send.
                data = event.get("data", {})
                output = data.get("output")
                if output:
                    text = ""
                    if hasattr(output, "content"): text = output.content
                    elif hasattr(output, "text"): text = output.text
                    
                    # If it's a tool call turn, we usually don't want to yield the tool arguments as text 
                    # unless they are part of a thought process. For now, we yield visible content.
                    if text and not (hasattr(output, "tool_calls") and output.tool_calls and not text):
                        # Yield the full content as a fallback if no tokens were streamed
                        yield text

            # Robust fallback for token extraction from chunks
            chunk = event.get("data", {}).get("chunk")
            if chunk and kind not in ["on_chat_model_end"]: # avoided repeated content
                text = ""
                if hasattr(chunk, "content"): text = chunk.content
                elif hasattr(chunk, "text"): text = chunk.text
                if text:
                    yield text

        # Yield metadata about the resolved model at the end
        resolved_model_id = model_id or (db_model.id if db_model else (model_name or "default"))
        yield {"model_id": resolved_model_id}
