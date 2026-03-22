"""
Agent Service — Orchestrates Deep Agent invocations.

Encapsulates the logic of:
  1. Creating the LLM with connection resilience
  2. Building the Deep Agent
  3. Invoking the agent with the correct config
  4. Extracting the final response

This isolates the chat endpoint from agent internals (Dependency Inversion).
"""

import json
import re
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

    @staticmethod
    def _build_tool_context(t) -> str:
        """
        Build a rich, structured context block for a single ToolConfig so the LLM
        knows exactly how to invoke it via call_dynamic_mcp.

        Extracts:
          - HTTP method and full URL pattern
          - Path parameters (from {param} in the URL)
          - Query / body parameters from parameter_schema
          - Response field hints from parameter_schema["response"]
          - Filterable fields from parameter_schema["filterable_schema"] (key_figures + key_values)
        """
        config_data: dict = t.config or {}

        # ── URL & Method ────────────────────────────────────────────────────
        effective_url = config_data.get("url") or t.api_url or "(unknown url)"
        effective_method = (config_data.get("method") or t.method or "GET").upper()
        transport = config_data.get("transport", "rest")

        # ── Path parameters (from {curly_braces} in the URL) ────────────────
        path_params: list[str] = re.findall(r'\{(.*?)\}', effective_url)

        # ── Parameter schema breakdown ───────────────────────────────────────
        schema: dict = t.parameter_schema or {}
        properties: dict = schema.get("properties", {})
        required_fields: list = schema.get("required", [])
        response_fields: dict = schema.get("response", {})  # optional hint
        filterable_schema: dict = schema.get("filterable_schema", {})  # new: real discoverable fields

        lines = [
            f"━━━ TOOL: {t.name} ━━━",
            f"  Description : {t.description}",
            f"  Transport   : {transport.upper()}",
            f"  Method      : {effective_method}",
            f"  URL pattern : {effective_url}",
        ]

        # ── Path params summary ──────────────────────────────────────────────
        if path_params:
            lines.append(f"  Path params : {', '.join(path_params)}  ← required, embedded in the URL")

        # ── Input parameters ─────────────────────────────────────────────────
        if properties:
            lines.append("  Parameters:")
            for param_name, param_def in properties.items():
                p_type = param_def.get("type", "any")
                p_desc = param_def.get("description", "")
                p_enum = param_def.get("enum")
                p_default = param_def.get("default")
                is_required = param_name in required_fields or param_name in path_params
                req_tag = "[required]" if is_required else "[optional]"
                enum_hint = f"  choices: {p_enum}" if p_enum else ""
                default_hint = f"  default: {p_default}" if p_default is not None else ""
                lines.append(
                    f"    - {param_name} ({p_type}) {req_tag}: {p_desc}{enum_hint}{default_hint}"
                )
        elif path_params:
            # Fallback when schema is empty but path params exist
            lines.append("  Parameters:")
            for pp in path_params:
                lines.append(f"    - {pp} (string) [required]: Path parameter for the URL")
        else:
            placement = "body" if effective_method in ("POST", "PUT", "PATCH") else "query string"
            lines.append(f"  Parameters  : none — send an empty dict {{}} as arguments ({placement})")

        # ── Response hints ───────────────────────────────────────────────────
        if response_fields:
            lines.append("  Expected response fields:")
            for field, field_def in response_fields.items():
                f_type = field_def.get("type", "any") if isinstance(field_def, dict) else str(field_def)
                f_unit = field_def.get("unit", "") if isinstance(field_def, dict) else ""
                f_desc = field_def.get("description", "") if isinstance(field_def, dict) else ""
                unit_str = f" [{f_unit}]" if f_unit else ""
                desc_str = f" — {f_desc}" if f_desc else ""
                lines.append(f"    - {field} ({f_type}){unit_str}{desc_str}")

        # ── Filterable fields (injected from discovered schema) ──────────────
        kf_fields: list = filterable_schema.get("key_figures", [])
        kv_fields: dict = filterable_schema.get("key_values", {})

        if kf_fields or kv_fields:
            lines.append("  Filterable fields (use in 'arguments' → key_values / key_figures):")
            if kf_fields:
                lines.append(f"    [NUMERIC]  — key_figures fields: {', '.join(kf_fields)}")
                lines.append( "                 Usage: {\"key_figures\": [{\"field\": \"<name>\", \"min\": X, \"max\": Y}]}")
            if kv_fields:
                lines.append( "    [CATEGORICAL] — key_values fields and available values:")
                for kv_field, kv_vals in kv_fields.items():
                    # Limit to first 15 values to keep context compact
                    vals_preview = kv_vals[:15]
                    suffix = f" ...+{len(kv_vals) - 15} more" if len(kv_vals) > 15 else ""
                    lines.append(f"      · {kv_field}: {vals_preview}{suffix}")
                lines.append( "                 Usage: {\"key_values\": {\"<field>\": [\"<value1>\", ...]}}")

        return "\n".join(lines)

    def _interpolate_variables(self, text: str, tools: List) -> str:
        """
        Parses `{{tool_name.property}}` variables in the text and replaces them 
        with stringified values from the tool's parameter_schema.
        """
        if not text or "{{" not in text:
            return text

        def replacer(match):
            var_path = match.group(1).strip()
            parts = var_path.split('.')
            if len(parts) < 2:
                return match.group(0)
                
            # Check if this is a deep field reference (e.g. source.tool.key_figures.Temperatura)
            # Find the keyword index to support field names that contain dots:
            if len(parts) >= 4:
                for kw in ['params', 'key_figures', 'key_values']:
                    try:
                        idx = parts.index(kw)
                        if idx >= 2:
                            return ".".join(parts[idx+1:])
                    except ValueError:
                        continue
            
            prop = parts[-1]
            tool_name = parts[-2]
            
            # Find the tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                return match.group(0)
                
            schema = tool.parameter_schema or {}
            
            if prop == 'params':
                # Return stringified list of required/optional params
                params_dict = schema.get("properties", {})
                return ", ".join(params_dict.keys()) if params_dict else "Ninguno"
                
            filterable = schema.get("filterable_schema", {})
            if prop == 'key_figures':
                figures = filterable.get("key_figures", [])
                return ", ".join(figures) if figures else "Ninguno"
            elif prop == 'key_values':
                values = filterable.get("key_values", {})
                return ", ".join(values.keys()) if values else "Ninguno"
                
            return match.group(0)

        # Regex to find everything inside {{ }}
        interpolated = re.sub(r'\{\{(.*?)\}\}', replacer, text)
        if interpolated != text:
            logger.info(f"[AgentService] Interpolated variables in text. New text: {interpolated}")
        return interpolated


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
        
        dynamic_tools_list = [
            self._build_tool_context(t) for t in all_tools
        ]
        tools_context = "\n\n".join(dynamic_tools_list) if dynamic_tools_list else "No dynamic tools currently registered."
        
        custom_prompt = db_model.system_prompt if db_model else None
        agent = create_industrial_agent(
            model=llm, checkpointer=checkpointer, store=store,
            custom_system_prompt=custom_prompt,
            mcp_tools_context=tools_context,
            enable_knowledge=(knowledge_base_id != "none"),
            enable_mcp=(mcp_source_id != "none")
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

        # Interpolate variables before building messages
        query = self._interpolate_variables(query, all_tools)
        if params and hasattr(params, 'system_prompt') and params.system_prompt:
            params.system_prompt = self._interpolate_variables(params.system_prompt, all_tools)

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
        
        dynamic_tools_list = [
            self._build_tool_context(t) for t in all_tools
        ]
        tools_context = "\n\n".join(dynamic_tools_list) if dynamic_tools_list else "No dynamic tools currently registered."
        
        custom_prompt = db_model.system_prompt if db_model else None
        agent = create_industrial_agent(
            model=llm, checkpointer=checkpointer, store=store,
            custom_system_prompt=custom_prompt,
            mcp_tools_context=tools_context,
            enable_knowledge=(knowledge_base_id != "none"),
            enable_mcp=(mcp_source_id != "none")
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "knowledge_base_id": knowledge_base_id,
                "session": session,
            }
        }

        # State for filtering <think>...</think> tokens emitted by reasoning models (Qwen, DeepSeek r1, etc.)
        any_token_yielded = False
        last_output_had_tool_calls = False
        inside_think_block = False
        think_buffer = ""

        # Interpolate variables before building messages
        query = self._interpolate_variables(query, all_tools)
        if params and hasattr(params, 'system_prompt') and params.system_prompt:
            params.system_prompt = self._interpolate_variables(params.system_prompt, all_tools)

        async for event in agent.astream_events(
            {
                "messages": self._build_messages(query, params),
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Reset per-model-turn counters when a new LLM call starts
            if kind == "on_chat_model_start":
                any_token_yielded = False
                last_output_had_tool_calls = False
                inside_think_block = False
                think_buffer = ""

            # --- Primary: stream tokens as they arrive ---
            if kind in ["on_chat_model_stream", "on_llm_stream"]:
                data = event.get("data", {})
                chunk = data.get("chunk")

                text = ""
                if hasattr(chunk, "content"):
                    text = chunk.content
                elif hasattr(chunk, "text"):
                    text = chunk.text
                elif isinstance(chunk, dict):
                    text = chunk.get("content", "")



                if text:
                    # Filter out <think>...</think> blocks emitted by reasoning models
                    think_buffer += text
                    visible = ""
                    while think_buffer:
                        if inside_think_block:
                            end_idx = think_buffer.find("</think>")
                            if end_idx != -1:
                                # Found end of think block
                                think_buffer = think_buffer[end_idx + len("</think>"):]
                                inside_think_block = False
                            else:
                                # Still inside think block. We can discard most of the buffer
                                # to save memory, but keep the last 7 chars in case they are 
                                # a partial "</think>" tag.
                                if len(think_buffer) > 7:
                                    think_buffer = think_buffer[-7:]
                                break
                        else:
                            start_idx = think_buffer.find("<think>")
                            if start_idx != -1:
                                # Found think block start. Emit text before it.
                                visible += think_buffer[:start_idx]
                                think_buffer = think_buffer[start_idx + len("<think>"):]
                                inside_think_block = True
                            else:
                                # No think block start found.
                                # Emit everything EXCEPT the last 6 chars, which might be 
                                # a partial "<think>" tag.
                                if len(think_buffer) > 6:
                                    visible += think_buffer[:-6]
                                    think_buffer = think_buffer[-6:]
                                break

                    if visible:
                        any_token_yielded = True
                        yield visible
                    # Note: if only think-content was processed (no visible text),
                    # we do NOT set any_token_yielded so the on_chat_model_end fallback can still fire.

                continue

            # --- Fallback: emit content if Ollama returned it without streaming ---
            if kind == "on_chat_model_end":
                if not inside_think_block and think_buffer:
                    yield think_buffer
                    any_token_yielded = True
                think_buffer = ""
                inside_think_block = False

                data = event.get("data", {})
                output = data.get("output")
                if output:
                    # Track whether this turn requested a tool call
                    last_output_had_tool_calls = bool(
                        hasattr(output, "tool_calls") and output.tool_calls
                    )
                    raw_content = getattr(output, "content", None) or getattr(output, "text", "")
                    # Only emit via fallback if (a) no tokens were streamed AND (b) it's NOT a pure tool call turn
                    if not any_token_yielded and not last_output_had_tool_calls:
                        text = raw_content or ""
                        if text:
                            import re as _re
                            # Try to strip think blocks
                            text_stripped = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
                            
                            if not text_stripped and text.strip():
                                # The model ONLY returned a think block! It forgot to write the actual answer.
                                # Therefore, yield the contents of the think block minus the tags.
                                text = _re.sub(r'</?think>', '', text).strip()
                            else:
                                text = text_stripped

                        if text:
                            yield text


        # Yield metadata about the resolved model at the end
        resolved_model_id = model_id or (db_model.id if db_model else (model_name or "default"))
        yield {"model_id": resolved_model_id}

