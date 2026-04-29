"""
Agent Service � Orchestrates Deep Agent invocations.

Encapsulates the logic of:
  1. Creating the LLM with connection resilience
  2. Building the Deep Agent
  3. Invoking the agent with the correct config
  4. Extracting the final response

This isolates the chat endpoint from agent internals (Dependency Inversion).
"""

import asyncio
from functools import lru_cache
import hashlib
import json
import re
import uuid
from typing import Any, Dict, List
from datetime import datetime, timezone
import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from deepagents.backends.utils import create_file_data
from loguru import logger

from app.core.config import settings
from app.core.llm import LLMFactory, LLMProvider
from app.domain.proactiva.agent.factory import create_industrial_agent
from app.domain.proactiva.agent.orchestrator import create_generalist_orchestrator
from app.domain.proactiva.agent.prompts import AGENTS_MD_CONTENT, TEMPORAL_ROUTER_PROMPT
from app.persistence.proactiva.repositories.model_repository import ModelRepository
from app.persistence.proactiva.vl_replay_buffer import vl_replay_buffer

_agent_cache: Dict[str, dict] = {}
_cache_lock = asyncio.Lock()
MAX_CACHE_SIZE = 100


async def _vllm_model_exists(base_url: str, model_name: str) -> bool:
    """
    Probe vLLM /v1/models to verify a model or LoRA adapter is loaded.

    LLMFactory.get_llm() never raises � it just creates a config object.
    The 404 only fires on the first actual request. This probe lets us check
    upfront and fall back to the base model when a LoRA hasn't been trained yet.
    """
    models_url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(models_url)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return any(m.get("id") == model_name for m in data)
    except Exception as exc:
        logger.debug(f"[AgentService] vLLM probe failed for '{model_name}': {exc}")
    return False


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

    async def _check_temporal_route(self, query: str, llm) -> bool:
        """Zero-shot router to determine if query is strictly historical (>6 months)."""
        prompt = TEMPORAL_ROUTER_PROMPT.format(
            current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            query=query
        )
        try:
            res = await llm.ainvoke(prompt)
            content = res.content.strip()
            think_match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if not content and think_match:
                content = think_match.group(1).strip()

            if content.startswith("```json") and len(content) > 10:
                content = content[7:].rstrip("`").strip()
            elif content.startswith("```") and len(content) > 6:
                content = content[3:].rstrip("`").strip()

            # Strip invisible Unicode chars (zero-width space, BOM, etc.) that .strip() misses
            content = ''.join(c for c in content if c.isprintable()).strip()
            if not content:
                return False

            # Extract JSON object via regex � handles cases where reasoning text surrounds the JSON
            # (e.g. when Qwen3.5 puts everything inside <think> and the fallback is raw reasoning)
            json_match = re.search(r'\{[^{}]*\}', content)
            if not json_match:
                return False
            data = json.loads(json_match.group(0))
            return bool(data.get("is_historical_only", False))
        except Exception as e:
            logger.warning(f"[TemporalRouter] Failed: {e}. Defaulting to non-historical.")
            return False

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

        # -- URL & Method ----------------------------------------------------
        effective_url = config_data.get("url") or t.api_url or "(unknown url)"
        effective_method = (config_data.get("method") or t.method or "GET").upper()
        transport = config_data.get("transport", "rest")

        # -- Path parameters (from {curly_braces} in the URL) ----------------
        path_params: list[str] = re.findall(r'\{(.*?)\}', effective_url)

        # -- Parameter schema breakdown ---------------------------------------
        schema: dict = t.parameter_schema or {}
        properties: dict = schema.get("properties", {})
        required_fields: list = schema.get("required", [])
        response_fields: dict = schema.get("response", {})  # optional hint
        filterable_schema: dict = schema.get("filterable_schema", {})  # new: real discoverable fields

        lines = [
            f"??? TOOL: {t.name} ???",
            f"  Description : {t.description}",
            f"  Transport   : {transport.upper()}",
            f"  Method      : {effective_method}",
            f"  URL pattern : {effective_url}",
        ]

        # -- Path params summary ----------------------------------------------
        if path_params:
            lines.append(f"  Path params : {', '.join(path_params)}  ? required, embedded in the URL")

        # -- Input parameters -------------------------------------------------
        if properties:
            lines.append("  Parameters:")
            for param_name, param_def in properties.items():
                p_type = param_def.get("type", "any")
                p_desc = param_def.get("description", "")
                p_enum = param_def.get("enum")
                p_default = param_def.get("default")
                is_required = "[req]" if (param_name in required_fields or param_name in path_params) else "[opt]"
                p_desc_short = (p_desc[:60] + "..") if len(p_desc) > 60 else p_desc
                lines.append(f"    - {param_name} ({p_type}) {is_required}: {p_desc_short}")
        elif path_params:
            # Fallback when schema is empty but path params exist
            lines.append("  Parameters:")
            for pp in path_params:
                lines.append(f"    - {pp} (string) [required]: Path parameter for the URL")
        else:
            placement = "body" if effective_method in ("POST", "PUT", "PATCH") else "query string"
            lines.append(f"  Parameters  : none � send an empty dict {{}} as arguments ({placement})")

        # -- Response hints ---------------------------------------------------
        if response_fields:
            lines.append("  Expected response fields:")
            for field, field_def in response_fields.items():
                f_type = field_def.get("type", "any") if isinstance(field_def, dict) else str(field_def)
                f_unit = field_def.get("unit", "") if isinstance(field_def, dict) else ""
                f_desc = field_def.get("description", "") if isinstance(field_def, dict) else ""
                unit_str = f" [{f_unit}]" if f_unit else ""
                desc_str = f" � {f_desc}" if f_desc else ""
                lines.append(f"    - {field} ({f_type}){unit_str}{desc_str}")

        # -- Filterable fields (injected from discovered schema) --------------
        kf_fields: list = filterable_schema.get("key_figures", [])
        kv_fields: dict = filterable_schema.get("key_values", {})

        if kf_fields or kv_fields:
            lines.append("  -- FILTERABLE FIELDS (pass as DIRECT parameters to call_dynamic_mcp) --")
            if kf_fields:
                lines.append(f"    [NUMERIC] key_figures fields: {', '.join(kf_fields)}")
            if kv_fields:
                lines.append("    [CATEGORICAL] key_values fields and known values:")
                for kv_field, kv_vals in kv_fields.items():
                    vals_preview = kv_vals[:15]
                    suffix = f" ... (+{len(kv_vals) - 15} more)" if len(kv_vals) > 15 else ""
                    lines.append(f"      � {kv_field}: {vals_preview}{suffix}")

            # -- Dynamic few-shot examples (auto-generated from real discovered data) --
            lines.append("")
            lines.append(f"  -- EXAMPLE CALLS for '{t.name}' --")

            # Auto-generate sample arguments if the API has required fields
            sample_args_str = ""
            if required_fields or path_params:
                sample_args = {}
                for pp in path_params:
                    sample_args[pp] = f"{pp}_value"
                for req in required_fields:
                    if req not in sample_args:
                        sample_args[req] = f"{req}_value"
                if sample_args:
                    import json
                    sample_args_str = f', arguments={json.dumps(sample_args)}'

            # Example 1: Categorical filter � pick the first kv field and its first value
            if kv_fields:
                ex_kv_field = next(iter(kv_fields))
                ex_kv_value = kv_fields[ex_kv_field][0] if kv_fields[ex_kv_field] else "example"
                lines.append(
                    f'    If user asks about a specific {ex_kv_field} (e.g. "{ex_kv_value}"):'
                )
                lines.append(
                    f'    ? call_dynamic_mcp(tool_config_name="{t.name}"{sample_args_str}, '
                    f'key_values={{"{ex_kv_field}": ["{ex_kv_value}"]}})'
                )

            # Example 2: Numeric range filter � pick the first kf field
            if kf_fields:
                ex_kf_field = kf_fields[0]
                lines.append(
                    f'    If user asks for {ex_kf_field} above a threshold:'
                )
                lines.append(
                    f'    ? call_dynamic_mcp(tool_config_name="{t.name}"{sample_args_str}, '
                    f'key_figures=[{{"field": "{ex_kf_field}", "min": 100}}])'
                )

            # Example 3: Combined filter � only if both types exist
            if kv_fields and kf_fields:
                ex_kv_field = next(iter(kv_fields))
                ex_kv_value = kv_fields[ex_kv_field][0] if kv_fields[ex_kv_field] else "example"
                ex_kf_field = kf_fields[0]
                lines.append(
                    f'    Combined (specific {ex_kv_field} + {ex_kf_field} range):'
                )
                lines.append(
                    f'    ? call_dynamic_mcp(tool_config_name="{t.name}"{sample_args_str}, '
                    f'key_values={{"{ex_kv_field}": ["{ex_kv_value}"]}}, '
                    f'key_figures=[{{"field": "{ex_kf_field}", "min": 50, "max": 200}}])'
                )
            elif not kf_fields and not kv_fields:
                # Example 4: No filters, only arguments
                lines.append(
                    '    Simple call with parameters:'
                )
                lines.append(
                    f'    ? call_dynamic_mcp(tool_config_name="{t.name}"{sample_args_str})'
                )

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

    async def _prepare_agent(
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
        use_generalist: bool = False,
        mode: str = "chat",
        mode_context: dict | None = None,
    ) -> dict:
        """
        Shared preparation logic for invoke() and stream().

        Returns a context dict with:
          agent, config, query (interpolated), params, all_tools, resolved_model_id
        """
        from types import SimpleNamespace
        from app.persistence.proactiva.repositories.tool_config_repository import ToolConfigRepository

        # 1. Resolve Provider and Model Name from DB
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

        if "stop_sequence" in merged_params:
            stop_val = merged_params.pop("stop_sequence")
            if stop_val:
                merged_params["stop"] = [stop_val]

        # 3. Create Orchestrator LLM
        # Cap output tokens: orchestrator routes/synthesizes only � prevents context overflow
        merged_params.setdefault("max_tokens", 1024)
        ui_generalist_llm = await LLMFactory.get_llm(
            provider=provider,
            model_name=model_name,
            session=session,
            **merged_params
        )

        # 4. Apply resilience settings
        if hasattr(ui_generalist_llm, "max_retries"):
            ui_generalist_llm.max_retries = settings.llm_max_retries
        if hasattr(ui_generalist_llm, "request_timeout"):
            ui_generalist_llm.request_timeout = settings.llm_request_timeout

        # 4.5 Temporal Router (reusing the same generalist instance)
        is_historical = await self._check_temporal_route(query, ui_generalist_llm)
        if is_historical:
            logger.info("ROUTER: Query is purely historical! Bypassing MCP/RAG tools to save tokens.")
            mcp_source_id = "none"
            knowledge_base_id = "none"

        # 4.6 Expert Loader: Deferred � captured via default arg to avoid closure over
        # a mutable session reference (the session may be closed before the lambda fires).
        
        # Determine LoRA identifiers from settings (single source of truth)
        expert_lora_target = settings.system1_historico_model   # e.g. "aura_tenant_01-v2"
        vl_lora_target = settings.system1_model                 # e.g. "aura_tenant_01-vl"

        _captured_session = session
        # Factory con fallback: verifica en vLLM si el LoRA existe antes de usarlo.
        # LLMFactory.get_llm() nunca lanza excepci�n (solo crea config) � el 404 ocurre
        # en la primera llamada real, por eso necesitamos probar antes.
        async def expert_llm_factory(sess=_captured_session):
            if settings.system1_force_base_model:
                logger.info(
                    f"[AgentService] system1_force_base_model=True: Using base model "
                    f"'{settings.system1_base_model}' directly (skipping LoRA '{expert_lora_target}')"
                )
                return await LLMFactory.get_llm(
                    provider=LLMProvider.VLLM,
                    model_name=settings.system1_base_model,
                    session=sess,
                    base_url=settings.vllm_base_url,
                )
            lora_ready = await _vllm_model_exists(settings.vllm_base_url, expert_lora_target)
            if lora_ready:
                return await LLMFactory.get_llm(
                    provider=LLMProvider.VLLM,
                    model_name=expert_lora_target,
                    session=sess,
                    base_url=settings.vllm_base_url,
                )
            logger.warning(
                f"[AgentService] LoRA '{expert_lora_target}' not loaded in vLLM � "
                f"falling back to base model '{settings.system1_base_model}'. "
                f"Train and deploy a LoRA via ApiLLMOps to activate the expert subagent."
            )
            return await LLMFactory.get_llm(
                provider=LLMProvider.VLLM,
                model_name=settings.system1_base_model,
                session=sess,
                base_url=settings.vllm_base_url,
            )

        # 4.7 Worker LLM: reuse generalist instance
        worker_llm = ui_generalist_llm

        # 4.8 Vision LLM � Sistema 1 VL (fine-tuned VL LoRA or base model fallback)
        # Gemma 4 26B-A4B is natively multimodal � if the VL LoRA doesn't exist yet,
        # fall back to the base model so sistema1-vl (computer use loop) still works.
        # Gemma 4 recommended params: temp=1.0 (Google docs), top_p=0.95
        vision_llm = None
        _vision_kwargs = {
            "temperature": 1.0,      # Gemma 4 recommended: temp=1.0 (Google official)
            "max_tokens": 4096,      # Needed: thinking block (~1500t) + tool call (~500t) + margin
            "streaming": False,      # Ensure complete responses for tool calls
            "stop": [],              # Disable stop tokens: VL needs to generate full tool calls
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "base_url": settings.vllm_base_url,
        }
        if settings.system1_enabled:
            if settings.system1_force_base_model:
                vision_llm = await LLMFactory.get_llm(
                    provider=LLMProvider.VLLM,
                    model_name=settings.system1_base_model,
                    session=session,
                    **_vision_kwargs,
                )
                logger.info(
                    f"[AgentService] Sistema 1 VL: base model direct "
                    f"(system1_force_base_model=True): '{settings.system1_base_model}'"
                )
            else:
                vl_ready = await _vllm_model_exists(settings.vllm_base_url, vl_lora_target)
                if vl_ready:
                    vision_llm = await LLMFactory.get_llm(
                        provider=LLMProvider.VLLM,
                        model_name=vl_lora_target,
                        session=session,
                        **_vision_kwargs,
                    )
                    logger.info(f"[AgentService] Sistema 1 VL LoRA loaded: {vl_lora_target}")
                else:
                    vision_llm = await LLMFactory.get_llm(
                        provider=LLMProvider.VLLM,
                        model_name=settings.system1_base_model,
                        session=session,
                        **_vision_kwargs,
                    )
                    logger.warning(
                        f"[AgentService] VL LoRA '{vl_lora_target}' not loaded in vLLM \u2014 "
                        f"using base model '{settings.system1_base_model}' for vision/computer-use."
                    )
        # 4.9 Expert LLM instance � Sistema 1 Hist�rico (fine-tuned text LoRA, ZERO tools)
        # Resuelto como instancia directa (no factory) para que sistema1-historico
        # pueda ser ensamblado en tiempo de build del grafo.
        # FALLBACK: Si el LoRA no existe, usa el modelo base para mantener funcionalidad.
        expert_model_instance = None
        if settings.system1_enabled:
            # expert_llm_factory already probes vLLM and falls back to base model if LoRA missing
            expert_model_instance = await expert_llm_factory()

        # 5. System prompt composition
        if params and not params.system_prompt and db_model and db_model.system_prompt:
            params.system_prompt = db_model.system_prompt
        elif not params and db_model and db_model.system_prompt:
            params = SimpleNamespace(system_prompt=db_model.system_prompt)

        # 6. Build tool context
        tool_repo = ToolConfigRepository(session)
        if mcp_source_id == "none":
            all_tools = []
        elif mcp_source_id:
            all_tools = await tool_repo.get_by_source(uuid.UUID(mcp_source_id), user_id=uuid.UUID(user_id))
        else:
            all_tools = await tool_repo.get_all_by_user(uuid.UUID(user_id))

        dynamic_tools_list = [self._build_tool_context(t) for t in all_tools]
        tools_context = "\n\n".join(dynamic_tools_list) if dynamic_tools_list else "No dynamic tools currently registered."
        custom_prompt = db_model.system_prompt if db_model else None

        # --- Build Cache Key (includes vision availability to avoid stale graphs) ---
        def _stable_hash(s: str) -> str:
            return hashlib.sha256(s.encode()).hexdigest()[:16]

        _vision_available = vision_llm is not None
        _expert_available = expert_model_instance is not None
        cache_key = (
            f"{user_id}_{model_id}_{use_generalist}_{knowledge_base_id}_"
            f"{mcp_source_id}_{_vision_available}_{_expert_available}_"
            f"{_stable_hash(tools_context)}_{_stable_hash(str(custom_prompt))}"
        )

        async def _build_agent_async():
            if use_generalist:
                logger.info("[AgentService] Assembling Generalist Orchestrator...")
                resolved_expert_llm = await expert_llm_factory()
                return create_generalist_orchestrator(
                    generalist_model=ui_generalist_llm,
                    expert_model=resolved_expert_llm,
                    expert_model_instance=expert_model_instance,
                    vision_model=vision_llm,
                    worker_model=worker_llm,
                    checkpointer=checkpointer,
                    store=store,
                    mcp_tools_context=tools_context,
                    enable_knowledge=(knowledge_base_id != "none"),
                    enable_mcp=(mcp_source_id != "none"),
                    enable_system1=settings.system1_enabled,
                    enable_computer_use=settings.computer_use_enabled,
                    vl_replay_buffer=vl_replay_buffer,
                )
            else:
                expert_llm = await expert_llm_factory()
                return create_industrial_agent(
                    model=expert_llm, worker_model=worker_llm,
                    checkpointer=checkpointer, store=store,
                    custom_system_prompt=custom_prompt,
                    mcp_tools_context=tools_context,
                    enable_knowledge=(knowledge_base_id != "none"),
                    enable_mcp=(mcp_source_id != "none"),
                )

        async with _cache_lock:
            if cache_key in _agent_cache:
                entry = _agent_cache.pop(cache_key)
                _agent_cache[cache_key] = entry
                agent = entry["agent"]
            else:
                agent = await _build_agent_async()
                if len(_agent_cache) >= MAX_CACHE_SIZE:
                    first_key = next(iter(_agent_cache))
                    del _agent_cache[first_key]
                _agent_cache[cache_key] = {"agent": agent}

        # Inject mode-specific context into query for event mode
        if mode == "event" and mode_context:
            event_ctx = (
                f"[REACTIVE EVENT CONTEXT]\n"
                f"Event ID: {mode_context.get('event_id', 'N/A')}\n"
                f"Source: {mode_context.get('source_type', 'N/A')}\n"
                f"Severity: {mode_context.get('severity', 'N/A')}\n"
                f"Title: {mode_context.get('title', 'N/A')}\n"
                f"Description: {mode_context.get('description', 'N/A')}\n"
                f"Payload: {mode_context.get('payload', 'N/A')}\n"
                f"---\n\n"
            )
            query = event_ctx + query

        # Interpolate variables
        query = self._interpolate_variables(query, all_tools)
        if params and hasattr(params, "system_prompt") and params.system_prompt:
            params.system_prompt = self._interpolate_variables(params.system_prompt, all_tools)

        resolved_model_id = model_id or (db_model.id if db_model else (model_name or "default"))

        return {
            "agent": agent,
            "config": {
                "recursion_limit": 100,
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "knowledge_base_id": knowledge_base_id,
                    "session": session,
                }
            },
            "query": query,
            "params": params,
            "all_tools": all_tools,
            "resolved_model_id": resolved_model_id,
        }

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
        use_generalist: bool = False,
        mode: str = "chat",
        mode_context: dict | None = None,
    ) -> tuple[str, str]:
        """
        Invoke the Deep Agent and return the assistant's response text.
        """
        ctx = await self._prepare_agent(
            user_id=user_id,
            thread_id=thread_id,
            query=query,
            knowledge_base_id=knowledge_base_id,
            mcp_source_id=mcp_source_id,
            session=session,
            checkpointer=checkpointer,
            store=store,
            params=params,
            model_id=model_id,
            use_generalist=use_generalist,
            mode=mode,
            mode_context=mode_context,
        )

        logger.info(f"Invoking agent for thread {thread_id} mode={mode} with query: {ctx['query'][:200]}...")
        response = await ctx["agent"].ainvoke(
            {
                "messages": self._build_messages(ctx["query"], ctx["params"]),
                "files": {"/AGENTS.md": create_file_data(AGENTS_MD_CONTENT)},
            },
            config=ctx["config"],
        )

        last_message = response["messages"][-1]
        return last_message.content, ctx["resolved_model_id"]

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
        use_generalist: bool = False,
        mode: str = "chat",
        mode_context: dict | None = None,
    ):
        """
        Stream the Deep Agent response, yielding text chunks as they arrive,
        plus a final metadata chunk with model info.
        Uses LangGraph's astream_events v2 API.
        """
        ctx = await self._prepare_agent(
            user_id=user_id,
            thread_id=thread_id,
            query=query,
            knowledge_base_id=knowledge_base_id,
            mcp_source_id=mcp_source_id,
            session=session,
            checkpointer=checkpointer,
            store=store,
            params=params,
            model_id=model_id,
            use_generalist=use_generalist,
            mode=mode,
            mode_context=mode_context,
        )
        agent = ctx["agent"]
        config = ctx["config"]
        query = ctx["query"]
        params = ctx["params"]

        # State for filtering <think>...</think> tokens emitted by reasoning models
        any_token_yielded = False
        last_output_had_tool_calls = False
        inside_think_block = False
        think_buffer = ""
        # Buffer tokens per model step; flush only when we confirm no tool calls
        step_text_buffer: list = []
        # Nesting depth: >0 means we are inside a subagent tool call.
        # Only stream tokens when depth == 0 (top-level orchestrator).
        subagent_depth = 0

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

            # --- Emit Subagent (Tool) running status ---
            if kind == "on_tool_start":
                subagent_depth += 1
                input_data = event.get("data", {}).get("input", {})
                # deepagents wraps all subagents under a single "task" tool.
                # Extract the real subagent name from subagent_type when available.
                effective_name = input_data.get("subagent_type", name) if isinstance(input_data, dict) else name
                yield {"type": "subagent", "status": "running", "name": effective_name, "input": input_data}

            if kind == "on_tool_end":
                subagent_depth = max(0, subagent_depth - 1)
                yield {"type": "subagent", "status": "complete", "name": name}

            if kind == "on_tool_error":
                subagent_depth = max(0, subagent_depth - 1)
                yield {"type": "subagent", "status": "error", "name": name}

            # Reset per-model-turn counters when a new top-level LLM call starts.
            # Gated on depth==0 to avoid subagent LLM turns clobbering orchestrator state.
            if kind == "on_chat_model_start" and subagent_depth == 0:
                any_token_yielded = False
                last_output_had_tool_calls = False
                inside_think_block = False
                think_buffer = ""
                step_text_buffer = []

            # --- Live Screen Viewer: forward screenshot events from computer_use observe node ---
            if kind == "on_custom_event" and name == "screenshot":
                yield {"type": "screenshot", "data": event.get("data", {})}

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
                                # No <think> start found.
                                # Also check for orphan </think> � Qwen3/vLLM sometimes
                                # streams thinking content WITHOUT emitting the opening
                                # <think> tag, but DOES emit the closing </think>.
                                close_idx = think_buffer.find("</think>")
                                if close_idx != -1:
                                    # Discard everything up to and including the orphan </think>.
                                    # Also retroactively discard step_text_buffer: content already
                                    # accumulated before this chunk was pre-think reasoning text.
                                    think_buffer = think_buffer[close_idx + len("</think>"):]
                                    step_text_buffer = []
                                    # Continue loop � there may be real content after it
                                else:
                                    # No think tag found at all.
                                    # Keep last 7 chars (partial of either <think> or </think>).
                                    if len(think_buffer) > 7:
                                        visible += think_buffer[:-7]
                                        think_buffer = think_buffer[-7:]
                                    break

                    if visible and subagent_depth == 0:
                        step_text_buffer.append(visible)
                    # Tokens are buffered and flushed at on_chat_model_end only if no tool calls.

                continue

            # --- Flush buffered tokens / fallback for non-streaming models ---
            # Only process end events from the top-level orchestrator.
            if kind == "on_chat_model_end" and subagent_depth == 0:
                # Flush any remaining think_buffer into the step buffer
                if not inside_think_block and think_buffer:
                    step_text_buffer.append(think_buffer)
                think_buffer = ""
                inside_think_block = False

                data = event.get("data", {})
                output = data.get("output")
                if output:
                    last_output_had_tool_calls = bool(
                        hasattr(output, "tool_calls") and output.tool_calls
                    )
                    # Fallback for non-streaming models: if no tokens were buffered
                    # and this is not a tool-call turn, pull from output.content
                    if not step_text_buffer and not last_output_had_tool_calls:
                        raw_content = getattr(output, "content", None) or getattr(output, "text", "")
                        text = raw_content or ""
                        if text:
                            import re as _re
                            text_stripped = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
                            if not text_stripped and text.strip():
                                text = _re.sub(r'</?think>', '', text).strip()
                            else:
                                text = text_stripped
                        if text:
                            step_text_buffer.append(text)

                # Only emit buffered tokens if this step had NO tool calls.
                # Suppresses orchestrator pre-delegation reasoning (e.g. "I need to call sistema1-vl...").
                if not last_output_had_tool_calls and step_text_buffer:
                    for chunk in step_text_buffer:
                        yield chunk
                    any_token_yielded = True
                step_text_buffer = []


        yield {"model_id": ctx["resolved_model_id"]}

