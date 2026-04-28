"""Reactive MCP Tool — operates on ReactiveToolConfig and ReactiveMCPSource.

Identical interface to call_dynamic_mcp but queries the reactive tool
configuration tables. System-scoped (no user_id filter).
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.domain.shared.services.mcp_service import MCPService
from app.persistence.db import async_session_factory
from app.persistence.reactiva.repositories.reactive_tool_config_repository import ReactiveToolConfigRepository
from app.domain.reactiva.schemas.reactive_mcp_source import ReactiveMCPSource
from loguru import logger
import json

# Lazy-init singleton
_mcp_service: MCPService | None = None

# In-memory URL resolution cache for reactive sources
_reactive_url_cache: dict = {}


def _get_mcp_service() -> MCPService:
    global _mcp_service
    if _mcp_service is None:
        _mcp_service = MCPService()
    return _mcp_service


@tool
async def call_reactive_mcp(
    config: RunnableConfig,
    tool_config_name: str,
    key_values: dict = None,
    key_figures: list = None,
    arguments: dict = None,
) -> str:
    """
    Call any registered REACTIVE MCP tool or API endpoint for event diagnosis.
    Use this to get real-time sensor data, SCADA readings, or alarm system status.

    Input:
        - tool_config_name: The name of the reactive tool (e.g., 'get_sensor_data').
        - arguments: Standard parameters required by the API path/query.
        - key_values: Filter by categorical field values.
                      Format: {"FieldName": ["value1", "value2"]}
        - key_figures: Filter by numeric field ranges.
                       Format: [{"field": "FieldName", "min": X, "max": Y}]

    Rules:
        - STRICT FILTERING MANDATE: You MUST use filters to narrow down data for the affected equipment.
        - NEVER fetch the entire dataset without filtering unless explicitly needed.
    """
    if arguments is None:
        arguments = {}

    if key_values:
        arguments["key_values"] = key_values
    if key_figures:
        arguments["key_figures"] = key_figures

    logger.info(
        f"[ReactiveMCP] Calling: {tool_config_name} "
        f"filters: kv={bool(key_values)}, kf={bool(key_figures)}, args={arguments}"
    )

    provided_session = config.get("configurable", {}).get("session")
    if provided_session:
        return await _do_call_reactive_mcp(provided_session, tool_config_name, arguments)

    async with async_session_factory() as session:
        return await _do_call_reactive_mcp(session, tool_config_name, arguments)


async def _do_call_reactive_mcp(
    session,
    tool_config_name: str,
    arguments: dict = {},
) -> str:
    repo = ReactiveToolConfigRepository(session)
    tool_config = await repo.get_by_name(tool_config_name)

    if not tool_config:
        return json.dumps({"error": f"Reactive tool '{tool_config_name}' not found."})

    mcp_service = _get_mcp_service()

    config_data = tool_config.config or {}
    execution_url = config_data.get("url") or tool_config.api_url
    transport_type = config_data.get("transport", "mcp")
    method = config_data.get("method") or tool_config.method or "GET"

    parameter_schema = tool_config.parameter_schema or {}
    schema_hints = parameter_schema.get("response") or {}

    # Extract smart filters
    clean_arguments = arguments.copy()
    key_values_filter = clean_arguments.pop("key_values", None)
    key_figures_filter = clean_arguments.pop("key_figures", None)

    if key_values_filter:
        logger.info(f"[ReactiveMCP] Applying key_values filter: {key_values_filter}")
    if key_figures_filter:
        logger.info(f"[ReactiveMCP] Applying key_figures filter: {key_figures_filter}")

    # Robust URL resolution (with cache)
    if execution_url and "://" not in execution_url:
        cache_key = f"reactive:{tool_config.source_id}:{execution_url}"
        if cache_key in _reactive_url_cache:
            execution_url = _reactive_url_cache[cache_key]
        else:
            source = await session.get(ReactiveMCPSource, tool_config.source_id)
            if source and source.url:
                base_url = source.url.rstrip("/")
                path = execution_url.lstrip("/")
                execution_url = f"{base_url}/{path}"
                _reactive_url_cache[cache_key] = execution_url
                logger.info(f"[ReactiveMCP] Resolved URL: {execution_url}")

    if execution_url and "://" in execution_url:
        scheme, rest = execution_url.split("://", 1)
        while "//" in rest:
            rest = rest.replace("//", "/")
        execution_url = f"{scheme}://{rest}"

    # Heuristic: detect REST transport
    if transport_type == "mcp" and execution_url and "://" in execution_url:
        if any(domain in execution_url for domain in ["api.", "/api/"]):
            transport_type = "rest"

    logger.info(f"[ReactiveMCP] Executing {tool_config_name} via {transport_type} at {execution_url}")

    response = await mcp_service.execute_tool(
        base_url=execution_url,
        tool_name=tool_config_name,
        arguments=clean_arguments,
        is_stdio=(transport_type == "stdio"),
        transport_type=transport_type,
        method=method,
        schema_hints=schema_hints or None,
        key_values_filter=key_values_filter,
        key_figures_filter=key_figures_filter,
    )

    if response.error:
        return json.dumps({"error": f"Error from {tool_config_name}: {response.error}"})

    result = {
        "source": response.source,
        "key_figures": [
            {"name": kf.name, "value": kf.value, "unit": kf.unit}
            for kf in response.key_figures
        ],
        "key_values": [
            {"name": kv.name, "value": kv.value}
            for kv in response.key_values
        ],
    }

    if not response.key_figures and not response.key_values:
        result["warning"] = "No structured data could be extracted from the response."

    logger.info(
        f"[ReactiveMCP] Returning {len(result['key_figures'])} key figures "
        f"and {len(result['key_values'])} key values for {tool_config_name}"
    )
    return json.dumps(result, ensure_ascii=False)
