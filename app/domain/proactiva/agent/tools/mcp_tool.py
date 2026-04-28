from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.domain.shared.services.mcp_service import MCPService
from app.persistence.db import async_session_factory
from app.persistence.proactiva.repositories.tool_config_repository import ToolConfigRepository
from app.domain.proactiva.schemas.mcp_source import MCPSource
from loguru import logger
import json

# Lazy-init singleton
_mcp_service: MCPService | None = None

# In-memory URL resolution cache — evita un DB round-trip por cada llamada MCP.
# Las URLs de source solo cambian si se reconfigura la fuente (muy infrecuente).
_url_cache: dict = {}

def _get_mcp_service() -> MCPService:
    global _mcp_service
    if _mcp_service is None:
        _mcp_service = MCPService()
    return _mcp_service

@tool
async def call_dynamic_mcp(
    config: RunnableConfig,
    tool_config_name: str,
    key_values: dict = None,
    key_figures: list = None,
    arguments: dict = None,
) -> str:
    """
    Call any registered MCP tool or API endpoint dynamically with STRICT precision.
    Use this to get real-time data, sensor readings, or perform external actions.

    Input:
        - tool_config_name: The name of the tool as registered in the system (e.g., 'get_maquinaria').
        - arguments: Any other standard parameters required by the API path/query.
        - key_values: (REQUIRED if user asks for specific items) Filter by categorical field values.
                      Format: {"FieldName": ["value1", "value2"]}
        - key_figures: (REQUIRED if user asks for ranges) Filter by numeric field ranges.
                       Format: [{"field": "FieldName", "min": X, "max": Y}]

    Rules:
        - STRICT FILTERING MANDATE: You MUST use `key_values` or `key_figures` filters to narrow down the data.
        - NEVER fetch the entire dataset lazily without filtering unless the user explicitly demands "all records without exception".
        - ALWAYS extract the exact field names provided in 'Filterable fields' under the tool's context. If the user asks for "Motor 1", you MUST provide a `key_values` filter matching that name.

    Returns structured JSON with key_figures (metrics) and key_values (info).
    """
    if arguments is None:
        arguments = {}

    # Pack the explicitly named filters back into the arguments dict
    if key_values:
        arguments["key_values"] = key_values
    if key_figures:
        arguments["key_figures"] = key_figures

    logger.info(f"[MCP Tool] Calling dynamic tool: {tool_config_name} with filters: kv={bool(key_values)}, kf={bool(key_figures)}, args={arguments}")

    provided_session = config.get("configurable", {}).get("session")
    if provided_session:
        return await _do_call_dynamic_mcp(provided_session, tool_config_name, arguments)
    
    async with async_session_factory() as session:
        return await _do_call_dynamic_mcp(session, tool_config_name, arguments)

async def _do_call_dynamic_mcp(
    session,
    tool_config_name: str,
    arguments: dict = {},
) -> str:
    repo = ToolConfigRepository(session)
    tool_config = await repo.get_by_name(tool_config_name)

    if not tool_config:
        return json.dumps({"error": f"Tool configuration '{tool_config_name}' not found."})

    mcp_service = _get_mcp_service()

    config_data = tool_config.config or {}
    execution_url = config_data.get("url") or tool_config.api_url
    transport_type = config_data.get("transport", "mcp")
    method = config_data.get("method") or tool_config.method or "GET"

    parameter_schema = tool_config.parameter_schema or {}
    schema_hints = parameter_schema.get("response") or {}

    # -- Extract smart filters from arguments (pop before sending to API) --
    clean_arguments = arguments.copy()
    key_values_filter = clean_arguments.pop("key_values", None)
    key_figures_filter = clean_arguments.pop("key_figures", None)

    if key_values_filter:
        logger.info(f"[MCP Tool] Applying key_values filter: {key_values_filter}")
    if key_figures_filter:
        logger.info(f"[MCP Tool] Applying key_figures filter: {key_figures_filter}")

    # -- Robust URL resolution (with in-memory cache) -----------------------
    if execution_url and "://" not in execution_url:
        cache_key = f"{tool_config.source_id}:{execution_url}"
        if cache_key in _url_cache:
            execution_url = _url_cache[cache_key]
            logger.debug(f"[MCP Tool] URL resolved from cache: {execution_url}")
        else:
            source = await session.get(MCPSource, tool_config.source_id)
            if source and source.url:
                base_url = source.url.rstrip("/")
                path = execution_url.lstrip("/")
                execution_url = f"{base_url}/{path}"
                _url_cache[cache_key] = execution_url
                logger.info(f"[MCP Tool] Resolved relative URL to: {execution_url}")

    if execution_url and "://" in execution_url:
        scheme, rest = execution_url.split("://", 1)
        while "//" in rest:
            rest = rest.replace("//", "/")
        execution_url = f"{scheme}://{rest}"

    # Heuristic: Detect REST transport
    if transport_type == "mcp" and execution_url and "://" in execution_url:
        if any(domain in execution_url for domain in ["pokeapi.co", "api.", "/api/"]):
            logger.info(f"[MCP Tool] Heuristic detected REST transport for {execution_url}")
            transport_type = "rest"

    logger.info(f"[MCP Tool] Executing {tool_config_name} via {transport_type} at {execution_url}")

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
        f"[MCP Tool] Returning {len(result['key_figures'])} key figures "
        f"and {len(result['key_values'])} key values for {tool_config_name}"
    )
    return json.dumps(result, ensure_ascii=False)
