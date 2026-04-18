from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.domain.services.mcp_service import MCPService
from app.persistence.db import async_session_factory
from app.persistence.repositories.tool_config_repository import ToolConfigRepository
from app.domain.schemas.tool_config import ToolConfig
from app.domain.schemas.mcp_source import MCPSource
from loguru import logger
import json

# Lazy-init singleton
_mcp_service: MCPService | None = None

def _get_mcp_service() -> MCPService:
    global _mcp_service
    if _mcp_service is None:
        _mcp_service = MCPService()
    return _mcp_service

@tool
async def call_dynamic_mcp(
    config: RunnableConfig,
    tool_config_name: str,
    arguments: dict = {},
) -> str:
    """
    Call any registered MCP tool or API endpoint dynamically.
    Use this to get real-time data, sensor readings, or perform external actions.

    Input:
        - tool_config_name: The name of the tool as registered in the system (e.g., 'get_maquinaria').
        - arguments: A dictionary of parameters for the call (filters, IDs, etc.).

    SMART FILTERING — you can include these optional keys in 'arguments' to reduce
    the data returned to only what the user needs (saves tokens and improves accuracy):

        key_values  : Filter by categorical field values.
                      Format: {"FieldName": ["value1", "value2"]}
                      Example: {"Category": ["Thermal"]} — returns only thermal equipment.
                      Example: {"Status": ["Maintenance_Req"]} — equipment needing maintenance.

        key_figures : Filter by numeric field ranges.
                      Format: [{"field": "FieldName", "min": X, "max": Y}]
                      Example: [{"field": "Value", "min": 1000}] — values above 1000.
                      Example: [{"field": "Value", "min": 0, "max": 50}] — values in 0-50 range.

    Rules:
        - If the user asks for ALL data, omit key_values and key_figures entirely.
        - If the user specifies a category, status, or name → use key_values.
        - If the user specifies a numeric threshold or range → use key_figures.
        - You can combine both filters in a single call.
        - CRITICAL: You MUST use the exact field names provided in 'Filterable fields' under the tool's context (e.g., 'Value', 'TagName'). DO NOT guess or invent field names.
        - CRITICAL TOKEN LIMIT: If your filter returns empty results ("No structured data could be extracted"), DO NOT call the tool again without filters. Calling a tool without filters just to read raw JSON will consume too many tokens and crash the system. Instead, just inform the user that no items matched their filter criteria.

    Returns structured JSON with key_figures (metrics) and key_values (info).
    """
    logger.info(f"[MCP Tool] Calling dynamic tool: {tool_config_name} with args: {arguments}")

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

    # ── Extract smart filters from arguments (pop before sending to API) ──
    clean_arguments = arguments.copy()
    key_values_filter = clean_arguments.pop("key_values", None)
    key_figures_filter = clean_arguments.pop("key_figures", None)

    if key_values_filter:
        logger.info(f"[MCP Tool] Applying key_values filter: {key_values_filter}")
    if key_figures_filter:
        logger.info(f"[MCP Tool] Applying key_figures filter: {key_figures_filter}")

    # ── Robust URL resolution ──────────────────────────────────────────────
    if execution_url and "://" not in execution_url:
        source = await session.get(MCPSource, tool_config.source_id)
        if source and source.url:
            base_url = source.url.rstrip("/")
            path = execution_url.lstrip("/")
            execution_url = f"{base_url}/{path}"
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
