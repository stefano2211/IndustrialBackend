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
        - tool_config_name: The name of the tool as registered in the system (e.g., 'plant-sensor-api').
        - arguments: A dictionary of parameters for the call (filters, IDs, etc.).
    Returns structured JSON with key_figures (metrics) and key_values (info).
    The orchestrator uses this data to compose the final answer.
    """
    logger.info(f"[MCP Tool] Calling dynamic tool: {tool_config_name} with args: {arguments}")
    
    async with async_session_factory() as session:
        repo = ToolConfigRepository(session)
        tool_config = await repo.get_by_name(tool_config_name)
        
        if not tool_config:
            return json.dumps({"error": f"Tool configuration '{tool_config_name}' not found."})

        mcp_service = _get_mcp_service()
        
        # Determine transport type and execution parameters
        config_data = tool_config.config or {}
        
        # Priority: explicit config["url"] > tool_config.api_url
        execution_url = config_data.get("url") or tool_config.api_url
        transport_type = config_data.get("transport", "mcp")
        method = config_data.get("method") or tool_config.method or "GET"
        
        # Extract schema hints for deterministic response filtering
        parameter_schema = tool_config.parameter_schema or {}
        schema_hints = parameter_schema.get("response") or {}

        # --- ROBUST URL RESOLUTION ---
        # If the execution_url is relative, we join it with the base source URL
        if execution_url and "://" not in execution_url:
            source = await session.get(MCPSource, tool_config.source_id)
            if source and source.url:
                base_url = source.url.rstrip("/")
                path = execution_url.lstrip("/")
                execution_url = f"{base_url}/{path}"
                logger.info(f"[MCP Tool] Resolved relative URL to: {execution_url}")

        # Normalize double slashes in path (e.g. https://host//path → https://host/path)
        if execution_url and "://" in execution_url:
            scheme, rest = execution_url.split("://", 1)
            while "//" in rest:
                rest = rest.replace("//", "/")
            execution_url = f"{scheme}://{rest}"

        # Heuristic: Detect if transport type is 'mcp' but it's clearly a REST API (like PokeAPI)
        if transport_type == "mcp" and execution_url and "://" in execution_url:
            if any(domain in execution_url for domain in ["pokeapi.co", "api.", "/api/"]):
                logger.info(f"[MCP Tool] Heuristic detected REST transport for {execution_url}")
                transport_type = "rest"

        logger.info(f"[MCP Tool] Executing {tool_config_name} via {transport_type} at {execution_url}")
        
        response = await mcp_service.execute_tool(
            base_url=execution_url,
            tool_name=tool_config_name,
            arguments=arguments,
            is_stdio=(transport_type == "stdio"),
            transport_type=transport_type,
            method=method,
            schema_hints=schema_hints or None,
        )

        if response.error:
            return json.dumps({"error": f"Error from {tool_config_name}: {response.error}"})

        # --- Return structured JSON to the orchestrator ---
        # The orchestrator (main LLM) is responsible for presenting this data to the user.
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
