import json
import httpx
import re
from typing import Any, Dict, List, Optional
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from app.domain.schemas.mcp import MCPResponse, KeyFigure, KeyValue
from app.core.llm import LLMFactory

class MCPService:
    """
    Service for dynamic MCP tool discovery and execution.
    Implements the 'Zero-Config' pattern.
    """

    def __init__(self):
        pass

    async def _auto_map_response(self, source: str, data: Any) -> MCPResponse:
        """
        Recursively maps a JSON payload to KeyFigures and KeyValues.
        Numerical Fields -> KeyFigure
        Strings/Bools -> KeyValue
        """
        response = MCPResponse(source=source)
        
        if not isinstance(data, dict):
            response.key_values.append(KeyValue(name="raw_data", value=str(data)))
            return response

        def process_dict(d: dict, prefix: str = ""):
            for key, val in d.items():
                name = f"{prefix}{key}"
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    response.key_figures.append(KeyFigure(name=name, value=float(val)))
                elif isinstance(val, dict):
                    process_dict(val, f"{name}.")
                elif isinstance(val, list):
                    response.key_values.append(KeyValue(name=name, value=json.dumps(val)))
                else:
                    response.key_values.append(KeyValue(name=name, value=val))

        process_dict(data)
        return response

    async def execute_tool(
        self, 
        base_url: str, 
        tool_name: str, 
        arguments: Dict[str, Any],
        is_stdio: bool = False,
        transport_type: str = "mcp",
        method: str = "GET"
    ) -> MCPResponse:
        """
        Dynamically connects to an MCP server, calls a tool, and maps the result.
        """
        logger.info(f"[MCP Service] Executing tool '{tool_name}' ({transport_type}) on {base_url} using {method}")
        
        try:
            if transport_type == "rest":
                # Universal REST Bridge execution
                async with httpx.AsyncClient() as client:
                    method = method.upper()
                    target_url = base_url
                    
                    # 1. URL Parameter Substitution
                    # We find all {param} in the URL and replace them with values from arguments
                    url_params = re.findall(r'\{(.*?)\}', target_url)
                    remaining_args = arguments.copy()
                    
                    for p in url_params:
                        if p in remaining_args:
                            val = str(remaining_args.pop(p))
                            target_url = target_url.replace(f"{{{p}}}", val)
                        else:
                            logger.warning(f"[MCP Service] Missing URL parameter '{p}' for {base_url}")
                    
                    logger.info(f"[MCP Service] Final target URL: {target_url}")

                    if method == "GET":
                        # For GET, we pass remaining arguments as query parameters
                        response = await client.get(target_url, params=remaining_args, timeout=30.0)
                    else:
                        # For POST/PUT, we pass remaining arguments as JSON body
                        response = await client.request(method, target_url, json=remaining_args, timeout=30.0)
                    
                    response.raise_for_status()
                    return await self._auto_map_response(tool_name, response.json())

            if is_stdio:
                # Local stdio transport
                server_params = StdioServerParameters(command="python", args=[base_url])
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        return await self._auto_map_response(tool_name, result.content)
            else:
                # Remote SSE transport
                async with sse_client(base_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        # We extract text content and try to parse it as JSON if it looks like it
                        content = result.content[0].text if result.content else "{}"
                        try:
                            data = json.loads(content)
                        except:
                            data = {"text": content}
                        return await self._auto_map_response(tool_name, data)
        except Exception as e:
            logger.error(f"[MCP Service] Execution failed: {str(e)}")
            return MCPResponse(source=tool_name, error=str(e))

    async def discover_tools(self, base_url: str, is_stdio: bool = False, is_resource: bool = False) -> List[Dict[str, Any]]:
        """
        Lists tools from an MCP server for dynamic registration.
        Supports Hybrid Discovery: Native MCP -> AI REST Bridge.
        """
        # 1. Proactive Handshake for HTTP(S) URLs
        if not is_stdio and "://" in base_url:
            try:
                async with httpx.AsyncClient() as client:
                    # We do a GET to see what kind of server it is
                    resp = await client.get(base_url, timeout=10.0)
                    content_type = resp.headers.get("content-type", "").lower()
                    
                    if "text/event-stream" not in content_type:
                        logger.info(f"[MCP Service] URL {base_url} is {content_type}, fallback to AI REST Bridge (is_resource={is_resource}).")
                        try:
                            # If this fails, we want to know why, but we DON'T want to try native MCP
                            # because we know it's not a native MCP server from the content-type.
                            return await self._discover_rest_bridge(base_url, initial_response=resp, is_resource=is_resource)
                        except Exception as bridge_err:
                            logger.error(f"[MCP Service] AI Bridge failed for REST-detected endpoint: {bridge_err}")
                            raise Exception(f"AI Discovery failed: {bridge_err}")
            except Exception as e:
                # If the proactive check fails COMPLETELY (e.g. connection error), we log warning and continue
                logger.warning(f"[MCP Service] Proactive network check failed for {base_url}: {str(e)}")

        # 2. Native MCP Trace
        try:
            if is_stdio:
                server_params = StdioServerParameters(command="python", args=[base_url])
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        return [t.model_dump() for t in tools_result.tools]
            else:
                # Use a combined approach: if the context manager fails, it might be a protocol error
                try:
                    async with sse_client(base_url) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_result = await session.list_tools()
                            return [t.model_dump() for t in tools_result.tools]
                except Exception as sse_err:
                    error_msg = str(sse_err).lower()
                    if "text/event-stream" in error_msg or "404" in error_msg or "405" in error_msg:
                        logger.info(f"[MCP Service] Protocol mismatch on {base_url}, triggering AI Bridge (is_resource={is_resource})...")
                        return await self._discover_rest_bridge(base_url, is_resource=is_resource)
                    raise sse_err
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[MCP Service] Discovery fatal error: {error_msg}")
            
            # Final fallback for generic connection errors
            if not is_stdio and ("connection" in error_msg.lower() or "timeout" in error_msg.lower()):
                try:
                    return await self._discover_rest_bridge(base_url, is_resource=is_resource)
                except Exception as bridge_err:
                    error_msg = f"Fallo total (MCP y Bridge): {str(bridge_err)}"
            
            raise Exception(error_msg)

    async def _discover_rest_bridge(self, url: str, initial_response: Optional[httpx.Response] = None, is_resource: bool = False) -> List[Dict[str, Any]]:
        """
        Uses a deterministic approach for tool discovery:
        1. Regex extracts parameters from the URL template.
        2. LLM is used ONLY to generate a concise description based on sample data.
        3. The tool definition is constructed via code for 100% structural reliability.
        """
        try:
            # 1. Parameter Extraction (Deterministic)
            params = re.findall(r'\{(.*?)\}', url)
            
            # 2. Get Sample Data for Context
            sample_data = ""
            if initial_response and initial_response.status_code < 400:
                sample_data = initial_response.text[:1000]
            else:
                async with httpx.AsyncClient() as client:
                    # Sanitize URL for discovery fetch (remove placeholders)
                    fetch_url = re.sub(r'\{.*?\}', '', url).rstrip('/')
                    try:
                        response = await client.get(fetch_url, timeout=10.0)
                        sample_data = response.text[:500]
                    except Exception as fetch_err:
                        sample_data = f"Resource URL: {url} (Fetch failed: {fetch_err})"

            # 3. Request Description from LLM (Lightweight)
            from app.core.llm import LLMProvider
            llm = await LLMFactory.get_llm(
                provider=LLMProvider.OLLAMA,
                temperature=0,
                max_tokens=64, # Small for speed/stability
                timeout=60.0
            )

            description = f"Acceso al recurso {url}"
            prompt = f"Basado en esta muestra de datos: {sample_data[:500]}\nEscribe una descripción corta (máx 15 palabras) de lo que hace este endpoint: {url}\nResponde solo con la descripción."
            
            try:
                res = await llm.ainvoke(prompt)
                llm_desc = res.content.strip()
                if llm_desc and len(llm_desc) > 5:
                    description = llm_desc
            except Exception as llm_err:
                logger.warning(f"[MCP Service] LLM description failed, using fallback: {llm_err}")

            # 4. Construct MCP Tool Definition (Deterministic)
            # Create a clean name from the URL (use the resource name, not the param)
            parts = [p for p in url.split('/') if p and not p.startswith('http')]
            if parts and parts[-1].startswith('{') and len(parts) > 1:
                resource_name = parts[-2]
            elif parts:
                resource_name = parts[-1].replace('{', '').replace('}', '')
            else:
                resource_name = "resource"
            
            tool_name = f"get_{resource_name}"
            
            # Build inputSchema
            properties = {}
            for p in params:
                properties[p] = {
                    "type": "string",
                    "description": f"Parámetro {p} para el endpoint"
                }

            tool_def = {
                "name": tool_name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": params
                },
                "config": {
                    "transport": "rest",
                    "url": url,
                    "method": "GET"
                }
            }

            logger.info(f"[MCP Service] Deterministic discovery successful for {url} (found {len(params)} params)")
            return [tool_def]

        except Exception as e:
            logger.error(f"[MCP Service] Deterministic Discovery failed for {url}: {str(e)}")
            # Even if everything fails, return a basic tool based on regex params
            try:
                params = re.findall(r'\{(.*?)\}', url)
                return [{
                    "name": "get_api_resource",
                    "description": f"Acceso manual al endpoint {url}",
                    "inputSchema": {"type": "object", "properties": {p: {"type": "string"} for p in params}, "required": params},
                    "config": {"transport": "rest", "url": url, "method": "GET"}
                }]
            except:
                return []
