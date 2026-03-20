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

    async def _auto_map_response(self, source: str, data: Any, schema_hints: dict = None) -> MCPResponse:
        """
        Maps a JSON payload to KeyFigures and KeyValues using pure deterministic logic.
        1. Recursively maps all scalar fields from the raw JSON.
        2. If schema_hints are provided (from parameter_schema.response), filters to only relevant fields.
        """
        response = MCPResponse(source=source)

        if not isinstance(data, dict):
            response.key_values.append(KeyValue(name="raw_data", value=str(data)))
            return response

        def process_dict(d: dict, prefix: str = "", depth: int = 0):
            for key, val in d.items():
                name = f"{prefix}{key}" if prefix else key
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    response.key_figures.append(KeyFigure(name=name, value=float(val)))
                elif isinstance(val, dict) and depth < 3:
                    process_dict(val, f"{name}.", depth + 1)
                elif isinstance(val, list):
                    if len(val) > 0:
                        if isinstance(val[0], (str, int, float)):
                            # Simple list → join first 15 items
                            response.key_values.append(
                                KeyValue(name=name, value=", ".join(map(str, val[:15])))
                            )
                        elif isinstance(val[0], dict) and depth < 2:
                            # List of objects → extract the most useful string value
                            flat = []
                            for item in val[:15]:
                                if not isinstance(item, dict):
                                    continue
                                # Prefer 'name', then first string/number value
                                candidate = item.get("name")
                                if candidate is None:
                                    for v in item.values():
                                        if isinstance(v, (str, int, float)) and v is not None:
                                            candidate = v
                                            break
                                if candidate is not None:
                                    flat.append(str(candidate))
                            if flat:
                                response.key_values.append(KeyValue(name=name, value=", ".join(flat)))

                elif val is not None and not isinstance(val, bool):
                    response.key_values.append(KeyValue(name=name, value=str(val)))

        process_dict(data)

        # Apply schema-based filter if hints are available
        if schema_hints:
            response = self._filter_response(response, schema_hints)

        return response

    def _filter_response(self, response: MCPResponse, schema_hints: dict) -> MCPResponse:
        """
        Filters MCPResponse using the response schema hints stored during discovery.
        Only keeps key_figures and key_values whose names match (or partially match)
        the expected response field names.
        No LLM involved — pure Python string matching.
        """
        if not schema_hints:
            return response

        # Build a set of normalized hint names for matching
        hint_keys = {k.lower().replace("_", "").replace(" ", "") for k in schema_hints.keys()}

        def _matches_hint(name: str) -> bool:
            # Normalize: remove dots, underscores, spaces, lowercase
            # Also match against any segment of a dotted path (e.g., "damage_relations.double_damage_to")
            parts = name.lower().replace(" ", "").replace("_", "").split(".")
            return any(p in hint_keys or any(h in p for h in hint_keys) for p in parts)

        filtered = MCPResponse(source=response.source)
        filtered.key_figures = [kf for kf in response.key_figures if _matches_hint(kf.name)]
        filtered.key_values = [kv for kv in response.key_values if _matches_hint(kv.name)]

        # Fallback: if filter was too strict and returned nothing, return the top-N unfiltered
        if not filtered.key_figures and not filtered.key_values:
            logger.info(f"[MCP Service] Schema filter removed everything for {response.source}, using top-20 unfiltered.")
            filtered.key_figures = response.key_figures[:10]
            filtered.key_values = response.key_values[:10]

        return filtered

    async def execute_tool(
        self, 
        base_url: str, 
        tool_name: str, 
        arguments: Dict[str, Any],
        is_stdio: bool = False,
        transport_type: str = "mcp",
        method: str = "GET",
        schema_hints: Optional[dict] = None,
    ) -> MCPResponse:
        """
        Dynamically connects to an MCP server, calls a tool, and maps the result.
        schema_hints: Optional dict from parameter_schema["response"] to filter
                      irrelevant fields before returning to the orchestrator.
        """
        logger.info(f"[MCP Service] Executing tool '{tool_name}' ({transport_type}) on {base_url} using {method}")
        
        try:
            if transport_type == "rest":
                # Universal REST Bridge execution
                async with httpx.AsyncClient() as client:
                    method = method.upper()
                    target_url = base_url
                    
                    # 1. URL Parameter Substitution
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
                        response = await client.get(target_url, params=remaining_args, timeout=30.0)
                    else:
                        response = await client.request(method, target_url, json=remaining_args, timeout=30.0)
                    
                    response.raise_for_status()
                    return await self._auto_map_response(tool_name, response.json(), schema_hints=schema_hints)

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

    async def discover_tools(self, base_url: str, is_stdio: bool = False, is_resource: bool = False, method: str = "GET") -> List[Dict[str, Any]]:
        """
        Lists tools from an MCP server for dynamic registration.
        Supports Hybrid Discovery: Native MCP -> AI REST Bridge.
        """
        method = method.upper()

        # 1. Proactive Handshake for HTTP(S) URLs
        if not is_stdio and "://" in base_url:
            try:
                async with httpx.AsyncClient() as client:
                    # We do a GET to see what kind of server it is
                    resp = await client.get(base_url, timeout=10.0)
                    content_type = resp.headers.get("content-type", "").lower()
                    
                    if "text/event-stream" not in content_type:
                        logger.info(f"[MCP Service] URL {base_url} is {content_type}, fallback to AI REST Bridge (method={method}).")
                        try:
                            return await self._discover_rest_bridge(base_url, initial_response=resp, is_resource=is_resource, method=method)
                        except Exception as bridge_err:
                            logger.error(f"[MCP Service] AI Bridge failed for REST-detected endpoint: {bridge_err}")
                            raise Exception(f"AI Discovery failed: {bridge_err}")
            except Exception as e:
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
                try:
                    async with sse_client(base_url) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_result = await session.list_tools()
                            return [t.model_dump() for t in tools_result.tools]
                except Exception as sse_err:
                    error_msg = str(sse_err).lower()
                    if "text/event-stream" in error_msg or "404" in error_msg or "405" in error_msg:
                        logger.info(f"[MCP Service] Protocol mismatch on {base_url}, triggering AI Bridge (method={method})...")
                        return await self._discover_rest_bridge(base_url, is_resource=is_resource, method=method)
                    raise sse_err
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[MCP Service] Discovery fatal error: {error_msg}")
            
            if not is_stdio and ("connection" in error_msg.lower() or "timeout" in error_msg.lower()):
                try:
                    return await self._discover_rest_bridge(base_url, is_resource=is_resource, method=method)
                except Exception as bridge_err:
                    error_msg = f"Fallo total (MCP y Bridge): {str(bridge_err)}"
            
            raise Exception(error_msg)

    async def _discover_rest_bridge(self, url: str, initial_response: Optional[httpx.Response] = None, is_resource: bool = False, method: str = "GET") -> List[Dict[str, Any]]:
        """
        Smart tool discovery for REST APIs:
        1. Regex detects path parameters from {curly_braces} in the URL.
        2. LLM analyzes the URL + sample response and returns structured JSON with:
             - A human-readable description of the endpoint
             - Parameter definitions (type, description, example) for each path/query param
             - Response field hints (type, unit, description) extracted from the sample data
        3. The final tool definition is assembled deterministically from LLM output.
        """
        try:
            # 1. Path parameter extraction (always deterministic)
            path_params: list[str] = re.findall(r'\{(.*?)\}', url)

            # 2. Fetch a sample response for context
            sample_data = ""
            if initial_response and initial_response.status_code < 400:
                sample_data = initial_response.text[:1500]
            else:
                async with httpx.AsyncClient() as client:
                    fetch_url = re.sub(r'\{.*?\}', 'example', url)
                    try:
                        resp = await client.get(fetch_url, timeout=10.0)
                        sample_data = resp.text[:1000]
                    except Exception as fe:
                        sample_data = f"(fetch failed: {fe})"

            # 3. Build resource name deterministically from URL path
            parts = [p for p in url.split('/') if p and not p.startswith('http') and '{' not in p]
            resource_name = parts[-1] if parts else "resource"
            # Prefix tool name by method so GET /users and POST /users are distinct
            method_prefix = {"GET": "get", "POST": "create", "PUT": "update", "PATCH": "patch", "DELETE": "delete"}
            tool_name = f"{method_prefix.get(method, 'call')}_{resource_name}"

            # 4. Ask the LLM to analyse the endpoint and return structured JSON
            from app.core.llm import LLMProvider
            llm = await LLMFactory.get_llm(
                provider=LLMProvider.OLLAMA,
                temperature=0,
                max_tokens=512,
            )

            path_params_hint = (
                f"The URL has these path parameters (already detected): {path_params}"
                if path_params else "The URL has no path parameters."
            )

            # Body vs query params hint for the LLM
            param_placement = "JSON request body" if method in ("POST", "PUT", "PATCH") else "URL query string"

            analysis_prompt = f"""You are an API analyst. Analyze the following REST endpoint and respond with a valid JSON object ONLY (no markdown, no explanation, no extra text).

Endpoint URL: {url}
HTTP Method: {method}
{path_params_hint}
Extra parameters location: {param_placement}

Sample response data (first 1000 chars):
{sample_data[:1000]}

Return ONLY this JSON structure:
{{
  "description": "<one sentence, max 20 words, describing what this endpoint does>",
  "params": {{
    "<param_name>": {{
      "type": "<string|integer|number|boolean|object>",
      "description": "<what this param does>",
      "example": "<a realistic example value>"
    }}
  }},
  "response_fields": {{
    "<field_name>": {{
      "type": "<string|number|boolean|array|object>",
      "unit": "<measurement unit if numeric, e.g. °C, kW, %, or empty string>",
      "description": "<brief description of what this field means>"
    }}
  }}
}}

Rules:
- Include ALL path parameters in "params" (those wrapped in {{}} in the URL). They are always [required].
- For {method} requests, also infer body/query parameters from the URL name and sample data.
- In "response_fields" include only the top-level fields visible in the sample response (max 10).
- If the sample is a list, describe fields of ONE item.
- Do not include internal/system fields (e.g. _id, __v).
"""

            llm_result = None
            try:
                res = await llm.ainvoke(analysis_prompt)
                raw = res.content.strip()
                # Strip accidental markdown fences
                raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
                raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE).strip()
                llm_result = json.loads(raw)
                logger.info(f"[MCP Service] LLM analysis succeeded for {url}")
            except Exception as llm_err:
                logger.warning(f"[MCP Service] LLM analysis failed, using deterministic fallback: {llm_err}")

            # 5. Build the final parameter_schema from LLM output (with deterministic fallback)
            description = (llm_result or {}).get("description") or f"Access to endpoint {url}"
            llm_params: dict = (llm_result or {}).get("params", {})
            llm_response: dict = (llm_result or {}).get("response_fields", {})

            # Merge: LLM params + any path params the LLM missed
            properties: dict = {}
            for pp in path_params:
                if pp in llm_params:
                    properties[pp] = llm_params[pp]
                else:
                    properties[pp] = {
                        "type": "string",
                        "description": f"Path parameter '{pp}' embedded in the URL",
                        "example": pp,
                    }

            # Add any extra query params from LLM that aren't path params
            for pname, pdef in llm_params.items():
                if pname not in properties:
                    properties[pname] = pdef

            parameter_schema = {
                "type": "object",
                "properties": properties,
                "required": path_params,
                "response": llm_response,
            }

            tool_def = {
                "name": tool_name,
                "description": description,
                "inputSchema": parameter_schema,
                "parameter_schema": parameter_schema,
                "config": {
                    "transport": "rest",
                    "url": url,
                    "method": method,
                },
            }

            logger.info(
                f"[MCP Service] Smart discovery successful for {url} "
                f"(path_params={path_params}, response_fields={list(llm_response.keys())})"
            )
            return [tool_def]

        except Exception as e:
            logger.error(f"[MCP Service] Smart Discovery failed for {url}: {e}")
            # Hard fallback — bare minimum so the tool isn't lost
            try:
                pp = re.findall(r'\{(.*?)\}', url)
                return [{
                    "name": "get_api_resource",
                    "description": f"Access to endpoint {url}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {p: {"type": "string", "description": f"Parameter {p}"} for p in pp},
                        "required": pp,
                    },
                    "parameter_schema": {
                        "type": "object",
                        "properties": {p: {"type": "string", "description": f"Parameter {p}"} for p in pp},
                        "required": pp,
                        "response": {},
                    },
                    "config": {"transport": "rest", "url": url, "method": method},
                }]
            except Exception:
                return []
