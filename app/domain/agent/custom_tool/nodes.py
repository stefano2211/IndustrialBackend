import json
import httpx
from langchain_core.messages import ToolMessage, SystemMessage
from app.core.config import settings
from app.core.llm import LLMFactory
from app.domain.agent.custom_tool.state import CustomToolState
from loguru import logger

# Initialize LLM
llm = LLMFactory.get_llm(role="subagent", temperature=0)

def agent_node(state: CustomToolState):
    """
    Uses the tool's system prompt to extract parameters from the user's query.
    """
    messages = state.get("messages", [])
    tool_config = state.get("tool_config")
    
    if not tool_config:
        return {"messages": [ToolMessage(content="Error: Tool configuration missing.", tool_call_id="unknown")]}

    # Construct system prompt
    base_prompt = tool_config.system_prompt
    schema_str = json.dumps(tool_config.parameter_schema, indent=2)
    
    system_prompt = SystemMessage(content=f"""{base_prompt}

Your goal is to extract the necessary parameters to call the API.
The required parameters schema is:
{schema_str}

Return ONLY a JSON object with the extracted parameters. If a parameter is missing, try to infer it or return null.
Example: {{"symbol": "AAPL"}}
""")

    response = llm.invoke([system_prompt] + messages)
    content = response.content.strip()
    
    # Clean up JSON (remove markdown code blocks if present)
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    try:
        params = json.loads(content)
        return {"extracted_params": params}
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON parameters: {content}")
        return {"messages": [ToolMessage(content="Error: Could not extract parameters.", tool_call_id="unknown")]}

async def api_call_node(state: CustomToolState):
    """
    Executes the HTTP request using the extracted parameters and auth config.
    """
    tool_config = state.get("tool_config")
    params = state.get("extracted_params", {})
    
    if not tool_config:
        return {"messages": [ToolMessage(content="Error: Tool configuration missing.", tool_call_id="unknown")]}

    url = tool_config.api_url
    method = tool_config.method.upper()
    headers = tool_config.headers.copy() if tool_config.headers else {}
    auth = tool_config.auth_config
    
    # Handle Auth
    if auth:
        auth_type = auth.get("type")
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {auth.get('value')}"
        elif auth_type == "api_key":
            key = auth.get("key", "X-API-Key")
            value = auth.get("value")
            location = auth.get("location", "header")
            if location == "header":
                headers[key] = value
            elif location == "query":
                # Append to URL or params later
                pass # TODO: Handle query param auth
        elif auth_type == "basic":
            # Handled by httpx auth
            pass

    # Replace path parameters in URL (e.g., /stocks/{symbol})
    for key, value in params.items():
        if value and f"{{{key}}}" in url:
            url = url.replace(f"{{{key}}}", str(value))

    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                # Add remaining params as query params
                query_params = {k: v for k, v in params.items() if f"{{{k}}}" not in tool_config.api_url}
                response = await client.get(url, headers=headers, params=query_params)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=params)
            else:
                return {"messages": [ToolMessage(content=f"Error: Unsupported method {method}", tool_call_id="unknown")]}
            
            response.raise_for_status()
            data = response.json()
            
            return {
                "api_response": data,
                "messages": [ToolMessage(content=json.dumps(data), tool_call_id="unknown")]
            }
            
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return {"messages": [ToolMessage(content=f"Error calling API: {str(e)}", tool_call_id="unknown")]}
