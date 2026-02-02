from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator
from app.domain.models.tool_config import ToolConfig

class CustomToolState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tool_config: ToolConfig
    extracted_params: Dict[str, Any]
    api_response: Dict[str, Any]
