from typing import TypedDict, Annotated, List, Dict, Any, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """The shared state of the agent workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    user_id: str
