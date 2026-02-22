from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from app.domain.agent.subagents.rag_industrial.graph import create_industrial_graph
from app.domain.agent.subagents.rag_placeholder.graph import placeholder_node

# Initialize graphs/nodes
# Note: In a real app we might want to cache the graph compilation or initialization
financial_graph = create_industrial_graph()

# Wrapper for Industrial RAG
@tool
async def ask_industrial_agent(
    query: str, 
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Use this tool to ask questions about industrial safety, regulations (OSHA, ISO), compliance, hazards, or incident reports.
    Input should be the specific question asking for information.
    """
    # Extract user_id from config
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    # Create the state for the sub-agent
    response = await financial_graph({
        "messages": [HumanMessage(content=query)],
        "user_id": user_id
    })
    
    # Extract the final answer (last message)
    return response["messages"][-1].content

# Wrapper for Placeholder/General Agent
@tool
async def ask_placeholder_agent(query: str) -> str:
    """
    Use this tool to ask questions about other topics, or features currently in development.
    """
    # Placeholder node is a simple async function
    result = await placeholder_node({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content
