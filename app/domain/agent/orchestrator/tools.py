from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from app.domain.agent.subagents.rag_industrial.graph import create_industrial_graph
from app.domain.agent.subagents.rag_placeholder.graph import placeholder_node

# Initialize graphs/nodes
# Note: In a real app we might want to cache the graph compilation or initialization
financial_graph = create_industrial_graph()

# Wrapper for Knowledge Base RAG
@tool
async def ask_knowledge_agent(
    query: str, 
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Use this tool for EVERYTHING related to searching information, reading documents, invoices, manuals, or any user data.
    Input should be the specific question asking for information, summarizing exactly what the user wants to know.
    """
    # Extract user_id and optionally knowledge_base_id and session from config
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    kb_id = config.get("configurable", {}).get("knowledge_base_id", None)
    session = config.get("configurable", {}).get("session", None)
    
    # Create the state for the sub-agent
    response = await financial_graph({
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "knowledge_base_id": kb_id,
        "session": session
    })
    
    # Extract the final answer (last message)
    return response["messages"][-1].content

# Wrapper for Placeholder/General Agent
@tool
async def ask_placeholder_agent(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Use this tool to ask questions about other topics, or features currently in development.
    """
    session = config.get("configurable", {}).get("session", None)
    # Placeholder node is a simple async function
    result = await placeholder_node({
        "messages": [HumanMessage(content=query)],
        "session": session
    })
    return result["messages"][-1].content
