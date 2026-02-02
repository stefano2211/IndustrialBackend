from langchain_core.tools import tool
from app.domain.agent.subagents.rag_financial.graph import create_financial_graph
from app.domain.agent.subagents.rag_placeholder.graph import placeholder_node
from langchain_core.messages import HumanMessage

# Initialize graphs/nodes
# Note: In a real app we might want to cache the graph compilation or initialization
financial_graph = create_financial_graph()

# Wrapper for Financial RAG
@tool
async def ask_financial_agent(query: str) -> str:
    """
    Use this tool to ask questions about financial data, revenue, sales, or business reports.
    Input should be the specific question asking for information.
    """
    # Create the state for the sub-agent
    # The sub-agent expects a dictionary with "messages"
    # financial_graph is actually a node function (async) in current implementation
    response = await financial_graph({"messages": [HumanMessage(content=query)]})
    
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
