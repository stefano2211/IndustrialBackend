from langchain_core.tools import tool
from app.domain.retrieval.searcher import SemanticSearcher

from typing import List

searcher = SemanticSearcher()

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve financial documents relevant to the query. 
    Use this tool when you need to find information in the documents to answer a user's question.
    """
    results = searcher.search(query, limit=5)
    
    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        text = res["text"]
        formatted_docs.append(f"--- Documento: {source} ---\n{text}\n")
    
    return "\n".join(formatted_docs)

@tool
async def use_custom_tool(tool_name: str, query: str) -> str:
    """
    Use this tool to fetch data from external APIs configured in the system.
    Args:
        tool_name: The name of the tool to use (e.g., 'finance_api', 'weather_api').
        query: The user's specific question or query for this tool.
    """
    from app.persistence.db import get_session
    from app.domain.services.tool_config_service import ToolConfigService
    from app.domain.agent.custom_tool.graph import custom_tool_graph
    from langchain_core.messages import HumanMessage
    
    # Get DB session
    async for session in get_session():
        service = ToolConfigService(session)
        tool_config = await service.get_by_name(tool_name)
        
        if not tool_config:
            return f"Error: Tool '{tool_name}' not found."
        
        # Invoke subgraph
        result = await custom_tool_graph.ainvoke({
            "messages": [HumanMessage(content=query)],
            "tool_config": tool_config
        })
        
        api_response = result.get("api_response")
        if api_response:
            return str(api_response)
        
        # Fallback to last message if no structured response
        return result["messages"][-1].content

