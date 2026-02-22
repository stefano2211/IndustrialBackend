from app.core.config import settings
from app.core.llm import LLMFactory
from app.domain.agent.subagents.rag_industrial.tools import retrieve_documents
from app.domain.agent.subagents.rag_industrial.prompts import prompt
from langchain_core.messages import AIMessage, ToolMessage
import json

def create_industrial_graph():
    llm = LLMFactory.get_llm(role="subagent", temperature=0)
    tools = [retrieve_documents]
    llm_with_tools = llm.bind_tools(tools)
    
    # Chain: Prompt -> LLM with Tools
    chain = prompt | llm_with_tools

    async def industrial_node(state):
        messages = state["messages"]
        
        # Invoke the chain
        response = await chain.ainvoke({"messages": messages})
        
        # If the LLM wants to call tools, we execute them and potentially loop.
        # However, as a simple LangGraph node, we generally return the AIMessage.
        # The parent graph (orchestrator) handles tool execution via ToolNode if 
        # this node is treated as a component. 
        # But if this is a stand-alone node returning a final answer:
        
        if response.tool_calls:
            # Simple direct tool execution for this specific node
            for tool_call in response.tool_calls:
                if tool_call["name"] == "retrieve_documents":
                    # Execute tool
                    tool_output = await retrieve_documents.ainvoke(tool_call["args"])
                    
                    # Add messages to history and re-invoke to get final answer
                    new_messages = messages + [response, ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=str(tool_output)
                    )]
                    
                    final_response = await chain.ainvoke({"messages": new_messages})
                    return {
                        "messages": [AIMessage(content=final_response.content, name="Industrial_RAG")]
                    }
        
        return {
            "messages": [AIMessage(content=response.content, name="Industrial_RAG")]
        }

    return industrial_node
