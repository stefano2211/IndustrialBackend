import sys
import os
import asyncio

# Add the project root to the python path
sys.path.append(os.getcwd())

from app.domain.agent.orchestrator.nodes import orchestrator_node
from langchain_core.messages import HumanMessage, SystemMessage

def test_agent_routing():
    print("--- Testing Agent Routing for Industrial Compliance ---")
    
    # Simulate a user query about an industrial incident
    query = "What is the penalty for the sulfuric acid leak in the storage area?"
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the orchestrator node
    result = orchestrator_node(initial_state)
    last_message = result["messages"][-1]
    
    print(f"\nQuery: {query}")
    print(f"Orchestrator Response: {last_message.content}")
    
    # Check if tool calls are present (indicating routing)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"\nTool Calls: {last_message.tool_calls}")
        tool_name = last_message.tool_calls[0]['name']
        if tool_name == 'ask_industrial_agent':
            print("SUCCESS: Routed to Industrial Agent")
        else:
            print(f"FAILURE: Routed to {tool_name}")
    else:
        print("FAILURE: No tool call made (Agent might have answered directly)")

if __name__ == "__main__":
    test_agent_routing()
