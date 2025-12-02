from langgraph.graph import StateGraph, END
from app.agent.custom_tool.state import CustomToolState
from app.agent.custom_tool.nodes import agent_node, api_call_node

# Define the graph
workflow = StateGraph(CustomToolState)

# Add nodes
workflow.add_node("agent_node", agent_node)
workflow.add_node("api_call_node", api_call_node)

# Define edges
workflow.set_entry_point("agent_node")
workflow.add_edge("agent_node", "api_call_node")
workflow.add_edge("api_call_node", END)

# Compile the graph
custom_tool_graph = workflow.compile()
