from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition
from app.agent.state import AgentState
from app.agent.nodes import agent, tool_node

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.add_edge(START, "agent")

# Add conditional edges
# If the agent decides to call a tool, go to "tools". Otherwise, END.
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# From tools, always go back to agent
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()
