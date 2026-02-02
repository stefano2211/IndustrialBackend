from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from app.domain.agent.orchestrator.state import AgentState
from app.domain.agent.orchestrator.nodes import orchestrator_node, TOOLS

# Create the graph
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("tools", ToolNode(TOOLS))

# Edges
workflow.add_edge(START, "orchestrator")

# Conditional Edge: 
# Check if the orchestrator called a tool (tools_condition).
# If yes -> go to "tools"
# If no -> END (return response to user)
workflow.add_conditional_edges(
    "orchestrator",
    tools_condition,
)

# Edge back from tools to orchestrator
# This allows the orchestrator to read the tool output and formulate a final answer
workflow.add_edge("tools", "orchestrator")

graph = workflow.compile()
