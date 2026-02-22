from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from app.domain.agent.orchestrator.state import AgentState
from app.domain.agent.orchestrator.nodes import orchestrator_node, TOOLS

def create_workflow():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tools", ToolNode(TOOLS))

    # Edges
    workflow.add_edge(START, "orchestrator")

    # Conditional Edge
    workflow.add_conditional_edges(
        "orchestrator",
        tools_condition,
    )

    # Edge back from tools
    workflow.add_edge("tools", "orchestrator")
    
    return workflow

def compile_graph(checkpointer=None, store=None):
    workflow = create_workflow()
    return workflow.compile(checkpointer=checkpointer, store=store)

# Base graph (can be overwritten by memory initialized one)
graph = compile_graph()
