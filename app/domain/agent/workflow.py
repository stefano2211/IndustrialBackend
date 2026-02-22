from app.domain.agent.orchestrator.graph import compile_graph

# Global app instance starting with default (no memory)
app = compile_graph()

def reload_with_memory(checkpointer, store):
    """
    Overwrites the global app instance with a memory-aware graph.
    Should be called during app startup.
    """
    global app
    app = compile_graph(checkpointer=checkpointer, store=store)
