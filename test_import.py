import sys
import os

sys.path.append(os.getcwd())

try:
    print("Attempting to import create_industrial_graph...")
    from app.domain.agent.subagents.rag_industrial.graph import create_industrial_graph
    print("Success!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

try:
    print("\nAttempting to import orchestrator_node...")
    from app.domain.agent.orchestrator.nodes import orchestrator_node
    print("Success!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
