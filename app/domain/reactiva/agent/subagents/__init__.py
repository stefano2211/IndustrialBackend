"""
Reactive Subagents — Sistema 1 wrappers for the reactive domain.

These are the reactive-domain counterparts of the proactive Sistema 1 subagents.
They use the same shared LoRA infrastructure but with reactive-specific prompts
and naming, ensuring full architectural independence between proactiva and reactiva.

Only shared with proactiva:
  - create_system1_historico_agent / create_system1_vl_agent (infraestructura LoRA)
  - create_composite_backend (infraestructura LangGraph)
"""
