"""
Satellite System Agents — Specialized connectors for external platforms.

These agents handle tasks outside the industrial IoT/RAG domain:
- SAP: ERP data, orders, inventory.
- Google: Calendar, search, workspace.
- Office: Outlook, documents, teams.

Each is wrapped as a CompiledSubAgent if it needs its own LLM logic, 
or as a simple Tool if it's a direct API connector.
"""

from langchain_core.tools import tool
from deepagents import CompiledSubAgent, create_deep_agent
from loguru import logger

# ── SAP AGENT ──────────────────────────────────────────────────────────────
def create_sap_agent(model, checkpointer=None, store=None):
    """Handles ERP and Supply Chain requests."""
    system_prompt = (
        "You are an SAP ERP specialist. You have access to inventory, "
        "purchase orders, and supplier data. Answer user queries about "
        "stock levels or order status accurately. If data is missing, "
        "explain that the SAP connector is currently in read-only mode."
    )
    
    # Placeholder tools for SAP
    @tool
    def check_inventory(part_id: str):
        """Check stock levels in SAP ERP."""
        return f"SAP: Part {part_id} has 42 units in Warehouse A."

    graph = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_inventory],
        checkpointer=checkpointer,
        store=store
    )
    
    return CompiledSubAgent(
        name="sap-agent",
        description="Access SAP ERP for inventory, orders, and procurement data.",
        graph=graph
    )

# ── GOOGLE AGENT ────────────────────────────────────────────────────────────
def create_google_agent(model, checkpointer=None, store=None):
    """Handles Google Workspace and Search requests."""
    system_prompt = (
        "You are a Google Workspace assistant. Use Google Search for general "
        "external knowledge and Workspace tools for calendar or mail tasks."
    )
    
    @tool
    def google_search(query: str):
        """Search the public internet via Google."""
        return f"Google Search Result for '{query}': [Placeholder Result]"

    graph = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[google_search],
        checkpointer=checkpointer,
        store=store
    )
    
    return CompiledSubAgent(
        name="google-agent",
        description="Search the internet or access Google Workspace (Mail, Calendar).",
        graph=graph
    )

# ── OFFICE AGENT ────────────────────────────────────────────────────────────
def create_office_agent(model, checkpointer=None, store=None):
    """Handles Microsoft 365 requests."""
    system_prompt = (
        "You are an Office 365 assistant. Access Outlook emails and "
        "OneDrive documents to help the user."
    )
    
    @tool
    def read_outlook_email(count: int = 5):
        """Read latest emails from Outlook."""
        return "Office: No new security alerts in the last 5 emails."

    graph = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[read_outlook_email],
        checkpointer=checkpointer,
        store=store
    )
    
    return CompiledSubAgent(
        name="office-agent",
        description="Access Microsoft 365 (Outlook, OneDrive, Teams).",
        graph=graph
    )
