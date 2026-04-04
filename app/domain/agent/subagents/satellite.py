"""
Satellite System Agents — Specialized connectors for external platforms.

These agents handle tasks outside the industrial IoT/RAG domain:
  - SAP: ERP data, orders, inventory.
  - Google: Calendar, search, workspace.
  - Office: Outlook, documents, Teams.

STATUS: These integrations are currently in DEVELOPMENT.
All tools return a clear DEMO disclaimer — no real data is returned.

Each is wrapped as a CompiledSubAgent using create_deep_agent so they can
have their own LLM reasoning loop when the external connector is real.
"""

from langchain_core.tools import tool
from deepagents import CompiledSubAgent, create_deep_agent
from loguru import logger


# ── SAP AGENT ──────────────────────────────────────────────────────────────
def create_sap_agent(model, checkpointer=None, store=None):
    """Handles ERP and Supply Chain requests (DEMO — connector not configured)."""
    system_prompt = (
        "You are an SAP ERP specialist. You have access to inventory, "
        "purchase orders, and supplier data. Answer user queries about "
        "stock levels or order status accurately. If data is missing, "
        "explain that the SAP connector is currently in read-only mode."
    )

    # ⚠️ DEMO: SAP connector not yet configured — returns placeholder data
    @tool
    def check_inventory(part_id: str):
        """Check stock levels in SAP ERP."""
        return (
            f"[DEMO — SAP connector not configured] "
            f"The SAP ERP integration is currently under development. "
            f"Real inventory data for part '{part_id}' is not available yet. "
            f"Please contact your administrator to set up the SAP connector."
        )

    graph = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_inventory],
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="sap-agent",
        description="Access SAP ERP for inventory, orders, and procurement data.",
        graph=graph,
    )


# ── GOOGLE AGENT ────────────────────────────────────────────────────────────
def create_google_agent(model, checkpointer=None, store=None):
    """Handles Google Workspace and Search requests (DEMO — connector not configured)."""
    system_prompt = (
        "You are a Google Workspace assistant. Use Google Search for general "
        "external knowledge and Workspace tools for calendar or mail tasks."
    )

    # ⚠️ DEMO: Google connector not yet configured — returns placeholder data
    @tool
    def google_search(query: str):
        """Search the public internet via Google."""
        return (
            f"[DEMO — Google Search connector not configured] "
            f"The Google Workspace integration is currently under development. "
            f"Real search results for '{query}' are not available yet. "
            f"Please contact your administrator to set up the Google connector."
        )

    graph = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[google_search],
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="google-agent",
        description="Search the internet or access Google Workspace (Mail, Calendar).",
        graph=graph,
    )


# ── OFFICE AGENT ────────────────────────────────────────────────────────────
def create_office_agent(model, checkpointer=None, store=None):
    """Handles Microsoft 365 requests (DEMO — connector not configured)."""
    system_prompt = (
        "You are an Office 365 assistant. Access Outlook emails and "
        "OneDrive documents to help the user."
    )

    # ⚠️ DEMO: M365 connector not yet configured — returns placeholder data
    @tool
    def read_outlook_email(count: int = 5):
        """Read latest emails from Outlook."""
        return (
            f"[DEMO — Microsoft 365 connector not configured] "
            f"The Outlook/OneDrive integration is currently under development. "
            f"Real email data is not available yet. "
            f"Please contact your administrator to set up the M365 connector."
        )

    graph = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[read_outlook_email],
        checkpointer=checkpointer,
        store=store,
    )

    return CompiledSubAgent(
        name="office-agent",
        description="Access Microsoft 365 (Outlook, OneDrive, Teams).",
        graph=graph,
    )
