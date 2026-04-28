from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.domain.proactiva.agent.retrieval.searcher import SemanticSearcher
from loguru import logger

# Lazy-init singleton — created on first call, not on import (testable via DI)
_searcher: SemanticSearcher | None = None


def _get_searcher() -> SemanticSearcher:
    """Returns the shared SemanticSearcher instance (lazy init)."""
    global _searcher
    if _searcher is None:
        _searcher = SemanticSearcher()
    return _searcher


@tool
async def ask_knowledge_agent(
    config: RunnableConfig,
    query: str = "",
    **kwargs,
) -> str:
    """
    Search the user's Knowledge Base for documents, invoices, manuals, regulations, incident reports, and any uploaded files.
    ALWAYS use this tool when the user asks about their documents or needs information from uploaded files.
    The tool automatically knows which user and knowledge base to search.
    Input should be a clear search query describing what information you need.
    """
    # -------------------------------------------------------------
    # Handle Llama 3.1 hallucinated nested structure: 
    # {"tool": "ask_knowledge_agent", "parameters": {"query": "invoice"}}
    # -------------------------------------------------------------
    if not query:
        parameters = kwargs.get("parameters") or kwargs.get("args") or {}
        query = parameters.get("query", "")
        
    if not query:
        return "Error: No query provided. Please provide a search query."

    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    knowledge_base_id = configurable.get("knowledge_base_id")
    session = configurable.get("session")

    logger.info(
        f"[Knowledge Tool] Searching: query='{query}', "
        f"user_id={user_id}, kb_id={knowledge_base_id}"
    )

    if not user_id:
        return "Error: No user_id found in the config. Cannot search."

    if not knowledge_base_id:
        return (
            "Error: No hay ninguna base de conocimientos (Knowledge Base) seleccionada en este chat. "
            "No puedes buscar documentos porque el usuario seleccionó 'Sin Contexto'. "
            "Indícale al usuario que debe seleccionar una colección de documentos para poder buscar información."
        )

    searcher = _get_searcher()
    # Now it dynamically reads limit from SystemSettings if session is present
    results = await searcher.search(
        query,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        session=session
    )

    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        section = res["metadata"].get("section", "")
        text = res["text"]
        score = res["score"]
        section_str = f" › {section}" if section and section != "No section" else ""
        # Trim excessively long chunks to save tokens (max ~800 chars)
        if len(text) > 800:
            text = text[:800] + "…"
        formatted_docs.append(
            f"--- [{source}{section_str}] (relevancia: {score:.0%}) ---\n{text}\n"
        )

    if not formatted_docs:
        return (
            "No se encontraron documentos relevantes en la base de conocimientos. "
            "El usuario puede necesitar subir los documentos primero."
        )

    logger.info(f"[Knowledge Tool] Found {len(formatted_docs)} results")
    return "\n".join(formatted_docs)
