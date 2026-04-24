"""Reactive Knowledge RAG Tool — searches the reactive Qdrant collection.

Identical interface to ask_knowledge_agent but queries the isolated
reactive knowledge base (SOPs, emergency procedures, maintenance manuals).
Uses tenant_id scope instead of user_id.
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.domain.reactiva.retrieval.reactive_searcher import ReactiveSemanticSearcher

# Lazy-init singleton
_searcher: ReactiveSemanticSearcher | None = None


def _get_searcher() -> ReactiveSemanticSearcher:
    global _searcher
    if _searcher is None:
        _searcher = ReactiveSemanticSearcher()
    return _searcher


@tool
async def ask_reactive_knowledge(
    config: RunnableConfig,
    query: str = "",
    **kwargs,
) -> str:
    """
    Search the reactive Knowledge Base for SOPs, emergency procedures,
    maintenance manuals, incident response protocols, and safety regulations.
    ALWAYS use this tool when you need reference documents for event diagnosis.
    Input should be a clear search query describing what information you need.
    """
    if not query:
        parameters = kwargs.get("parameters") or kwargs.get("args") or {}
        query = parameters.get("query", "")

    if not query:
        return "Error: No query provided. Please provide a search query."

    configurable = config.get("configurable", {})
    knowledge_base_id = configurable.get("knowledge_base_id")
    session = configurable.get("session")
    tenant_id = configurable.get("tenant_id", "default")

    logger.info(
        f"[ReactiveKnowledge] Searching: query='{query}', "
        f"tenant_id={tenant_id}, kb_id={knowledge_base_id}"
    )

    searcher = _get_searcher()
    results = await searcher.search(
        query,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        session=session,
    )

    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        section = res["metadata"].get("section", "")
        text = res["text"]
        score = res["score"]
        section_str = f" › {section}" if section and section != "No section" else ""
        if len(text) > 800:
            text = text[:800] + "…"
        formatted_docs.append(
            f"--- [{source}{section_str}] (relevancia: {score:.0%}) ---\n{text}\n"
        )

    if not formatted_docs:
        return (
            "No se encontraron documentos relevantes en la base de conocimientos reactiva. "
            "Puede ser necesario cargar SOPs y procedimientos de emergencia primero."
        )

    logger.info(f"[ReactiveKnowledge] Found {len(formatted_docs)} results")
    return "\n".join(formatted_docs)
