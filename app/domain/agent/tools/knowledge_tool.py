from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.domain.retrieval.searcher import SemanticSearcher
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
    query: str,
    config: RunnableConfig,
) -> str:
    """
    Search the user's Knowledge Base for documents, invoices, manuals, regulations, incident reports, and any uploaded files.
    ALWAYS use this tool when the user asks about their documents or needs information from uploaded files.
    The tool automatically knows which user and knowledge base to search.
    Input should be a clear search query describing what information you need.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    knowledge_base_id = configurable.get("knowledge_base_id")

    logger.info(
        f"[Knowledge Tool] Searching: query='{query}', "
        f"user_id={user_id}, kb_id={knowledge_base_id}"
    )

    if not user_id:
        return "Error: No user_id found in the config. Cannot search."

    searcher = _get_searcher()
    results = searcher.search(
        query,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        limit=5,
    )

    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        text = res["text"]
        score = res["score"]
        formatted_docs.append(
            f"--- Documento: {source} (score: {score:.2f}) ---\n{text}\n"
        )

    if not formatted_docs:
        return (
            "No se encontraron documentos relevantes en la base de conocimientos. "
            "El usuario puede necesitar subir los documentos primero."
        )

    logger.info(f"[Knowledge Tool] Found {len(formatted_docs)} results")
    return "\n".join(formatted_docs)
