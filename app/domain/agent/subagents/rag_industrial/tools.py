from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from app.domain.retrieval.searcher import SemanticSearcher
from typing import List

searcher = SemanticSearcher()

@tool
def retrieve_documents(
    query: str,
    user_id: Annotated[str, InjectedToolArg]
) -> str:
    """
    Retrieve industrial safety reports, OSHA/ISO regulations, incident logs, and compliance documents relevant to the query.
    Use this tool to answer questions about hazards, safety standards, and past incidents.
    """
    results = searcher.search(query, user_id=user_id, limit=5)
    
    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        text = res["text"]
        formatted_docs.append(f"--- Documento: {source} ---\n{text}\n")
    
    if not formatted_docs:
        return "No se encontraron documentos relevantes."
        
    return "\n".join(formatted_docs)
