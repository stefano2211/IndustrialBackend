from langchain_core.tools import tool
from app.domain.retrieval.searcher import SemanticSearcher
from typing import List

searcher = SemanticSearcher()

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve financial reports, contracts, and internal documents relevant to the query.
    Use this tool to answer questions about the company's data.
    """
    results = searcher.search(query, limit=5)
    
    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        text = res["text"]
        formatted_docs.append(f"--- Documento: {source} ---\n{text}\n")
    
    if not formatted_docs:
        return "No se encontraron documentos relevantes."
        
    return "\n".join(formatted_docs)
