from fastapi import APIRouter, Query
from app.retrieval.searcher import SemanticSearcher
from app.agent.workflow import app as agent_app
from app.schemas.api import SearchResponse, ChatRequest, ChatResponse
from typing import Optional

router = APIRouter()
searcher = SemanticSearcher()

@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=3), 
    limit: int = Query(5, ge=1, le=20),
    entity_filter: Optional[str] = Query(None),
    category_filter: Optional[str] = Query(None)
):
    """Búsqueda sin cambios (usa Qdrant)."""
    results = searcher.search(q, limit=limit, entity_filter=entity_filter, category_filter=category_filter)
    return {"query": q, "results": results}

from langchain_core.messages import HumanMessage

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint RAG: Responde preguntas usando el agente LangGraph (con Tools)."""
    # El input ahora es una lista de mensajes
    inputs = {"messages": [HumanMessage(content=request.query)]}
    result = await agent_app.ainvoke(inputs)
    
    # El resultado final está en el último mensaje del historial
    last_message = result["messages"][-1]
    answer = last_message.content
    
    return {
        "answer": answer,
        "sources": [] # TODO: Extraer fuentes de los mensajes de herramienta si es necesario
    }
