from pydantic import BaseModel
from typing import List, Dict, Any

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]  

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    user_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    thread_id: str | None = None