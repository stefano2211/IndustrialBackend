from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]  

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class ModelParams(BaseModel):
    """Advanced model parameters controllable from the chat UI."""
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop_sequence: Optional[str] = None

class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    query: str
    thread_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    model_id: Optional[str] = None
    params: Optional[ModelParams] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    thread_id: str | None = None