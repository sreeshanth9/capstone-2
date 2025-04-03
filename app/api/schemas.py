from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None