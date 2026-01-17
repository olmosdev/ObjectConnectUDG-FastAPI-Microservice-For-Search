from pydantic import BaseModel, Field
from typing import List, Optional

# --- Input ---
class PublicationCreate(BaseModel):
    id: int = Field(..., description="ID único de la publicación")
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    category: str = Field(..., example="mochila")

class SearchQuery(BaseModel):
    title: Optional[str] = Field(None, description="Título o palabras clave")
    description: str = Field(..., min_length=3)
    category: str = Field(..., description="Categoría para filtrar")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

# --- Output ---
class MatchResult(BaseModel):
    id: int
    title: str
    description: str
    category: str
    similarity: float

class SearchResponse(BaseModel):
    total_found: int
    matches: List[MatchResult]

class StatusResponse(BaseModel):
    status: str
    message: str
    processed_id: Optional[int] = None

class BatchStatusResponse(BaseModel):
    status: str
    items_processed: int
    message: str

class VectorRecord(BaseModel):
    id: int
    vector: List[float]
    cluster: int