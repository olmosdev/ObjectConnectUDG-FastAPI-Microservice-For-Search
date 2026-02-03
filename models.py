from pydantic import BaseModel, Field
from typing import List, Optional

# --- Input ---
class PublicationCreate(BaseModel):
    id: int = Field(..., description="ID único de la publicación")
    title: str
    description: str
    category_name: str = Field(..., description="Nombre de la categoría del producto")

class SearchQuery(BaseModel):
    title: Optional[str] = Field(None, description="Título o palabras clave")
    description: str = Field(..., min_length=3)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

# --- Output ---
class MatchResult(BaseModel):
    id: int
    title: str
    description: str
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


class PostFromSupabase(BaseModel):
    """
    Optimized model for Machine Learning training.
    Represents the minimal data structure fetched from Supabase.
    """
    id: int
    title: str
    description: str
    product_category_id: int

    class Config:
        # 'from_attributes = True' acts as a universal translator:
        # It allows Pydantic to create an instance of this model even if 
        # the data from Supabase arrives as an object with attributes (e.g., data.titulo) 
        # instead of a standard dictionary (e.g., data["titulo"]). 
        # Essential for compatibility with Databases and ORMs.
        from_attributes = True


class PostVectorResponse(BaseModel):
    """
    Representa un registro de la tabla post_vectors.
    """
    post_id: int
    vector_embedding: List[float]
    cluster_id: int

    class Config:
        from_attributes = True