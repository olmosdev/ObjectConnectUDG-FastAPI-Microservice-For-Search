from pydantic import BaseModel, Field
from typing import List, Optional

# --- Input Models ---

class PublicationCreate(BaseModel):
    """
    Represents the creation or display structure of a publication.
    """
    id: int = Field(..., description="Unique ID of the publication")
    title: str
    description: str
    category_name: str = Field(..., description="Name of the product category")

class SearchQuery(BaseModel):
    """
    Data structure for performing a similarity search.
    """
    title: Optional[str] = Field(None, description="Title or keywords for the search")
    description: str = Field(..., min_length=3, description="Main text body for the search")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

# --- Output Models ---

class MatchResult(BaseModel):
    """
    Individual search result with its similarity score.
    """
    id: int
    title: str
    description: str
    similarity: float

class SearchResponse(BaseModel):
    """
    Envelope for a search operation result.
    """
    total_found: int
    matches: List[MatchResult]
    message: Optional[str] = None

class StatusResponse(BaseModel):
    """
    Standard response for single-item operations or status checks.
    """
    status: str
    message: str
    processed_id: Optional[int] = None
