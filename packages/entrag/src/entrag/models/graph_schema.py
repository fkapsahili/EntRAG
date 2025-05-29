from typing import List, Optional

from pydantic import BaseModel


class Entity(BaseModel):
    id: str  # unique identifier (e.g., "company:Apple Inc.")
    type: str  # e.g., "company", "ticker", "metric"
    name: str  # human-readable name
    chunk_id: Optional[str] = None  # optional: which chunk this entity came from


class Relationship(BaseModel):
    source: str  # entity id
    target: str  # entity id
    type: str  # e.g., "co-occurs", "represents", "mentions"
    weight: Optional[float] = 1.0
    chunk_id: Optional[str] = None  # optional: which chunk this relationship came from


class ChunkGraphExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
