from pydantic import BaseModel


class Document(BaseModel):
    """
    Model for a document used in retrieval.
    """

    title: str | None = None
    content: str
    source: str | None = None
    metadata: dict[str, str] | None = None


class DatasetDocument(BaseModel):
    """
    Model for a dataset document.
    """

    document_id: str
    document_name: str
    document_text: str


class Chunk(BaseModel):
    """
    A text chunk of a processed document.
    """

    document_id: str
    document_name: str
    document_page: int
    chunk_location_id: int
    chunk_text: str
    chunk_length_tokens: int


class ChunkEmbedding(BaseModel):
    """
    A chunk embedding.
    """

    document_id: str
    document_name: str
    document_page: int
    chunk_location_id: int
    embedding: list[float]


class ExternalChunk(BaseModel):
    """
    A text chunk coming from an API response.
    """

    source: str
    content: str
