from pydantic import BaseModel


class Document(BaseModel):
    """
    Model for a document used in retrieval.
    """

    title: str | None = None
    content: str
    source: str | None = None
    metadata: dict[str, str] | None = None
