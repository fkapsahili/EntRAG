import abc
import logging
from typing import TypeVar

from entrag.data_model.document import Chunk


eval_logger = logging.getLogger(__name__)

T = TypeVar("T", bound="RAGLM")


class RAGLM(abc.ABC):
    def __init__(self) -> None:
        """
        Base class for Retrieval-Augmented Generation (RAG) benchmark models.
        Defines the interface for different RAG approaches (e.g., vector-based, graph-based).
        """
        pass

    @abc.abstractmethod
    def embed_chunk(self, chunk: Chunk) -> list[float] | None:
        """
        Encode a chunk into a vector or other meaningful representation.
        Each subclass must define its own encoding logic or return `None` if not applicable.
        """
        pass

    @abc.abstractmethod
    def build_store(self, chunks: list[Chunk]) -> None:
        """
        Build a retrieval store from chunks.
        This could be a FAISS index, a knowledge graph, or another structure.
        """
        pass

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """
        Retrieve relevant chunks based on the input query.
        Retrieval logic can be vector-based, graph-based, or hybrid.
        """
        pass

    @abc.abstractmethod
    def generate(self, query: str, retrieved_chunks: list[Chunk], generation_kwargs: dict | None = None) -> str:
        """
        Generate a response using the given context and retrieved chunks.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, queries: list[str]) -> list[str]:
        """
        Evaluate the model on a set of queries.
        """
        pass
