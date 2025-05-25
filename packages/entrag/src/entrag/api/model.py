import abc
import logging
from typing import TypeVar

from entrag.data_model.document import Chunk, ExternalChunk


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
    def build_store(self, chunks: list[Chunk]) -> None:
        """
        Build a retrieval store from chunks.
        This could be a FAISS index, a knowledge graph, or another structure.
        """
        pass

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[Chunk], list[ExternalChunk]]:
        """
        Retrieve relevant chunks based on the input query.
        Retrieval logic can be vector-based, graph-based, or hybrid.

        Returns a tuple of (retrieved chunks, external chunks).
        """
        pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response using the given prompt.
        """
        pass
