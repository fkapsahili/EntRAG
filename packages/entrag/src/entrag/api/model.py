from abc import ABC, abstractmethod
from typing import TypeVar

from entrag.data_model.document import Chunk, ExternalChunk


T = TypeVar("T", bound="RAGLM")


class RAGLM(ABC):
    def __init__(self) -> None:
        """
        Base class for Retrieval-Augmented Generation (RAG) benchmark models.
        Defines the interface for different RAG approaches (e.g., vector-based, graph-based).
        """
        pass

    @abstractmethod
    def build_store(self, chunks: list[Chunk]) -> None:
        """
        Build a retrieval store from chunks.
        This could be a FAISS index, a knowledge graph, or another structure.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> tuple[list[Chunk], list[ExternalChunk]]:
        """
        Retrieve relevant chunks based on the input query.
        Retrieval logic can be vector-based, graph-based, or hybrid.

        Returns a tuple of (retrieved chunks, external chunks).
        """
        pass

    @abstractmethod
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response using the given prompts.
        """
        pass
