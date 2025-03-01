import abc
import logging
from typing import TypeVar, final
from pydantic import BaseModel
import numpy as np
import faiss

import os

eval_logger = logging.getLogger(__name__)


T = TypeVar("T", bound="RAGLM")


class Document(BaseModel):
    """
    Model for a document used in retrieval.
    """

    title: str | None = None
    content: str
    source: str | None = None
    metadata: dict[str, str] | None = None


class RAGLM(abc.ABC):
    def __init__(self, data_dir: str | None = None) -> None:
        """
        Base class for Retrieval-Augmented Generation (RAG) benchmark models.
        Defines the interface that should be implemented by all RAGLM subclasses.
        RAGLMs take text as input and yield strings as output.
        """
        self.data_dir = data_dir
        if self.data_dir is not None:
            os.makedirs(self.data_dir, exist_ok=True)

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Retrieve relevant documents based on the input query.

        :param query: The query string to retrieve documents for.
        :param top_k: Number of top documents to retrieve.
        :return: A list of retrieved documents.
        """
        pass

    @abc.abstractmethod
    def generate(
        self,
        context: str,
        retrieved_docs: list[Document],
        generation_kwargs: dict | None = None,
    ) -> str:
        """
        Generate a response using the given context and retrieved documents.

        :param context: The context string or question.
        :param retrieved_docs: List of retrieved document dictionaries.
        :param generation_kwargs: Additional generation parameters.
        :return: The generated answer.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, queries: list[str]) -> list[str]:
        """
        Evaluate the model on a list of queries. This method should handle retrieval and generation.

        :param queries: List of input queries.
        :return: List of model-generated responses.
        """
        pass

    @abc.abstractmethod
    def encode_document(self, document: Document) -> list[float]:
        """
        Encode the document into a vector representation.

        :param document: The document to encode.
        """
        pass

    @final
    def build_vector_store(self, documents: list[Document]) -> faiss.Index | None:
        """
        Build a vector store for the provided documents.
        Uses FAISS for the vector store creation.
        """
        if not documents:
            eval_logger.warning("No document provided to build the vector store.")
            return None

        embeddings = [self.encode_document(doc) for doc in documents]
        try:
            vectors = np.array(embeddings).astype("float32")
        except Exception as exc:
            eval_logger.error("Failed to convert embeddings to numpy array: %s", exc)
            return None

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            eval_logger.error(
                "Invalid embedding shape. Expected a 2D array with shape (num_docs, vector_dim)."
            )
            return None

        vector_dim = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dim)
        index.add(vectors)
        eval_logger.info(
            f"Built vector store with {index.ntotal} documents, dimension: {vector_dim}"
        )

        if self.data_dir:
            index_path = os.path.join(self.data_dir, "vector_store.index")
            faiss.write_index(index, index_path)
            eval_logger.info(f"Vector store saved to {index_path}")

        return index
