import os

import faiss
import numpy as np
from loguru import logger
from openai import Client

from entrag.api.model import RAGLM
from entrag.data_model.document import Chunk, ChunkEmbedding


class BaselineRAG(RAGLM):
    def __init__(self, *, storage_dir="./test_rag_vector_store", chunks: list[Chunk]) -> None:
        super().__init__()
        self.index = None
        self.chunks = chunks
        self.openai_client = Client(api_key=os.getenv("OPENAI_API_KEY"))
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")

        self.load_store()

    def embed_query(self, query: str) -> list[float]:
        response = self.openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        return response.data[0].embedding

    def persist_store(self) -> None:
        """
        Persist the FAISS Index on disk.
        """
        if self.index is not None:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info("Persisted the vector store to disk.")
        else:
            logger.warning("No vector store to persist.")

    def load_store(self) -> bool:
        """
        Load the FAISS index from disk if it exists.
        """
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded vector store with {self.index.ntotal} chunks")
                return True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
        return False

    def build_store(self, embeddings: list[ChunkEmbedding]) -> faiss.Index | None:
        """
        Build a vector store for the provided embeddings.
        Uses FAISS for the vector store creation.
        """
        if not embeddings:
            logger.warning("No embeddings provided to build the vector store.")
            return None

        if self.load_store():
            logger.info("Using existing vector store. Skipping build.")
            return self.index

        try:
            vectors = np.array(embeddings, dtype=np.float32)
        except Exception as exc:
            logger.error("Failed to convert embeddings to numpy array: %s", exc)
            return None

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            logger.error("Invalid embedding shape. Expected a 2D array with shape (num_chunks, vector_dim).")
            return None

        vector_dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(vector_dim)
        self.index.add(vectors)
        self.persist_store()

        logger.info(f"Built vector store with {self.index.ntotal} chunks, dimension: {vector_dim}")
        return self.index

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[Chunk], list[str]]:
        query_vector = np.array(self.embed_query(query), dtype=np.float32).reshape(1, -1)
        _, indices = self.index.search(query_vector, top_k)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks, []

    def generate(self, prompt: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}]
        )
        response = completion.choices[0].message.content
        logger.info(f"Generated response: {response}")
        return response
