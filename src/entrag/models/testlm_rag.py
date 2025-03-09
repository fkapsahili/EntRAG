import os
import pickle

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from entrag.api.model import RAGLM
from entrag.data_model.document import Chunk


class TestLMRAG(RAGLM):
    def __init__(self, storage_dir="./test_rag_vector_store") -> None:
        super().__init__()
        self.index = None
        self.chunk_store = []
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")
        self.chunks_path = os.path.join(storage_dir, "chunks.pkl")

        self.load_store()

    def encode_chunk(self, chunk: Chunk) -> list[float]:
        return self.embedding_model.encode(chunk.chunk_text).tolist()

    def encode_query(self, query: str) -> list[float]:
        return self.embedding_model.encode(query).tolist()

    def persist_store(self) -> None:
        """
        Persist the FAISS Index on disk.
        """
        if self.index is not None:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.chunks_path, "wb") as f:
                pickle.dump(self.chunk_store, f)
            logger.info("Persisted the vector store to disk.")
        else:
            logger.warning("No vector store to persist.")

    def load_store(self) -> bool:
        """
        Load the FAISS index and chunks from disk if they exist.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, "rb") as f:
                    self.chunk_store = pickle.load(f)
                logger.info(f"Loaded vector store with {self.index.ntotal} chunks")
                return True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
        return False

    def build_store(self, chunks: list[Chunk]) -> faiss.Index | None:
        """
        Build a vector store for the provided chunks.
        Uses FAISS for the vector store creation.
        """
        if not chunks:
            logger.warning("No document provided to build the vector store.")
            return None

        if self.load_store():
            logger.info("Using existing vector store. Skipping build.")
            return self.index

        embeddings = [self.encode_chunk(chunk) for chunk in chunks]
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
        self.chunk_store = chunks
        self.persist_store()

        logger.info(f"Built vector store with {self.index.ntotal} chunks, dimension: {vector_dim}")
        return self.index

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        query_vector = np.array(self.encode_query(query), dtype=np.float32).reshape(1, -1)
        _, indices = self.index.search(query_vector, top_k)
        retrieved_chunks = [self.chunk_store[i] for i in indices[0]]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks

    def generate(self, query: str, retrieved_chunks: list[Chunk], generation_kwargs: dict | None = None) -> str:
        combined_context = " ".join([chunk.chunk_text for chunk in retrieved_chunks])
        return "Hello, world!"

    def evaluate(self, queries: list[str]) -> list[str]:
        return [self.generate(q, self.retrieve(q)) for q in queries]
