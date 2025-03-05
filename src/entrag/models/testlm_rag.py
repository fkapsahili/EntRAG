import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from entrag.api.model import RAGLM
from entrag.data_model.document import Chunk


class TestLMRAG(RAGLM):
    def __init__(self) -> None:
        super().__init__()
        self.index = None
        self.chunk_store = []
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode_chunk(self, chunk: Chunk) -> list[float]:
        return self.embedding_model.encode(chunk.chunk_text).tolist()

    def build_store(self, chunks: list[Chunk]) -> faiss.Index | None:
        """
        Build a vector store for the provided chunks.
        Uses FAISS for the vector store creation.
        """
        if not chunks:
            logger.warning("No document provided to build the vector store.")
            return None

        embeddings = [self.encode_chunk(chunk) for chunk in chunks]
        try:
            vectors = np.array(embeddings).astype("float32")
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
        logger.info(f"Built vector store with {self.index.ntotal} chunks, dimension: {vector_dim}")

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        query_vector = np.array(self.encode_chunk(Chunk(chunk_text=query))).astype("float32").reshape(1, -1)
        _, indices = self.index.search(query_vector, top_k)
        retrieved_chunks = [self.chunk_store[i] for i in indices[0]]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks

    def generate(self, query: str, retrieved_chunks: list[Chunk], generation_kwargs: dict | None = None) -> str:
        combined_context = " ".join([chunk.chunk_text for chunk in retrieved_chunks])
        return "Hello, world!"

    def evaluate(self, queries: list[str]) -> list[str]:
        return [self.generate(q, self.retrieve(q)) for q in queries]
