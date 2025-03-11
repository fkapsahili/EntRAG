import os
import pickle

import faiss
import numpy as np
from google.genai import Client
from loguru import logger

from entrag.api.model import RAGLM
from entrag.data_model.document import Chunk
from entrag.prompts.default_prompts import SIMPLE_QA_PROMPT
from entrag.utils.prompt import get_query_time


class TestLMRAG(RAGLM):
    def __init__(self, storage_dir="./test_rag_vector_store") -> None:
        super().__init__()
        self.index = None
        self.chunk_store = []
        self.genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")
        self.chunks_path = os.path.join(storage_dir, "chunks.pkl")

        self.load_store()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        embeddings = self.genai_client.models.embed_content(
            model="text-embedding-004", contents=[chunk.chunk_text for chunk in chunks]
        )
        return [embedding.values for embedding in embeddings.embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self.genai_client.models.embed_content(model="text-embedding-004", contents=query).embeddings[0].values

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

    def build_store(self, chunks: list[Chunk], batch_size=100) -> faiss.Index | None:
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

        embeddings = []
        for i in range(0, len(chunks), batch_size):
            logger.info(f"Embedding chunks {i} to {i + batch_size}")
            embeddings.extend(self.embed_chunks(chunks[i : i + batch_size]))

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
        search_query = self.query_expansion(query)
        query_vector = np.array(self.embed_query(search_query), dtype=np.float32).reshape(1, -1)
        _, indices = self.index.search(query_vector, top_k)
        retrieved_chunks = [self.chunk_store[i] for i in indices[0]]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks

    def query_expansion(self, query: str) -> str:
        prompt = """
        Given the following query:
        {}
        Please generate a query which is suitable for retrieving relevant information
        in a vector-based retrieval system.

        Only respond with the query text.
        """.format(query)
        response = self.genai_client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
        logger.info(f"Expanded query: {response.text}")
        return response.text

    def generate(self, query: str, retrieved_chunks: list[Chunk], generation_kwargs: dict | None = None) -> str:
        combined_context = " ".join([chunk.chunk_text for chunk in retrieved_chunks])
        logger.info(f"Generated response for query: {query}")
        prompt_template = SIMPLE_QA_PROMPT.format(
            query=query, references=combined_context, query_time=get_query_time()
        )
        response = self.genai_client.models.generate_content(model="gemini-1.5-pro", contents=[prompt_template])
        logger.info(f"Generated response: {response.text}")
        return response.text

    def evaluate(self, queries: list[str]) -> list[str]:
        return [self.generate(q, self.retrieve(q)) for q in queries]
