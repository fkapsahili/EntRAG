import os

import faiss
from loguru import logger
from openai import Client

from entrag.api.model import RAGLM
from entrag.data_model.document import Chunk, ChunkEmbedding, ExternalChunk


class ZeroRAG(RAGLM):
    """
    Zero-RAG approach that does not use a retrieval step.
    This implementation is helpful for the evaluation of the hallucination baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.openai_client = Client(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_query(self, query: str) -> list[float]:
        response = self.openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        return response.data[0].embedding

    def build_store(self, embeddings: list[ChunkEmbedding]) -> faiss.Index | None:
        """
        Skip the build store step for Zero-RAG.
        """
        logger.info("Running [Zero-RAG]. Skipping vector store build.")
        return None

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[Chunk], list[ExternalChunk]]:
        return [], []

    def generate(self, prompt: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": prompt}]
        )
        response = completion.choices[0].message.content
        logger.debug(f"Generated response: {response}")
        return response
