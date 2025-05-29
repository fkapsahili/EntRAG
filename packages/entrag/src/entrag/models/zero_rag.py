import faiss
from loguru import logger

from entrag.api.ai import BaseAIEngine
from entrag.api.model import RAGLM
from entrag.data_model.document import Chunk, ChunkEmbedding, ExternalChunk


class ZeroRAG(RAGLM):
    """
    Zero-RAG approach that does not use a retrieval step.
    This implementation is helpful for the evaluation of the hallucination baseline.
    """

    def __init__(self, ai_engine: BaseAIEngine, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.ai_engine = ai_engine

    def build_store(self, embeddings: list[ChunkEmbedding]) -> faiss.Index | None:
        """
        Skip the build store step for Zero-RAG.
        """
        logger.info("Running [Zero-RAG]. Skipping vector store build.")
        return None

    def retrieve(self, query: str, top_k: int) -> tuple[list[Chunk], list[ExternalChunk]]:
        return [], []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.ai_engine.chat_completion(
            model=self.model_name,
            user=user_prompt,
            system=system_prompt,
        )
        logger.debug(f"Generated response: {response}")
        return response
