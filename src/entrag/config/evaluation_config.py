from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """
    Configuration section for chunking.
    """

    model_name: str = Field(
        description="The model name to use for chunking.", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    min_tokens: int = Field(description="Minimum number of tokens in a chunk.", default=256)
    max_tokens: int = Field(description="Maximum number of tokens in a chunk.", default=1024)
    target_chunk_size: int = Field(description="Target chunk size in tokens.", default=512)
    similarity_threshold: float = Field(description="Similarity threshold for chunking.", default=0.3)


class EvaluationConfig(BaseModel):
    """
    Model for the evaluation configuration.
    """

    config_name: str

    chunks: ChunkingConfig = Field(description="Chunking configuration to use.", default=ChunkingConfig())
