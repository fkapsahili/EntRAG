from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """
    Configuration section for chunking.
    """

    enabled: bool = Field(description="Flag to enable the chunking step.", default=True)
    output_directory: str = Field(description="Output directory for the chunked documents.", default="data/chunks")
    files_directory: str = Field(description="Input directory containing the text files to chunk.", default="data/raw")
    dataset_name: str = Field(description="Name of the dataset to use.", default="dataset")
    model_name: str = Field(
        description="The model name to use for chunking.", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    min_tokens: int = Field(description="Minimum number of tokens in a chunk.", default=256)
    max_tokens: int = Field(description="Maximum number of tokens in a chunk.", default=1024)
    target_chunk_size: int = Field(description="Target chunk size in tokens.", default=512)
    similarity_threshold: float = Field(description="Similarity threshold for chunking.", default=0.3)
    overlap_size: int = Field(description="Number of overlapping sentences between chunks.", default=2)


class QuestionAnsweringConfig(BaseModel):
    """
    Configuration section for question answering.
    """

    dataset_path: str = Field(description="File path of the dataset to use.", default="dataset")


class TasksConfig(BaseModel):
    """
    Configuration section for tasks.
    """

    question_answering: QuestionAnsweringConfig = Field(
        description="Question answering configuration to use.", default=QuestionAnsweringConfig()
    )


class EvaluationConfig(BaseModel):
    """
    Model for the evaluation configuration.
    """

    config_name: str

    chunking: ChunkingConfig = Field(description="Chunking configuration to use.", default=ChunkingConfig())
    tasks: TasksConfig = Field(description="Tasks configuration to use.", default=TasksConfig())
