from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """
    Configuration section for chunking.
    """

    enabled: bool = Field(description="Flag to enable the chunking step.", default=True)
    files_directory: str = Field(description="Input directory containing the text files to chunk.", default="data/raw")
    output_directory: str = Field(description="Output directory for the chunked documents.", default="data/chunks")
    dataset_name: str = Field(description="Name of the dataset to use.", default="dataset")
    max_tokens: int = Field(description="Maximum number of tokens in a chunk.", default=1024)


class EmbeddingConfig(BaseModel):
    """
    Configuration section for embeddings.
    """

    enabled: bool = Field(description="Flag to enable the embedding step.", default=True)
    model: str = Field(description="Model to use for embeddings.", default="text-embedding-3-small")
    batch_size: int = Field(description="Batch size for embedding.", default=4)
    output_directory: str = Field(description="Output directory for the embeddings.", default="data/embeddings")


class QuestionAnsweringConfig(BaseModel):
    """
    Configuration section for question answering.
    """

    dataset_path: str = Field(description="File path of the dataset to use.", default="dataset")
    run: bool = Field(description="Flag to enable the question answering step.", default=True)


class TasksConfig(BaseModel):
    """
    Configuration section for tasks.
    """

    question_answering: QuestionAnsweringConfig = Field(
        description="Question answering configuration to use.", default=QuestionAnsweringConfig()
    )
    dynamic_question_answering: QuestionAnsweringConfig = Field(
        description="Dynamic question answering configuration to use.", default=QuestionAnsweringConfig()
    )


class EvaluationConfig(BaseModel):
    """
    Model for the evaluation configuration.
    """

    config_name: str

    chunking: ChunkingConfig = Field(description="Chunking configuration to use.", default=ChunkingConfig())
    embedding: EmbeddingConfig = Field(description="Embedding configuration to use.", default=EmbeddingConfig())
    tasks: TasksConfig = Field(description="Tasks configuration to use.", default=TasksConfig())
