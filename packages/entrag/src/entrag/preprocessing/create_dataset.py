import uuid
from pathlib import Path

from loguru import logger

from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.document import DatasetDocument


def read_files_directory(files_directory: str) -> list[tuple[str, str]]:
    """
    Load the markdown files from the given directory and return a list of documents.
    """
    documents = []
    for file_path in Path(files_directory).rglob("*.md"):
        with file_path.open("r", encoding="utf-8") as file:
            logger.debug(f"Reading file: {file_path}")
            documents.append((file_path.name, file.read()))
    return documents


def create_dataset(config: EvaluationConfig) -> list[DatasetDocument]:
    """
    Create a dataset from files in the configured directory.
    """
    logger.info("Starting dataset creation.")

    files_data = read_files_directory(config.chunking.files_directory)
    logger.info(f"Loaded {len(files_data)} files from {config.chunking.files_directory}.")

    files_dataset = [
        DatasetDocument(
            document_id=str(uuid.uuid4()),
            document_name=file_data[0],
            document_text=file_data[1],
        )
        for file_data in files_data
    ]

    logger.info(f"Created dataset with {len(files_dataset)} documents.")

    return files_dataset
