import sys

from loguru import logger

from entrag.config.load_config import load_eval_config
from entrag.models.testlm_rag import TestLMRAG
from entrag.preprocessing.create_chunks import create_chunks_for_documents


logger.remove()  # Remove the default logger
logger.add(
    sys.stderr,
    level="DEBUG",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)


def main() -> None:
    """
    Main entry point for the evaluation script.
    """
    config = load_eval_config("default")

    chunks = create_chunks_for_documents(config)
    logger.info(f"Created {len(chunks)} chunks.")

    # We don't use the configured model provider for now
    model = TestLMRAG()
    model.build_store(chunks)
    retrieved_chunks = model.retrieve("What is the capital of France?")
    print("Retrieved chunks:")
    for chunk in retrieved_chunks:
        print(chunk.chunk_text)


if __name__ == "__main__":
    main()
