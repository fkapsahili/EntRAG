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
    logger.info(f"Loaded {len(chunks)} chunks.")

    model = TestLMRAG()
    model.build_store(chunks)

    retrieved_chunks = model.retrieve("Volkswagen emissions scandal", top_k=5)
    print("Retrieved chunks:")
    for chunk in retrieved_chunks:
        print(chunk.document_name)


if __name__ == "__main__":
    main()
