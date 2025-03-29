import sys

from loguru import logger

from entrag.config.load_config import load_eval_config
from entrag.models.baseline_rag import BaselineRAG
from entrag.preprocessing.create_chunks import create_chunks_for_documents
from entrag.tasks.question_answering import evaluate_question_answering


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

    model = BaselineRAG()
    model.build_store(chunks)

    evaluate_question_answering(model, config)


if __name__ == "__main__":
    main()
