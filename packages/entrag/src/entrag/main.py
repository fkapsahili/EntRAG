import sys

from loguru import logger

from entrag.config.load_config import load_eval_config
from entrag.models.baseline_rag import BaselineRAG
from entrag.models.hybrid_rag import HybridRAG
from entrag.preprocessing.create_chunks import create_chunks_for_documents
from entrag.preprocessing.create_embeddings import create_embeddings_for_chunks
from entrag.tasks.dynamic_question_answering import evaluate_dynamic_question_answering
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
    Main entry point for the evaluation pipeline.
    """

    config = load_eval_config("default")

    chunks = create_chunks_for_documents(config)
    logger.info(f"Loaded {len(chunks)} chunks.")

    embeddings = create_embeddings_for_chunks(config)

    model_kwargs = {"chunks": chunks}
    models = (
        BaselineRAG(**model_kwargs),
        # HybridRAG(),
    )
    for model in models:
        logger.info(f"Evaluating model: {model.__class__.__name__}")
        model.build_store(embeddings)

        if config.tasks.question_answering.run:
            evaluate_question_answering(model, config)

        if config.tasks.dynamic_question_answering.run:
            evaluate_dynamic_question_answering(model, config)


if __name__ == "__main__":
    main()
