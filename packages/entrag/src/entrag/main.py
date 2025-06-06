import os
import sys
from pathlib import Path
from typing import Literal

import click
from loguru import logger

from entrag.ai import GeminiEngine, OpenAIEngine
from entrag.api.ai import BaseAIEngine
from entrag.config.load_config import load_eval_config
from entrag.models.baseline_rag import BaselineRAG
from entrag.models.function_calling_rag import FunctionCallingRAG
from entrag.models.hybrid_rag import HybridRAG
from entrag.models.zero_rag import ZeroRAG
from entrag.preprocessing.create_chunks import create_chunks_for_documents
from entrag.preprocessing.create_embeddings import create_embeddings_for_chunks
from entrag.tasks.question_answering import evaluate_question_answering
from entrag.tasks.utils import create_run_id, save_combined_results, save_model_results


logger.remove()  # Remove the default logger
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)


def _create_ai_engine(provider: Literal["openai", "gemini"]) -> BaseAIEngine:
    if provider == "openai":
        return OpenAIEngine()
    elif provider == "gemini":
        return GeminiEngine()
    else:
        raise NotImplementedError(f"AI provider [{provider}] is not implemented yet.")


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="Name of the configuration to use (e.g. 'default' for evaluation_configs/default/config.yaml)",
)
def main(config: str) -> None:
    """
    Main entry point for the evaluation pipeline.

    This command runs the complete evaluation pipeline using the specified configuration.
    The config should be the name of a directory under evaluation_configs/ that contains
    a config.yaml file.

    Example:
        uv run poe entrag --config default
        (uses evaluation_configs/default/config.yaml)
    """

    eval_config = load_eval_config(config)

    llm_kwargs = {
        "model_name": eval_config.model_evaluation.model_name,
        "ai_engine": _create_ai_engine(eval_config.model_evaluation.model_provider),
    }

    run_id = create_run_id()
    output_dir = (
        Path(eval_config.model_evaluation.output_directory)
        / f"{str(run_id)}_{eval_config.model_evaluation.model_name}"
    )
    logger.info(f"Created evaluation run with ID: [{run_id}]")

    chunks = create_chunks_for_documents(eval_config)
    logger.info(f"Loaded [{len(chunks)}] chunks.")

    embeddings = create_embeddings_for_chunks(eval_config)

    logger.info(
        f"Using LLM: [{llm_kwargs['model_name']}] from provider [{eval_config.model_evaluation.model_provider}]"
    )

    model_kwargs = {"chunks": chunks}
    models = (
        ZeroRAG(**llm_kwargs),
        BaselineRAG(**model_kwargs, **llm_kwargs),
        HybridRAG(
            **model_kwargs, **llm_kwargs, reranking_model_name=eval_config.model_evaluation.reranking_model_name
        ),
        FunctionCallingRAG(
            **model_kwargs,
            **llm_kwargs,
        ),
    )

    all_results = {}

    for model in models:
        model_name = model.__class__.__name__
        logger.info(f"Evaluating model: [{model_name}]")
        model.build_store(embeddings)

        model_results = []

        if eval_config.tasks.question_answering.run:
            qa_results = evaluate_question_answering(
                model, eval_config, output_file=f"{output_dir}/{model_name}_qa_logs.json"
            )
            model_results.extend(qa_results)

        if model_results:
            save_model_results(model_name, model_results, output_dir)
            all_results[model_name] = model_results

    # If we have results from multiple models
    if len(all_results) > 1:
        logger.info("Creating model comparison visualizations")
        save_combined_results(all_results, output_dir)
        logger.info(f"All evaluation visualizations saved to [{output_dir}].")
