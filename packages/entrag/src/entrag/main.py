import sys
from pathlib import Path

from loguru import logger

from entrag.config.load_config import load_eval_config
from entrag.models.baseline_rag import BaselineRAG
from entrag.models.zero_rag import ZeroRAG
from entrag.preprocessing.create_chunks import create_chunks_for_documents
from entrag.preprocessing.create_embeddings import create_embeddings_for_chunks
from entrag.tasks.dynamic_question_answering import evaluate_dynamic_question_answering
from entrag.tasks.question_answering import evaluate_question_answering
from entrag.visualization import (
    plot_evaluation_results,
    plot_evaluator_distribution,
    plot_model_comparison,
    plot_question_level_scores,
    plot_radar_comparison,
)
from entrag.visualization.utils import (
    create_run_id,
    save_combined_results,
    save_model_results,
)


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

    run_id = create_run_id()
    output_dir = Path(config.model_evaluation.output_directory) / str(run_id)
    logger.info(f"Created evaluation run with ID: {run_id}")

    chunks = create_chunks_for_documents(config)
    logger.info(f"Loaded {len(chunks)} chunks.")

    embeddings = create_embeddings_for_chunks(config)

    model_kwargs = {"chunks": chunks}
    models = (
        ZeroRAG(),
        BaselineRAG(**model_kwargs),
        # HybridRAG(),
    )

    all_results = {}

    for model in models:
        model_name = model.__class__.__name__
        logger.info(f"Evaluating model: {model_name}")
        model.build_store(embeddings)

        model_results = []

        if config.tasks.question_answering.run:
            qa_results = evaluate_question_answering(model, config)
            model_results.extend(qa_results)

        if config.tasks.dynamic_question_answering.run:
            dqa_results = evaluate_dynamic_question_answering(model, config)
            model_results.extend(dqa_results)

        if model_results:
            save_model_results(model_name, model_results, output_dir)
            all_results[model_name] = model_results

            # Visualize individual model results
            plot_evaluation_results(
                qa_results, model_name=model_name, output_file=output_dir / f"{model_name}_results.png"
            )
            plot_question_level_scores(
                qa_results, model_name=model_name, output_file=output_dir / f"{model_name}_question_scores.png"
            )
            plot_evaluator_distribution(
                qa_results, model_name=model_name, output_file=output_dir / f"{model_name}_evaluator_distribution.png"
            )

    # If we have results from multiple models
    if len(all_results) > 1:
        logger.info("Creating model comparison visualizations")

        save_combined_results(all_results, output_dir)

        # Generate additional comparison plots
        plot_model_comparison(all_results, figsize=(14, 8), output_file=output_dir / "model_comparison.png")
        plot_radar_comparison(
            all_results, top_n_models=len(all_results), output_file=output_dir / "radar_comparison.png"
        )

        logger.info(f"All evaluation visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
