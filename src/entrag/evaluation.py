"""
Basic evaluation experiment with some pre-defined evaluation datasets.
"""

import click
import logging

from entrag.evaluator_utils import load_documents

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["20250301-entrag-dev"]),
    default="20250301-entrag-dev",
    show_default=True,
)
@click.option("--output", type=click.Path(), default="evaluation.json", show_default=True)
@click.option("--model", type=str, default="gpt-4o-mini", show_default=True)
@click.option("--num-repetitions", type=int, default=1, show_default=True)
def run_evaluation(dataset, output, num_repetitions, model):
    """
    Run the evaluation against the specified dataset.
    """
    # Load the dataset
    dataset_file = f"datasets/{dataset}.json"
    documents = load_documents(dataset_file)
    print(f"Loaded {len(documents)} documents from {dataset}")

    # # Run the evaluation
    # results = []
    # for i in range(num_repetitions):
    #     for qa_instance in dataset:
    #         question = qa_instance["question"]
    #         ground_truth = qa_instance["reference_answer"]
    #         # extra_context_formatted = "\n".join(extra_context)

    #         # inference_result = _run_model_inference(
    #         #     model, question, context=extra_context_formatted
    #         # )
    #         # results.append(_evaluate(question, ground_truth, inference_result))

    # # Save the results
    # with open(output, "w") as f:
    #     json.dump(results, f)

    # logger.info("Evaluation results saved to %s", output)


def _evaluate(question: str, ground_truth: str, inference_result: str): ...


def _run_model_inference(model: str, question: str, context: str = None) -> str:
    """
    Run inference with the specified model.
    """


if __name__ == "__main__":
    run_evaluation()
