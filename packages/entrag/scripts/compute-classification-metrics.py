import json

import click


@click.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the JSON file containing classification results.",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Name of the model to filter results for.",
)
def main(results_file: str, model: str) -> dict:
    with open(results_file, "r") as f:
        data = json.load(f)

    classifications = []
    for result in data:
        if result["model"] == model and result["evaluator"] == "answer_classification_llm":
            score = result["score"]
            if score == 1.0:
                classifications.append("perfect")
            elif score == 0.5:
                classifications.append("acceptable")
            elif score == 0.0:
                classifications.append("missing")
            elif score == -1.0:
                classifications.append("incorrect")

    total = len(classifications)
    if total == 0:
        return {"accuracy": 0, "hallucination": 0, "missing": 0, "truthfulness": 0}

    perfect_count = classifications.count("perfect")
    acceptable_count = classifications.count("acceptable")
    missing_count = classifications.count("missing")
    incorrect_count = classifications.count("incorrect")

    metrics = {
        "accuracy": ((perfect_count + acceptable_count) / total) * 100,
        "hallucination": (incorrect_count / total) * 100,
        "missing": (missing_count / total) * 100,
        "truthfulness": (perfect_count * 1.0 + acceptable_count * 0.5 + missing_count * 0.0 + incorrect_count * (-1.0))
        / total,
    }
    print(f"Metrics for model {model}:")
    print(metrics)
