"""
This script analyzed the questions of the given bechmark dataset and displays a breakdown of the questions.
"""

import json
import os

import click
from loguru import logger
from matplotlib import pyplot as plt

from entrag.data_model.question_answer import QuestionAnswerPair


@click.command()
@click.option(
    "--dataset",
    type=str,
    required=True,
    default="20250413-entrag-dev",
    help="The dataset to be analyzed.",
)
def main(dataset: str) -> None:
    """
    Analyze the dataset and display a breakdown of the questions.
    """
    dataset_path = os.path.join("datasets", f"{dataset}.json")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, "r") as file:
        dataset_ = json.load(file)
        questions = [QuestionAnswerPair(**item) for item in dataset_]

        # 1. Plot all counts by question type
        question_types = [item.question_type for item in questions]
        question_types_count = {item: question_types.count(item) for item in set(question_types)}
        total_count = len(questions)
        plt.figure(figsize=(10, 6))
        plt.bar(question_types_count.keys(), question_types_count.values())
        plt.title(f"Question Types Count (Total: {total_count})")
        plt.xlabel("Question Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
