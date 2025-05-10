"""
This script analyzes the questions of the given benchmark dataset and displays a breakdown of the questions
by question type and domain.
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
    Analyze the dataset and display a breakdown of the questions by type and domain.
    """
    dataset_path = os.path.join("datasets", f"{dataset}.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, "r") as file:
        dataset_ = json.load(file)
        questions = [QuestionAnswerPair(**item) for item in dataset_]
        total_count = len(questions)

        # Create a figure with two subplots side by side
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # 1. Plot counts by question type
        question_types = [item.question_type for item in questions]
        question_types_count = {item: question_types.count(item) for item in set(question_types)}

        ax1.bar(question_types_count.keys(), question_types_count.values())
        ax1.set_title(f"Question Types Count (Total: {total_count})")
        ax1.set_xlabel("Question Type")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Plot counts by domain
        domains = [item.domain for item in questions]
        domains_count = {item: domains.count(item) for item in set(domains)}

        # Sort domains by count in descending order for better visualization
        domains_sorted = dict(sorted(domains_count.items(), key=lambda x: x[1], reverse=True))

        ax2.bar(domains_sorted.keys(), domains_sorted.values())
        ax2.set_title(f"Question Domains Count (Total: {total_count})")
        ax2.set_xlabel("Domain")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Plot counts by dynamism
        dynamism = [item.dynamism for item in questions]
        dynamism_count = {item: dynamism.count(item) for item in set(dynamism)}
        ax3.bar(dynamism_count.keys(), dynamism_count.values())
        ax3.set_title(f"Question Dynamism Count (Total: {total_count})")
        ax3.set_xlabel("Dynamism")
        ax3.set_ylabel("Count")
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
