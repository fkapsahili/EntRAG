"""
This script analyzes the questions of the given benchmark dataset and displays a breakdown of the questions
by question type and domain.
"""

import json
import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from entrag.data_model.question_answer import QuestionAnswerPair


sns.set_style("whitegrid")
sns.set_palette("husl")


@click.command()
@click.option(
    "--dataset",
    type=str,
    required=True,
    default="20250413-entrag-dev",
    help="The dataset to be analyzed.",
)
@click.option(
    "--output",
    type=str,
    required=False,
    help="Output directory to save individual plots. If not specified, plots will only be displayed.",
)
def main(dataset: str, output: str = None) -> None:
    """
    Analyze the dataset and display a breakdown of the questions by type and domain.
    """
    dataset_path = os.path.join("datasets", f"{dataset}.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    if output:
        os.makedirs(output, exist_ok=True)
        logger.info(f"Output directory created/verified: {output}")

    with open(dataset_path, "r") as file:
        dataset_ = json.load(file)
        questions = [QuestionAnswerPair(**item) for item in dataset_]
        total_count = len(questions)

    df = pd.DataFrame([
        {"question_type": q.question_type, "domain": q.domain, "dynamism": q.dynamism} for q in questions
    ])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("EntRAG Dataset Analysis", fontsize=16, fontweight="bold")

    # 1. Question Types
    question_type_counts = df["question_type"].value_counts()
    sns.barplot(x=question_type_counts.values, y=question_type_counts.index, ax=axes[0], palette="viridis")
    axes[0].set_title(f"Question Types (Total: {total_count})", fontweight="bold")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Question Type")

    for i, v in enumerate(question_type_counts.values):
        axes[0].text(v + 0.1, i, str(v), va="center", fontweight="bold")

    # 2. Domains
    domain_counts = df["domain"].value_counts()
    sns.barplot(x=domain_counts.values, y=domain_counts.index, ax=axes[1], palette="muted")
    axes[1].set_title(f"Question Domains (Total: {total_count})", fontweight="bold")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Domain")

    for i, v in enumerate(domain_counts.values):
        axes[1].text(v + 0.1, i, str(v), va="center", fontweight="bold")

    # 3. Dynamism
    dynamism_counts = df["dynamism"].value_counts()
    colors = ["#2E8B57", "#CD853F"]
    sns.barplot(x=dynamism_counts.index, y=dynamism_counts.values, ax=axes[2], palette=colors)
    axes[2].set_title(f"Question Dynamism (Total: {total_count})", fontweight="bold")
    axes[2].set_xlabel("Dynamism")
    axes[2].set_ylabel("Count")

    for i, v in enumerate(dynamism_counts.values):
        axes[2].text(i, v + 1, str(v), ha="center", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output:
        # Individual plot 1: Question Types
        plt.figure(figsize=(8, 6))
        sns.barplot(x=question_type_counts.values, y=question_type_counts.index, palette="viridis")
        plt.title(f"Question Types (Total: {total_count})", fontweight="bold", fontsize=14)
        plt.xlabel("Count")
        plt.ylabel("Question Type")

        for i, v in enumerate(question_type_counts.values):
            plt.text(v + 0.1, i, str(v), va="center", fontweight="bold")

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.tight_layout()

        types_path = os.path.join(output, f"{dataset}_question_types.png")
        plt.savefig(types_path, dpi=300, bbox_inches="tight")
        logger.info(f"Question types plot saved: {types_path}")
        plt.close()

        # Individual plot 2: Domains
        plt.figure(figsize=(8, 6))
        sns.barplot(x=domain_counts.values, y=domain_counts.index, palette="muted")
        plt.title(f"Question Domains (Total: {total_count})", fontweight="bold", fontsize=14)
        plt.xlabel("Count")
        plt.ylabel("Domain")

        for i, v in enumerate(domain_counts.values):
            plt.text(v + 0.1, i, str(v), va="center", fontweight="bold")

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.tight_layout()

        domains_path = os.path.join(output, f"{dataset}_domains.png")
        plt.savefig(domains_path, dpi=300, bbox_inches="tight")
        logger.info(f"Domains plot saved: {domains_path}")
        plt.close()

        # Individual plot 3: Dynamism
        plt.figure(figsize=(6, 6))
        sns.barplot(x=dynamism_counts.index, y=dynamism_counts.values, palette=colors)
        plt.title(f"Question Dynamism (Total: {total_count})", fontweight="bold", fontsize=14)
        plt.xlabel("Dynamism")
        plt.ylabel("Count")

        for i, v in enumerate(dynamism_counts.values):
            plt.text(i, v + 1, str(v), ha="center", fontweight="bold")

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.tight_layout()

        dynamism_path = os.path.join(output, f"{dataset}_dynamism.png")
        plt.savefig(dynamism_path, dpi=300, bbox_inches="tight")
        logger.info(f"Dynamism plot saved: {dynamism_path}")
        plt.close()

    plt.show()


if __name__ == "__main__":
    main()
