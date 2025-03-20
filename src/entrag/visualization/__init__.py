import matplotlib.pyplot as plt
import pandas as pd

from entrag.data_model.question_answer import EvaluationResult


def plot_evaluation_results(results: list[EvaluationResult]):
    """
    Plot the evaluation results.
    """
    df = pd.DataFrame(results)

    if df.empty:
        print("No evaluation results to plot.")
        return

    df = df.sort_values(by="score", ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["evaluator"], df["score"], color=plt.cm.Paired.colors)

    plt.xlabel("Evaluator")
    plt.ylabel("Score")
    plt.title("Evaluation Scores by Evaluator")
    plt.xticks(rotation=45, ha="right")

    for bar, score in zip(bars, df["score"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.2f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.show()
