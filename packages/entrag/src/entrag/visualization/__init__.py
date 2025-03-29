import matplotlib.pyplot as plt
import pandas as pd

from entrag.data_model.question_answer import EvaluationResult


def print_evaluation_table(results: list[EvaluationResult]):
    """
    Print evaluation results as a formatted table.
    """
    if not results:
        print("No evaluation results to display.")
        return

    df = pd.DataFrame([r.model_dump() for r in results])
    df = df[["question_id", "evaluator", "score"]]  # control column order

    print("\nEvaluation Results:\n")
    print(df.to_string(index=False, float_format="%.4f"))


def plot_evaluation_results(results: list[EvaluationResult]):
    """
    Plot the evaluation results.
    """
    if not results:
        print("No evaluation results to plot.")
        return

    df = pd.DataFrame([r.model_dump() for r in results])

    if df.empty:
        print("No evaluation results to plot.")
        return

    df_grouped = df.groupby("evaluator", as_index=False)["score"].mean()
    df_grouped = df_grouped.sort_values(by="score", ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_grouped["evaluator"], df_grouped["score"], color=plt.cm.Paired.colors)

    plt.xlabel("Evaluator")
    plt.ylabel("Average Score")
    plt.title("Average Evaluation Score by Evaluator")
    plt.xticks(rotation=45, ha="right")

    for bar, score in zip(bars, df_grouped["score"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()
