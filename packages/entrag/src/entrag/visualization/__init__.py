import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from entrag.data_model.question_answer import EvaluationResult


def plot_evaluation_results(
    results: list[EvaluationResult,], *, model_name: str | None = None, output_file: str | None = None
):
    """
    Plot the evaluation results for different evaluator types.
    LLM evaluators (with 'llm' suffix) are scaled from 0-10, while others use 0-1.
    """
    if not results:
        print("No evaluation results to plot.")
        return

    df = pd.DataFrame([r.model_dump() for r in results])
    if df.empty:
        print("No evaluation results to plot.")
        return

    df_display = df.copy()

    llm_evaluators = [eval_name for eval_name in df["evaluator"].unique() if str(eval_name).lower().endswith("llm")]

    # Normalize LLM scores from 0-10 to 0-1 scale for consistency
    for evaluator in llm_evaluators:
        mask = df_display["evaluator"] == evaluator
        df_display.loc[mask, "score"] = df_display.loc[mask, "score"] / 10.0

    df_grouped = df_display.groupby("evaluator", as_index=False)["score"].mean()
    df_grouped = df_grouped.sort_values(by="score", ascending=False)

    plt.figure(figsize=(12, 7))

    # Create color map - different colors for LLM vs standard evaluators
    colors = []
    for evaluator in df_grouped["evaluator"]:
        if str(evaluator).lower().endswith("llm"):
            colors.append("orangered")
        else:
            colors.append("royalblue")

    bars = plt.bar(df_grouped["evaluator"], df_grouped["score"], color=colors)
    plt.ylabel("Average Score (0-1 scale)")
    plt.ylim(0, 1.05)  # Add a little headroom

    if any(llm_evaluators):
        ax2 = plt.twinx()
        ax2.set_ylabel("LLM Score (0-10 scale)")
        ax2.set_ylim(0, 10.5)

    plt.xlabel("Evaluator")
    title = "Average Evaluation Score by Evaluator"
    if model_name:
        title = f"{title} - {model_name}"
    plt.title(title)
    plt.xticks(rotation=45, ha="right")

    for bar, score, evaluator in zip(bars, df_grouped["score"], df_grouped["evaluator"]):
        if str(evaluator).lower().endswith("llm"):
            # Get the original mean score (0-10 scale) for display
            original_score = df[df["evaluator"] == evaluator]["score"].mean()
            display_text = f"{original_score:.2f}"
            y_pos = score
        else:
            display_text = f"{score:.2f}"
            y_pos = score

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            display_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if any(llm_evaluators):
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color="royalblue", label="Standard Metrics (0-1)"),
            plt.Rectangle((0, 0), 1, 1, color="orangered", label="LLM Metrics (0-10)"),
        ]
        plt.legend(handles=legend_elements, loc="best")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")


def plot_question_level_scores(
    results: list[EvaluationResult],
    model_name: str | None = None,
    top_n: int = 10,
    output_file: str | None = None,
):
    """
    Plot question-level scores to identify best and worst performing questions.
    """
    if not results:
        print("No evaluation results to plot.")
        return

    df = pd.DataFrame([r.model_dump() for r in results])
    if df.empty:
        print("No evaluation results to plot.")
        return

    # Average score per question
    question_scores = df.groupby("question_id")["score"].mean().sort_values()

    # Top and bottom N questions
    worst_questions = question_scores.head(top_n)
    best_questions = question_scores.tail(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    worst_questions.plot(kind="barh", ax=ax1, color="salmon")
    ax1.set_title(f"Worst {top_n} Performing Questions")
    ax1.set_xlabel("Average Score")
    ax1.set_ylabel("Question ID")

    best_questions.plot(kind="barh", ax=ax2, color="mediumseagreen")
    ax2.set_title(f"Best {top_n} Performing Questions")
    ax2.set_xlabel("Average Score")
    ax2.set_ylabel("Question ID")

    title = "Question-Level Performance"
    if model_name:
        fig.suptitle(f"{title} - {model_name}", fontsize=16)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")


def plot_evaluator_distribution(
    results: list[EvaluationResult], model_name: str | None = None, output_file: str | None = None
):
    """
    Plot score distribution for each evaluator.
    """
    if not results:
        print("No evaluation results to plot.")
        return

    df = pd.DataFrame([r.model_dump() for r in results])
    if df.empty:
        print("No evaluation results to plot.")
        return

    evaluators = df["evaluator"].unique()
    n_evaluators = len(evaluators)

    if n_evaluators <= 4:
        n_cols = 2
    else:
        n_cols = 3

    n_rows = (n_evaluators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, evaluator in enumerate(sorted(evaluators)):
        if i < len(axes):
            ax = axes[i]
            data = df[df["evaluator"] == evaluator]["score"]

            color = "orangered" if str(evaluator).lower().endswith("llm") else "royalblue"

            ax.hist(data, bins=10, alpha=0.7, color=color)
            ax.set_title(f"Distribution: {evaluator}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")

            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="green", linestyle="--", alpha=0.7, label=f"Median: {median_val:.2f}")
            ax.legend(fontsize=8)

    for i in range(n_evaluators, len(axes)):
        axes[i].set_visible(False)

    title = "Score Distribution by Evaluator"
    if model_name:
        fig.suptitle(f"{title} - {model_name}", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if model_name else 0.98)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")


def plot_model_comparison(
    results_dict: dict[str, list[EvaluationResult]],
    evaluator_filter: str | None = None,
    figsize: tuple = (14, 8),
    output_file: str | None = None,
):
    """
    Plot a comparison of evaluation results across models.
    """
    if not results_dict:
        logger.warning("No results available to plot")
        return

    all_results = []
    for model_name, results in results_dict.items():
        for result in results:
            result_dict = result.model_dump()
            result_dict["model"] = model_name
            all_results.append(result_dict)

    df = pd.DataFrame(all_results)

    if evaluator_filter:
        df = df[df["evaluator"].str.contains(evaluator_filter)]

    # Handle LLM evaluators with 0-10 scale
    df_display = df.copy()
    llm_evaluators = [eval_name for eval_name in df["evaluator"].unique() if str(eval_name).lower().endswith("llm")]

    df_grouped = df_display.groupby(["model", "evaluator"], as_index=False)["score"].mean()

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    g = sns.catplot(
        data=df_grouped,
        kind="bar",
        x="evaluator",
        y="score",
        hue="model",
        palette="viridis",
        height=figsize[1] / 2,
        aspect=figsize[0] / figsize[1] * 2,
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05 if not any(llm_evaluators) else 10.5)

    # Add labels and title
    plt.xlabel("Evaluator")
    plt.ylabel("Score")
    plt.title("Model Comparison by Evaluator")

    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_file}")


def plot_radar_comparison(
    results_dict: dict[str, list[EvaluationResult]],
    top_n_models: int = 5,
    figsize: tuple = (10, 10),
    output_file: str | None = None,
):
    """
    Create a radar plot comparing models across evaluation metrics.
    """
    if not results_dict:
        logger.warning("No results available to plot")
        return

    all_results = []
    for model_name, results in results_dict.items():
        for result in results:
            result_dict = result.model_dump()
            result_dict["model"] = model_name
            all_results.append(result_dict)

    df = pd.DataFrame(all_results)

    model_avg = df.groupby("model")["score"].mean().sort_values(ascending=False)
    top_models = model_avg.head(top_n_models).index.tolist()
    df_filtered = df[df["model"].isin(top_models)]
    radar_data = df_filtered.pivot_table(values="score", index="model", columns="evaluator", aggfunc="mean").fillna(0)

    # Normalize values to 0-1 scale for all evaluators
    for col in radar_data.columns:
        if col.lower().endswith("llm"):
            radar_data[col] = radar_data[col] / 10.0

    fig = plt.figure(figsize=figsize)

    categories = radar_data.columns.tolist()
    N = len(categories)

    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop

    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], categories, size=8)

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot each model
    for i, model in enumerate(radar_data.index):
        values = radar_data.loc[model].values.tolist()
        values += values[:1]  # Close the loop

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=model)
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Comparison Radar Chart", size=15, y=1.1)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved radar plot to {output_file}")
