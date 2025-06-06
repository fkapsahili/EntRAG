import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from entrag.data_model.question_answer import EvaluationResult


def create_run_id(custom_id: str | None = None) -> str:
    """
    Create a unique run ID for an evaluation run.
    """
    if custom_id:
        return custom_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_run_{timestamp}"


def save_model_results(model_name: str, results: list[EvaluationResult], output_dir: str | Path) -> Path:
    """
    Save evaluation results for a specific model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    results_data = [r.model_dump() for r in results]

    for result in results_data:
        result["model"] = model_name

    file_path = output_path / f"{model_name}_results.json"
    with open(file_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Saved {len(results)} evaluation results for model '{model_name}' to {file_path}")

    return file_path


def save_combined_results(
    results_dict: dict[str, list[EvaluationResult]], output_dir: str | Path
) -> tuple[Path, Path]:
    """
    Save combined results from multiple models.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    all_results = []
    for model_name, results in results_dict.items():
        for result in results:
            result_dict = result.model_dump()
            result_dict["model"] = model_name
            all_results.append(result_dict)

    df = pd.DataFrame(all_results)

    csv_path = output_path / "all_results.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_path / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Saved combined results from {len(results_dict)} models to {csv_path} and {json_path}")

    return csv_path, json_path


def load_results(file_path: str | Path) -> pd.DataFrame:
    """
    Load evaluation results from a file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded evaluation results from {file_path}")
    return df
