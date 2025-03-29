from pathlib import Path

import yaml

from entrag.config.evaluation_config import EvaluationConfig


def get_project_root() -> Path:
    """
    Get the project root directory.
    """
    current_dir = Path(__file__).resolve()
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").is_file():
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError("Could not find the project root directory (pyproject.toml not found).")


def load_eval_config(config_name: str) -> EvaluationConfig:
    """
    Load the evaluation configuration from the specified file.
    """
    config_path = get_project_root() / "evaluation_configs" / config_name / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError("Failed to parse config file: {config_path}") from e

    return EvaluationConfig(**config)
