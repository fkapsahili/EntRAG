import json
import os


def _load_finance_data() -> dict[str, dict]:
    data_path = os.path.join(os.path.dirname(__file__), "../../datasets/finance_data.json")
    try:
        with open(data_path, "r") as f:
            full_data = json.load(f)
            return full_data.get("data", {})
    except Exception as e:
        raise RuntimeError(f"Failed to load finance data: {e}")


def get_finance_data():
    if not hasattr(get_finance_data, "cache"):
        get_finance_data.cache = _load_finance_data()
    return get_finance_data.cache
