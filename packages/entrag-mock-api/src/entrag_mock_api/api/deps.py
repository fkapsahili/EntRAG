import json

from entrag_mock_api.config import FINANCE_DATA_PATH, SEC_DATA_PATH


def _load_finance_data() -> dict[str, dict]:
    try:
        with open(FINANCE_DATA_PATH, "r") as f:
            full_data = json.load(f)
            return full_data.get("data", {})
    except Exception as e:
        raise RuntimeError(f"Failed to load finance data: {e}")


def get_finance_data() -> dict[str, dict]:
    if not hasattr(get_finance_data, "cache"):
        get_finance_data.cache = _load_finance_data()
    return get_finance_data.cache


def _load_filings_data() -> dict[str, list[dict]]:
    try:
        with open(SEC_DATA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load filings data: {e}")


def get_filings_data() -> dict[str, list[dict]]:
    if not hasattr(get_filings_data, "cache"):
        get_filings_data.cache = _load_filings_data()
    return get_filings_data.cache
