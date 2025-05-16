import json

import pandas as pd

from entrag_mock_api.config import FINANCE_DATA_PATH, GPG_DATA_PATH, SEC_DATA_PATH, WEBSITES_PATH
from entrag_mock_api.schema import GPGStatistic, WebsiteResult


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


def _load_gpg_statistics_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]

    model_fields = GPGStatistic.model_fields.keys()
    return df[[col for col in model_fields if col in df.columns]]


def get_gpg_statistics_data() -> pd.DataFrame:
    if not hasattr(get_gpg_statistics_data, "cache"):
        data = _load_gpg_statistics_data(GPG_DATA_PATH)
        get_gpg_statistics_data.cache = data
    return get_gpg_statistics_data.cache


def _load_websites_data(websites_dir: str) -> list[WebsiteResult]:
    try:
        websites: list[WebsiteResult] = []
        for path in websites_dir.glob("*.html"):
            with open(path, "r") as f:
                websites.append(
                    WebsiteResult(
                        title=str(path).split("/")[-1],
                        content=f.read(),
                    )
                )
        return websites
    except Exception as e:
        raise RuntimeError(f"Failed to load websites data: {e}") from e
    return []


def get_websites_data() -> list[WebsiteResult]:
    if not hasattr(get_websites_data, "cache"):
        get_websites_data.cache = _load_websites_data(WEBSITES_PATH)
    return get_websites_data.cache
