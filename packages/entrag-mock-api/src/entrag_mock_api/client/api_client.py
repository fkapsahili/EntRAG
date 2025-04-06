import requests

from entrag_mock_api.schema import CompanyMetrics, MetricResponse, TimeseriesResponse


class MockAPIClient:
    """
    A mock API client for interacting with the EntRAG mock API.
    """

    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url

    def _get(self, path: str, params: dict = None) -> dict:
        response = requests.get(f"{self.base_url}/{path}", params=params)
        response.raise_for_status()
        return response.json()

    def get_finance_company_metrics(self, ticker: str) -> CompanyMetrics:
        data = self._get(f"finance/company/{ticker}")
        return CompanyMetrics.model_validate(data)

    def get_finance_metric_by_date(self, ticker: str, metric: str, date: str) -> MetricResponse:
        data = self._get(f"finance/company/{ticker}/metric", params={"metric": metric, "date": date})
        return MetricResponse.model_validate(data)

    def get_finance_timeseries(
        self, ticker: str, metric: str = "eps", start: str | None = None, end: str | None = None
    ) -> TimeseriesResponse:
        params = {"metric": metric}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        data = self._get(f"finance/company/{ticker}/timeseries", params=params)
        return TimeseriesResponse.model_validate(data)
