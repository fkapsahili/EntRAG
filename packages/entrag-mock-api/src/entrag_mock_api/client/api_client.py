from typing import Any

import requests
from loguru import logger
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from entrag_mock_api.client.errors import (
    MockAPIConnectionError,
    MockAPIError,
    MockAPINotFoundError,
)
from entrag_mock_api.schema import (
    CompanyMetrics,
    GPGStatisticsResponse,
    MetricResponse,
    TimeseriesResponse,
    WebsiteSearchResponse,
)


class MockAPIClient:
    """
    API client for communication with the EntRAG mock API.
    """

    def __init__(self, base_url: str = "http://localhost:8000/api", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = max(0.1, int(timeout))

    def _get(self, path: str, params: dict = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except (ConnectionError, Timeout) as e:
            logger.error(f"Network error for {url}: {e}")
            raise MockAPIConnectionError(f"Cannot reach API: {e}") from e

        except HTTPError as e:
            status_code = e.response.status_code
            error_detail = self._extract_error_detail(e.response)

            if status_code == 404:
                raise MockAPINotFoundError(error_detail) from e
            elif status_code >= 500:
                logger.error(f"Server error {status_code}: {error_detail}")
                raise MockAPIError(f"Server error: {error_detail}") from e
            else:
                logger.error(f"Client error {status_code}: {error_detail}")
                raise MockAPIError(f"API error: {error_detail}") from e

        except (RequestException, ValueError) as e:
            logger.error(f"Request failed for {url}: {e}")
            raise MockAPIError(f"Request failed: {e}") from e

    def _extract_error_detail(self, response: requests.Response) -> str:
        try:
            return response.json().get("detail", f"HTTP {response.status_code}")
        except (ValueError, AttributeError):
            return f"HTTP {response.status_code}"

    def get_finance_company_metrics(self, ticker: str) -> CompanyMetrics:
        """
        Get financial metrics for a company by its ticker symbol.
        """
        data = self._get(f"finance/company/{ticker}")
        return CompanyMetrics.model_validate(data)

    def get_finance_metric_by_date(self, ticker: str, metric: str, date: str) -> MetricResponse:
        """
        Get a specific financial metric for a company on a given date.
        """
        data = self._get(f"finance/company/{ticker}/metric", params={"metric": metric, "date": date})
        return MetricResponse.model_validate(data)

    def get_finance_timeseries(
        self, ticker: str, metric: str = "eps", start: str | None = None, end: str | None = None
    ) -> TimeseriesResponse:
        """
        Get a timeseries of financial metrics for a company.
        """
        params = {"metric": metric}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        data = self._get(f"finance/company/{ticker}/timeseries", params=params)
        return TimeseriesResponse.model_validate(data)

    def get_filings_by_type(self, filing_type: str, company: str | None = None) -> dict:
        """
        Get filings by filing type and filter optionally by company.
        """
        params = {}
        if company:
            params["company"] = company
        return self._get(f"filings/type/{filing_type}", params=params)

    def search_filings(self, filing_type: str | None = None, company: str | None = None) -> dict:
        """
        Search for SEC filings by type and/or company.
        """
        params = {}
        if filing_type:
            params["filing_type"] = filing_type
        if company:
            params["company"] = company
        return self._get("filings/search", params=params)

    def get_gpg_statistics(self, employer_name: str) -> GPGStatisticsResponse:
        """
        Get Gender Pay Gap statistics for a specified company/employer.
        """
        data = self._get("gpg-statistics", params={"employer_name": employer_name})
        return GPGStatisticsResponse.model_validate(data)

    def search_websites(self, query: str) -> WebsiteSearchResponse:
        """
        Search for website content using a query string.
        """
        data = self._get("search", params={"query": query})
        return WebsiteSearchResponse.model_validate(data)

    def get_company_filings(self, company_name: str, filing_types: list[str] | None = None) -> list[dict]:
        """
        Get all filings for a specific company, optionally filtered by filing types.
        """
        all_filings = []

        if filing_types:
            for filing_type in filing_types:
                try:
                    result = self.get_filings_by_type(filing_type, company_name)
                    all_filings.extend(result.get("filings", []))
                except requests.HTTPError:
                    continue
        else:
            result = self.search_filings(company=company_name)
            all_filings = result.get("results", [])

        return all_filings

    def get_latest_filing(self, company_name: str, filing_type: str) -> dict:
        """
        Get the most recent filing of a specific type for a company.
        """
        try:
            result = self.get_filings_by_type(filing_type, company_name)
            filings = result.get("filings", [])
            return filings[0] if filings else None
        except requests.HTTPError:
            return None
