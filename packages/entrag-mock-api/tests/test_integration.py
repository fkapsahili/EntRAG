import pytest
import requests

from entrag_mock_api.client.api_client import MockAPIClient
from entrag_mock_api.client.errors import MockAPIConnectionError, MockAPINotFoundError


@pytest.fixture
def client() -> MockAPIClient:
    """
    API client that assumes the mock api server is running on localhost:8000.
    """
    return MockAPIClient()


def test_server_is_running(client: MockAPIClient) -> None:
    """
    Verify the mock API server is working as expected.
    """
    try:
        requests.get(f"{client.base_url}/")
    except requests.ConnectionError:
        pytest.skip("Mock API server not running. Start with: uv run start")


def test_client_initialization() -> None:
    client = MockAPIClient()
    assert client.base_url == "http://localhost:8000/api"
    assert client.timeout == 30


def test_get_finance_company_metrics(client: MockAPIClient) -> None:
    result = client.get_finance_company_metrics("AAPL")
    assert result is not None


def test_get_finance_metric_by_date(client: MockAPIClient) -> None:
    result = client.get_finance_metric_by_date("AAPL", "eps", "2024-09-30")
    assert result is not None


def test_get_finance_timeseries(client: MockAPIClient) -> None:
    result = client.get_finance_timeseries("AAPL", "eps")
    assert result is not None


def test_get_filings_by_type(client: MockAPIClient) -> None:
    result = client.get_filings_by_type("10-K")
    assert result is not None
    assert "filings" in result


def test_search_filings(client: MockAPIClient) -> None:
    result = client.search_filings(filing_type="10-K")
    assert result is not None
    assert "results" in result


def test_get_gpg_statistics(client: MockAPIClient) -> None:
    result = client.get_gpg_statistics("Google")
    assert result is not None


def test_search_websites(client: MockAPIClient) -> None:
    result = client.search_websites("Apple")
    assert result is not None


def test_get_company_filings(client: MockAPIClient) -> None:
    result = client.get_company_filings("Apple", ["8-K", "10-Q"])
    assert isinstance(result, list)


def test_get_latest_filing(client: MockAPIClient) -> None:
    with pytest.raises(MockAPINotFoundError):
        client.get_latest_filing("Apple", "10-K")


def test_404_error_handling(client: MockAPIClient) -> None:
    with pytest.raises(MockAPINotFoundError):
        client.get_finance_company_metrics("INVALID_TICKER_XYZ")


def test_connection_error_handling() -> None:
    offline_client = MockAPIClient("http://localhost:9999/api")
    with pytest.raises(MockAPIConnectionError):
        offline_client.get_finance_company_metrics("AAPL")
