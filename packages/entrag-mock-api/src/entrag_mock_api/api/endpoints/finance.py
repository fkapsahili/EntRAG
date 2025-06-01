from fastapi import APIRouter, Depends, HTTPException, Query

from entrag_mock_api.api.deps import get_finance_data
from entrag_mock_api.schema import CompanyMetrics, MetricResponse, TimeseriesResponse


router = APIRouter(tags=["Finance"])


def find_company(ticker_or_name: str, data: dict):
    ticker_or_name = ticker_or_name.lower()
    for ticker, details in data.items():
        if ticker.lower() == ticker_or_name or ticker_or_name in details.get("company_name", "").lower():
            return ticker, details
    return None, None


@router.get("/company/{ticker}", response_model=CompanyMetrics)
def get_company_metrics(ticker: str, data=Depends(get_finance_data)):
    ticker, company = find_company(ticker, data)
    if not company:
        raise HTTPException(status_code=404, detail="Ticker not found")
    return company.get("metrics", {})


@router.get("/company/{ticker}/metric", response_model=MetricResponse)
def get_metric_by_date(
    ticker: str, metric: str = Query(..., enum=["eps"]), date: str = Query(...), data=Depends(get_finance_data)
):
    ticker, company = find_company(ticker, data)
    if not company:
        raise HTTPException(status_code=404, detail="Ticker not found")

    if metric != "eps":
        raise HTTPException(status_code=400, detail="Unsupported metric. Only 'eps' is available.")

    historical_eps = company.get("historical_eps", {})
    value = historical_eps.get(date)

    if value is None:
        raise HTTPException(status_code=404, detail="No data for given date")

    return MetricResponse(ticker=ticker, metric=metric, value=value, date=date)


@router.get("/company/{ticker}/timeseries", response_model=TimeseriesResponse)
def get_timeseries(
    ticker: str,
    metric: str = Query("eps", enum=["eps"]),
    start: str | None = None,
    end: str | None = None,
    data=Depends(get_finance_data),
):
    ticker, company = find_company(ticker, data)
    if not company:
        raise HTTPException(status_code=404, detail="Ticker not found")

    if metric != "eps":
        raise HTTPException(status_code=400, detail="Unsupported metric. Only 'eps' is available.")

    historical_eps = company.get("historical_eps", {})
    results = {
        date: value
        for date, value in historical_eps.items()
        if (not start or date >= start) and (not end or date <= end)
    }

    return TimeseriesResponse(ticker=ticker, metric=metric, results=results)
