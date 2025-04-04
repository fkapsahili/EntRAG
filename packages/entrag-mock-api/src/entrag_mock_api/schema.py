from pydantic import BaseModel


class CompanyMetrics(BaseModel):
    eps: float | None = None
    pe_ratio: float | None = None
    dividend: float | None = None
    market_cap: float | None = None


class MetricResponse(BaseModel):
    ticker: str
    metric: str
    value: float
    date: str


class TimeseriesResponse(BaseModel):
    ticker: str
    metric: str
    results: dict[str, float]
