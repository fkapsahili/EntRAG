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


class GPGStatistic(BaseModel):
    EmployerName: str
    EmployerId: int
    Address: str
    PostCode: str
    CompanyNumber: str
    DiffMeanHourlyPercent: float
    DiffMedianHourlyPercent: float
    DiffMedianHourlyPercent: float
    DiffMedianBonusPercent: float
    MaleBonusPercent: float
    FemaleBonusPercent: float
    MaleLowerQuartile: float
    FemaleLowerQuartile: float
    MaleLowerMiddleQuartile: float
    FemaleUpperMiddleQuartile: float
    MaleTopQuartile: float
    FemaleTopQuartile: float
    CurrentName: str


class GPGStatisticsResponse(BaseModel):
    statistics: list[GPGStatistic]
    total_count: int


class WebsiteResult(BaseModel):
    title: str
    page_result: str


class WebsiteSearchResponse(BaseModel):
    results: list[WebsiteResult]
