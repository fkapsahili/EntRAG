from datetime import date

from pydantic import BaseModel, HttpUrl


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
    DiffMeanHourlyPercent: float | None
    DiffMedianHourlyPercent: float | None
    DiffMedianHourlyPercent: float | None
    DiffMedianBonusPercent: float | None
    MaleBonusPercent: float | None
    FemaleBonusPercent: float | None
    MaleLowerQuartile: float | None
    FemaleLowerQuartile: float | None
    MaleLowerMiddleQuartile: float | None
    FemaleUpperMiddleQuartile: float | None
    MaleTopQuartile: float | None
    FemaleTopQuartile: float | None
    CurrentName: str


class GPGStatisticsResponse(BaseModel):
    statistics: list[GPGStatistic]
    total_count: int
    year: str


class WebsiteResult(BaseModel):
    title: str
    page_result: str


class WebsiteSearchResponse(BaseModel):
    results: list[WebsiteResult]


class SECFiling(BaseModel):
    company: str
    cik: str
    filing_type: str
    filing_date: date
    accession_number: str
    primary_document: str
    document_url: HttpUrl


class FilingsByTypeResponse(BaseModel):
    filings: list[SECFiling]


class FilingSearchResponse(BaseModel):
    results: list[SECFiling]
