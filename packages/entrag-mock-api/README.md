# EntRAG Mock API

The EntRAG Mock API is a mock server designed to simulate the behavior of the EntRAG API. It provides endpoints for retrieving financial data, SEC filings, and GPG statistics. 

## Quickstart

1. Run the data collection scripts to build the mock dataset:

```bash
python scripts/finance-data-collection.py --include-history
python scripts/sec-data-collection.py
```

2. Run the mock API server:

```bash
uv run start
```
3. Access the mock API at `http://localhost:8000`.

## Available API Endpoints

### `GET /api/finance/company/<ticker>`

Retrieve the latest financial data for a specific company.

| Parameter | Type   | Description                | Required |
|-----------|--------|---------------------------|----------|
| ticker    | string | Company ticker symbol     | Yes      |

---

### `GET /api/finance/company/<ticker>/metric`

Retrieve a specific financial metric for a company.

| Parameter | Type   | Description                | Required |
|-----------|--------|---------------------------|----------|
| ticker    | string | Company ticker symbol     | Yes      |
| metric    | string | Financial metric name     | Yes      |

---

### `GET /api/finance/company/<ticker>/timeseries`

Retrieve time series data for a specific financial metric.

| Parameter | Type   | Description                | Required |
|-----------|--------|---------------------------|----------|
| ticker    | string | Company ticker symbol     | Yes      |
| metric    | string | Financial metric name     | Yes      |
| start     | string | Start date (YYYY-MM-DD)   | No       |
| end       | string | End date (YYYY-MM-DD)     | No       |

---

### `GET /api/filings/type/<filing_type>`

Retrieve the latest SEC filing data for a specific filing type.

| Parameter   | Type   | Description           | Required |
|-------------|--------|----------------------|----------|
| filing_type | string | SEC filing type      | Yes      |

---

### `GET /api/filings/search`

Search for SEC filings by type and company.

| Parameter   | Type   | Description           | Required |
|-------------|--------|----------------------|----------|
| filing_type | string | SEC filing type      | Yes      |
| company     | string | Company name         | Yes      |

---

### `GET /api/gpg-statistics`

Retrieve GPG (Gender Pay Gap) statistics for a specific employer.

| Parameter      | Type   | Description           | Required |
|----------------|--------|----------------------|----------|
| employer_name  | string | Employer name        | Yes      |

---

### `GET /api/search`

Search for website content using a query string.

| Parameter | Type   | Description           | Required |
|-----------|--------|----------------------|----------|
| query     | string | Search query         | Yes      |