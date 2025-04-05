# EntRAG Mock API

## Quickstart

1. Run the finance data collection script to build the mock dataset:

```bash
python scripts/finance-data-collection.py --include-history
```

2. Run the mock API server:

```bash
uv run start
```
3. Access the mock API at `http://localhost:8000/api/`.