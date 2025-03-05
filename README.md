# EntRAG

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the development environment.

```bash
uv sync
```

## Quickstart

1. Run the document ingestion to parse the documents and store them as markdown:

```bash
python scripts/document-ingestion.py --input-dir data/raw --output-dir data/processed
```

