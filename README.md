# EntRAG Benchmark

EntRAG is a benchmark for evaluating the performance of Retrieval-Augmented Generation (RAG) systems in an enterprise context. The benchmark is designed to provide a comprehensive evaluation of RAG systems on a heterogeneous corpus of documents across various enterprise-like domains. The benchmark includes a QA datset of 100 handcrafted questions, that are used for the evaluation.

## Installation

The project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies. Make sure you have it installed on your local development machine.

## Quickstart

1. Clone the repository and navigate to the root directory:
```bash
git clone https://https://github.com/fkapsahili/EntRAG.git
cd EntRAG
```

2. Install the dependencies using `uv`:
```bash
uv sync
```

3. Get an API key for OpenAI and Gemini and add it to the .env file
```bash
touch .env
echo "OPENAI_API_KEY=<your_openai_api_key>" >> .env 
echo "GEMINI_API_KEY=<your_gemini_api_key>" >> .env
```

4. Run the pipeline with the default config
```bash
uv run entrag --config example/configs/default.yaml 
```