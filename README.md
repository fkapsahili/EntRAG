# EntRAG Benchmark

EntRAG is a benchmark for evaluating the performance of Retrieval-Augmented Generation (RAG) systems in an enterprise context. The benchmark is designed to provide a comprehensive evaluation of RAG systems on a heterogeneous corpus of documents across various enterprise-like domains. The benchmark includes a QA datset of 100 handcrafted questions, that are used for the evaluation.

## Installation

The project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for the project management. Make sure you have it installed on your local development machine.

## Poe Tasks

The project uses [Poe](https://poethepoet.natn.io/) for task management. The tasks are defined in the `pyproject.toml` file. You can run the tasks using the `uv run poe <task-name>` command.

### Available Poe Tasks

- `check`: Check the codebase for linting and formatting errors with [Ruff](https://docs.astral.sh/ruff/) 
- `style`: Apply style fixes to the codebase with [Ruff](https://docs.astral.sh/ruff/)
- `style-all`: Format and check the entire codebase for linting and formatting errors
- `quality`: Check the quality of the entire codebase

## Quickstart

1. Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/fkapsahili/EntRAG.git
cd EntRAG
```

2. Install the dependencies using `uv`:
```bash
uv sync --all-groups
```

3. Get an API key for OpenAI or Gemini and add it to the .env file
```bash
touch .env
echo "OPENAI_API_KEY=<your_openai_api_key>" >> .env 
echo "GEMINI_API_KEY=<your_gemini_api_key>" >> .env
```

4. Navigate to the mock API package
```bash
cd packages/entrag-mock-api

# Start the mock API server
uv run poe start

# The server will run on http://localhost:8000
# Keep this terminal open while running the benchmark
```

5. Run the benchmark with the default config in a new terminal:
```bash
# Navigate to the benchmark package
cd packages/entrag

# Run the pipeline
poe entrag --config example/configs/default.yaml 
```

 ### Data Processing

 #### Using Pre-processed Data

 If you have access to the pre-processed markdown data, place it in the `data/entrag_processed/` directory (or any directory specified as `chunking.files_directory` in your config file) and the benchmark will use it directly.


#### Preprocessing Raw Documents

If you have the raw documents that need to be processed, you can use the standalone processing script before running the benchmark. This script will process the documents and save them in the `data/entrag_processed/` directory.

```bash
cd packages/entrag
uv run poe create-markdown-documents.py \
    --input-dir <path_to_raw_documents> \
    --output-dir data/entrag_processed/ \
    --workers <number_of_workers> \
    --batch-size <batch_size>
```

**Note**: The processing script uses [Docling](https://github.com/DS4SD/docling) for Markdown conversion and requires significant computational resources if running multiple workers. Ensure you have enough memory and CPU resources available.

### Configuration

#### Using Existing Configurations

The benchmark includes a default configuration located at `evaluation_configs/default/config.yaml`. You can run the benchmark with this configuration:

```bash
uv run poe entrag --config default
```

The results will then be saved in the `evaluation_results/` directory.

#### Creating Custom Configurations

To create a custom configuration for your specific evaluation needs:

1. **Create a new configuration directory:**
```bash
mkdir evaluation_configs/my-custom-config
```

2. **Create the configuration file:**
```bash
touch evaluation_configs/my-custom-config/config.yaml
```

3. **Configure your evaluation settings** by editing the `config.yaml` file. Here's a template with explanations:

```yaml
config_name: my-custom-config
tasks:
  question_answering:
    run: true                           # Enable/disable QA evaluation
    hf_dataset_id: fkapsahili/EntRAG    # HuggingFace dataset ID
    dataset_path: null                  # Alternative: local dataset path
    split: train                        # Dataset split to use

chunking:
  enabled: true                          # Enable document chunking
  files_directory: data/entrag_processed # Input directory for processed docs
  output_directory: data/entrag_chunked  # Output directory for chunks
  dataset_name: entrag                   # Dataset identifier
  max_tokens: 2048                       # Maximum tokens per chunk

embedding:
  enabled: true                         # Enable embedding generation
  model: text-embedding-3-small         # Embedding model to use
  batch_size: 8                         # Batch size for embedding generation
  output_directory: data/embeddings     # Output directory for embeddings

model_evaluation:
  max_workers: 10                       # Parallel workers for LLM inference
  output_directory: evaluation_results  # Results output directory
  retrieval_top_k: 5                    # Number of documents to retrieve
  model_provider: openai                # AI provider: "openai" or "gemini"
  model_name: gpt-4o-mini               # Model name for evaluation
  reranking_model_name: gpt-4o-mini     # Model for reranking (if applicable)
```

4. **Run your custom configuration:**
```bash
uv run poe entrag --config my-custom-config
```

### Configuration Options

**Task Configuration:**
- `question_answering.run`: Whether to run QA evaluation
- `hf_dataset_id`: HuggingFace dataset containing evaluation questions
- `split`: Which dataset split to use for evaluation

**Chunking Configuration:**
- `files_directory`: Directory containing processed markdown documents
- `max_tokens`: Maximum size for document chunks
- `dataset_name`: Identifier for the dataset being processed

**Model Configuration:**
- `model_provider`: Choose between `openai` or `gemini`
- `model_name`: Specific model to use for evaluation
- `retrieval_top_k`: Number of relevant documents to retrieve for each question

**Available Models:**
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`
- **Gemini**: `gemini-2.0-pro`, `gemini-2.0-flash`

## Extending the Benchmark

### Custom AI Providers

You can add support for new AI providers (Claude, Ollama, local models, etc.) by implementing the `BaseAIEngine` interface:

1. **Create your custom AI engine** by extending `BaseAIEngine`
2. **Implement the `chat_completion` method** with your provider's API logic
3. **Add your provider to the `create_ai_engine` function** in `main.py`
4. **Update your configuration** to use the new provider

**Example configuration:**
```yaml
model_evaluation:
  model_provider: custom_provider
  model_name: your-custom-model
```

### Custom RAG Implementations

You can implement novel RAG approaches by extending the `RAGLM` base class:

1. **Create your custom RAG class** by extending `RAGLM`
2. **Implement the required methods:**
   - `build_store()`: Build your retrieval index/database
   - `retrieve()`: Implement your retrieval logic
   - `generate()`: Define your response generation strategy
3. **Add your model to the evaluation pipeline** in `main.py`

### Example Extensions

**Popular extensions you might implement:**
- **Graph-based RAG**: Using knowledge graphs for retrieval expansion
- **Agentic RAG**: RAG systems that can use tools and APIs
- **Hierarchical RAG**: Multi-level document organization and retrieval

The benchmark will automatically evaluate your custom implementations alongside the built-in models, providing standardized metrics for comparison.

## Getting the EntRAG Document Dataset
To access the complete EntRAG document dataset for evaluation:

1. Contact for dataset access: Send an [email](mailto:fabio.kapsahili@protonmail.com) requesting access to the EntRAG benchmark dataset
2. Include in your request:
- Your research affiliation
- Intended use case for the benchmark
- Whether you need raw documents or pre-processed markdown files

3. Available data formats:
- Pre-processed markdown: Ready-to-use documents for immediate benchmarking
- Raw PDF documents: Original source files if you want to test custom preprocessing
- Mock API datasets: Data files required for the mock API backend to simulate dynamic API queries

The QA dataset is available on HuggingFace at [fkapsahili/EntRAG](https://huggingface.co/datasets/fkapsahili/EntRAG) and will be automatically downloaded during benchmark execution.

## Packages
The project uses a Monorepo which is structured into several Python packages.

- `entrag`: The main benchmark package, which contains the code for the pipeline and the evaluation.
- `entrag-mock-api`: A mock API for the benchmark, which is used for the evaluation of dynamic questions, that require a call to an external API.
