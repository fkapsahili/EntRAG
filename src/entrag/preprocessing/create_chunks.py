import json
import os
import re

import nltk
import torch
from loguru import logger
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, GPT2Tokenizer

from entrag.config.evaluation_config import ChunkingConfig, EvaluationConfig
from entrag.data_model.document import Chunk
from entrag.preprocessing.create_dataset import create_dataset


# Block multithreading for PyTorch to avoid memory issues
torch.set_num_threads(1)

# Download NLTK data
nltk.download("punkt_tab")


def _clean_text(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    # Extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Quotes
    text = re.sub(r"[" '"]', '"', text)
    # Newlines
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _create_chunk_boundaries(sentences: list[str], sentence_embeddings, chunking_config: ChunkingConfig) -> list:
    """
    Create chunk boundaries based on semantic similarity.
    """
    cos_similarities = [
        cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
        for i in range(len(sentence_embeddings) - 1)
    ]

    boundaries = [0]
    for i, sim in enumerate(cos_similarities):
        if sim < chunking_config.similarity_threshold:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))
    return boundaries


def split_document_in_pages(document_text: str) -> list[tuple[int, str]]:
    pages = re.split(r"(##PAGE \d+##)", document_text)
    page_texts = []
    current_page = None

    for section in pages:
        match = re.match(r"##PAGE (\d+)##", section)
        if match:
            current_page = int(match.group(1))
        else:
            if section.strip():  # Ignore empty pages
                page_texts.append((current_page, section.strip()))
    return page_texts  # list of (page_number, page_content)


def _process_sentences_segment(
    sentences: list,
    start: int,
    end: int,
    tokenizer,
    chunking_config: ChunkingConfig,
    current_chunk_sentences: list,
    current_token_count: int,
) -> tuple:
    """
    Process a segment of sentences and create chunks.
    """
    segment_sentences = sentences[start:end]
    segment_text = " ".join(segment_sentences)
    segment_tokens = tokenizer.tokenize(segment_text)
    segment_num_tokens = len(segment_tokens)
    chunks = []
    overlap_size = chunking_config.overlap_size

    if current_token_count + segment_num_tokens <= chunking_config.max_tokens:
        current_chunk_sentences.extend(segment_sentences)
        current_token_count += segment_num_tokens
        if current_token_count >= chunking_config.target_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            # Keep the last n sentences for overlap
            current_chunk_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
            current_token_count = len(tokenizer.tokenize(" ".join(current_chunk_sentences)))
    else:
        if current_token_count >= chunking_config.min_tokens:
            chunks.append(" ".join(current_chunk_sentences))
            # Keep the last n sentences for overlap
            overlap_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
            current_chunk_sentences = overlap_sentences + segment_sentences
            current_token_count = len(tokenizer.tokenize(" ".join(current_chunk_sentences)))
        else:
            current_chunk_sentences.extend(segment_sentences)
            current_token_count += segment_num_tokens
            if current_token_count >= chunking_config.min_tokens or current_token_count >= chunking_config.max_tokens:
                chunks.append(" ".join(current_chunk_sentences))
                # Keep the last n sentences for overlap
                current_chunk_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
                current_token_count = len(tokenizer.tokenize(" ".join(current_chunk_sentences)))

    return chunks, current_chunk_sentences, current_token_count


def _save_document_chunks(config: EvaluationConfig, chunks: list[Chunk]):
    """
    Save the created document chunks to disk.
    """
    output_file = os.path.join(config.chunking.output_directory, config.chunking.dataset_name + ".jsonl")
    logger.debug(f"Saving document chunks to: {output_file}")
    os.makedirs(config.chunking.output_directory, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    logger.debug("Document chunks saved successfully.")


def _load_document_chunks(config: ChunkingConfig) -> list[Chunk]:
    """
    Load the document chunks from disk.
    """
    input_file = os.path.join(config.output_directory, config.dataset_name + ".jsonl")
    logger.debug(f"Loading document chunks from: {input_file}")
    chunks = []
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(Chunk.model_validate_json(line))
    logger.debug("Document chunks loaded successfully.")
    return chunks


def semantic_chunking(
    document_page: str, chunking_config: ChunkingConfig, model: SentenceTransformer, tokenizer: AutoTokenizer
):
    """
    Create chunks based on semantic similarity.
    """
    document = _clean_text(document_page)
    sentences = [s.strip() for s in sent_tokenize(document) if s.strip()]

    if not sentences:
        logger.warning("No sentences found in the document after cleaning.")
        return []

    logger.debug(f"Processing document page with {len(sentences)} sentences.")
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    chunk_boundaries = _create_chunk_boundaries(sentences, sentence_embeddings, chunking_config)
    logger.debug(f"Found {len(chunk_boundaries) - 1} chunk boundaries.")

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        chunk, current_chunk_sentences, current_token_count = _process_sentences_segment(
            sentences, start, end, tokenizer, chunking_config, current_chunk_sentences, current_token_count
        )
        chunks.extend(chunk)

    if current_chunk_sentences:
        if current_token_count >= chunking_config.min_tokens:
            chunks.append(" ".join(current_chunk_sentences))
        else:
            if chunks:
                chunks[-1] += " " + " ".join(current_chunk_sentences)
            else:
                chunks.append(" ".join(current_chunk_sentences))

    return chunks


def create_chunks_for_documents(config: EvaluationConfig) -> list[Chunk]:
    """
    Create chunks for the documents in the dataset.
    """
    logger.info("Starting document chunking.")

    if not config.chunking.enabled:
        logger.info("Chunking is disabled. Loading existing chunks.")
        return _load_document_chunks(config.chunking)

    source_directory = config.chunking.files_directory
    logger.debug(f"Using chunking configuration: {config.chunking}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {config.chunking.model_name}")
    model = SentenceTransformer(config.chunking.model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(config.chunking.model_name, use_fast=True, model_max_length=512)
    logger.debug("Model and tokenizer loaded successfully.")

    logger.debug("Initializing GPT2 tokenizer.")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    chunks_list = []
    logger.info(f"Loading documents from: {source_directory}")
    dataset = create_dataset(config)
    for index, document in enumerate(dataset):
        logger.debug(f"Processing document {index + 1}/{len(dataset)}.")
        pages = split_document_in_pages(document.document_text)

        for page_number, page_text in pages:
            chunks = semantic_chunking(page_text, config.chunking, model, tokenizer)
            logger.debug(f"Found {len(chunks)} chunks for the document.")

            chunks_formatted = [
                Chunk(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    document_page=page_number,
                    chunk_location_id=i,
                    chunk_text=chunk,
                    chunk_length_tokens=len(gpt2_tokenizer.encode(chunk)),
                )
                for i, chunk in enumerate(chunks)
            ]
            chunks_list.extend(chunks_formatted)

    logger.info(f"Document chunking completed. Created {len(chunks_list)} chunks.")
    _save_document_chunks(config, chunks_list)
    logger.success("Document chunking completed successfully.")
    return chunks_list
