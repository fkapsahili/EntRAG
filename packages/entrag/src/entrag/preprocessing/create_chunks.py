import json
import os
import re

import nltk
from loguru import logger
from transformers import GPT2Tokenizer

from entrag.config.evaluation_config import ChunkingConfig, EvaluationConfig
from entrag.data_model.document import Chunk
from entrag.preprocessing.create_dataset import create_dataset


# Download NLTK data
nltk.download("punkt_tab")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def load_document_chunks(config: ChunkingConfig) -> list[Chunk]:
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


def create_chunks_for_documents(config: EvaluationConfig) -> list[Chunk]:
    """
    Create chunks for the documents in the dataset.
    """
    logger.info("Starting document chunking.")

    if not config.chunking.enabled:
        logger.info("Chunking is disabled. Loading existing chunks.")
        return load_document_chunks(config.chunking)

    source_directory = config.chunking.files_directory
    logger.debug(f"Using chunking configuration: {config.chunking}")

    logger.debug("Initializing GPT2 tokenizer.")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    max_tokens = config.chunking.max_tokens
    chunks_list = []
    logger.info(f"Loading documents from: {source_directory}")
    dataset = create_dataset(config)
    for index, document in enumerate(dataset):
        logger.debug(f"Processing document [{document.document_name}] {index + 1}/{len(dataset)}.")
        pages = split_document_in_pages(document.document_text)
        logger.debug(f"Found {len(pages)} pages in document [{index + 1}].")

        for page_number, page_text in pages:
            logger.debug(f"Processing page [{page_number}] of document [{index + 1}].")
            cleaned_page_text = _clean_text(page_text)
            tokens = gpt2_tokenizer.encode(cleaned_page_text)
            logger.debug(f"Page token length: {len(tokens)}")
            num_chunks = (len(tokens) + max_tokens - 1) // max_tokens
            logger.debug(f"Found {num_chunks} chunks for page [{page_number}].")

            for chunk_id in range(num_chunks):
                start = chunk_id * max_tokens
                end = start + max_tokens
                chunk_tokens = tokens[start:end]
                chunk_text = gpt2_tokenizer.decode(chunk_tokens)

                chunks_list.append(
                    Chunk(
                        document_id=document.document_id,
                        document_name=document.document_name,
                        document_page=page_number,
                        chunk_location_id=chunk_id,
                        chunk_text=chunk_text,
                        chunk_length_tokens=len(chunk_tokens),
                    )
                )

    logger.info(f"Document chunking completed. Created {len(chunks_list)} chunks.")
    _save_document_chunks(config, chunks_list)

    logger.success("Document chunking completed successfully.")
    return chunks_list
