import re
from loguru import logger

import nltk
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK data
nltk.download("punkt_tab")


def _create_chunk_boundaries(sentences: list[str], sentence_embeddings)

def _clean_text(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    # Extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Quotes
    text = re.sub(r"[" '"]', '"', text)
    # Newlines
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def create_chunks_for_documents():
    logger.info("Starting document chunking")
