"""
This script processes raw PDF files from a specified input directory and saves the structured data as JSON.
"""

import json
import os
import re
import unicodedata
from typing import List, Optional

import click
import pymupdf
from loguru import logger
from pydantic import BaseModel


class Document(BaseModel):
    """
    Model for a document used in preprocessing steps.
    """

    id: Optional[str] = None
    title: Optional[str] = None
    content: str
    source: Optional[str] = None
    metadata: Optional[dict] = None


def _clean_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", text)
    # Remove common invisible characters
    cleaned_text = re.sub(r"[\u200B-\u200D\uFEFF]", "", normalized_text)
    return cleaned_text


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        doc = pymupdf.open(file_path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
    return text.strip()


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load documents from a directory containing PDF files,
    extract their text, and convert them into a list of Document objects.
    """
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                logger.info(f"Processing PDF: {full_path}")
                text = extract_text_from_pdf(full_path)
                text = _clean_text(text)
                if not text:
                    logger.warning(f"No text extracted from {full_path}. Skipping document.")
                    continue
                doc_obj = Document(id=file, title=file, content=text, source=full_path, metadata={"extension": "pdf"})
                documents.append(doc_obj)
    return documents


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to the directory containing PDF files.",
)
@click.option(
    "--output",
    type=click.Path(writable=True),
    required=True,
    help="Output JSON file path. Must contain `*.json` as file-ending.",
)
def main(input_dir: str, output: str) -> None:
    """
    Process raw PDF files and save them as structured JSON.
    """
    documents = load_documents_from_directory(input_dir)
    logger.info(f"Processed {len(documents)} documents.")

    with open(output, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump() for doc in documents], f, ensure_ascii=False, indent=2)
    logger.info(f"Documents saved to {output}")


if __name__ == "__main__":
    main()
