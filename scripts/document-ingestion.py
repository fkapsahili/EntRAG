"""
This script processes raw PDF files from a specified input directory and saves the structured data as markdown.
"""

import os
import re
import unicodedata
import uuid

import click
from loguru import logger
from markitdown import MarkItDown
from pydantic import BaseModel


class Document(BaseModel):
    """
    Model for a document used in preprocessing steps.
    """

    id: str | None = None
    title: str
    content: str
    source: str | None = None


def _clean_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", text)
    # Remove common invisible characters
    cleaned_text = re.sub(r"[\u200B-\u200D\uFEFF]", "", normalized_text)
    return cleaned_text


def load_documents_from_directory(directory_path: str) -> list[Document]:
    """
    Load documents from a directory containing files,
    extract their text, and convert them into a list of Documents.
    """
    md = MarkItDown()
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            logger.info(f"Processing file: {full_path}")
            result = md.convert(full_path)
            text = _clean_text(result.text_content)
            if not text:
                logger.warning(f"No text extracted from {full_path}. Skipping document.")
                continue
            filename = os.path.splitext(file)[0]
            document = Document(id=str(uuid.uuid4()), title=filename, content=result.text_content, source=file)
            documents.append(document)
    return documents


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to the directory containing PDF files.",
)
@click.option(
    "--output-dir",
    type=click.Path(writable=True),
    required=True,
    help="Output file path. The documents will be saved as markdown.",
)
def main(input_dir: str, output_dir: str) -> None:
    """
    Process raw PDF files and save them in the structured format.
    """
    documents = load_documents_from_directory(input_dir)
    logger.info(f"Processed {len(documents)} documents.")

    os.makedirs(output_dir, exist_ok=True)
    for doc in documents:
        file_path = os.path.join(output_dir, f"{doc.title}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc.title}\n\n{doc.content}")
        logger.info(f"Document saved as markdown: {file_path}")


if __name__ == "__main__":
    main()
