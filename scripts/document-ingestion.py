"""
This script processes raw PDF files from a specified input directory and saves the structured data as markdown.
"""

import os

import click
from docling.document_converter import DocumentConverter
from loguru import logger


def load_documents_from_directory(directory_path: str, output_directory) -> int:
    """
    Load documents from a directory containing files,
    extract their text, and save them as markdown in the specified output directory.
    """

    converter = DocumentConverter()
    count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Skip non-PDF files
            if not file.endswith(".pdf"):
                continue

            full_path = os.path.join(root, file)
            filename = os.path.splitext(file)[0]
            output_filename = f"{filename}.md"

            file_path = os.path.join(output_directory, output_filename)

            if os.path.exists(file_path):
                logger.warning(f"Document already exists: {file_path}. Skipping document.")
                continue

            logger.info(f"Processing file: {full_path}")
            result = converter.convert(full_path)
            markdown = result.document.export_to_markdown(add_page_markers=True)
            if not markdown:
                logger.warning(f"No text extracted from {full_path}. Skipping document.")
                continue

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {filename}\n\n{markdown}")
                logger.info(f"Document saved as markdown: {file_path}")
                count += 1
    return count


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
    Process raw PDF files and save them in a structured format.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = load_documents_from_directory(input_dir, output_dir)
    logger.info(f"Processed {count} documents.")


if __name__ == "__main__":
    main()
