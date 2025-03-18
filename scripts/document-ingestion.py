import os

import click
import PyPDF2
from docling.document_converter import DocumentConverter
from loguru import logger


def load_documents_from_directory(directory_path: str, output_directory) -> int:
    """
    Load documents from a directory containing files,
    extract their text page-wise, and save them as markdown in the specified output directory.
    """

    converter = DocumentConverter()
    count = 0

    for root, _, files in os.walk(directory_path):
        for file in files:
            if not file.endswith(".pdf"):
                continue

            full_path = os.path.join(root, file)
            rel_path = root[len(directory_path) :].strip(os.path.sep)
            filename = os.path.splitext(file)[0]

            logger.info(f"Processing file: {full_path}")

            with open(full_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)

                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if not page_text.strip():
                        logger.warning(f"No text extracted from page {page_number} of {full_path}. Skipping page.")
                        continue

                    result = converter.convert(page_text)
                    markdown = result.document.export_to_markdown()

                    if not markdown:
                        logger.warning(f"No markdown generated from page {page_number} of {full_path}. Skipping page.")
                        continue

                    if rel_path:
                        output_filename = f"{rel_path.replace(os.path.sep, '_')}_{filename}_page_{page_number}.md"
                    else:
                        output_filename = f"{filename}_page_{page_number}.md"

                    file_path = os.path.join(output_directory, output_filename)

                    if os.path.exists(file_path):
                        logger.warning(f"Document already exists: {file_path}. Skipping page.")
                        continue

                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"<!-- page_number: {page_number} -->\n\n# {filename} - Page {page_number}\n\n{markdown}"
                        )

                    logger.info(f"Page saved as markdown: {file_path}")
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
    help="Output directory path. The documents will be saved as markdown.",
)
def main(input_dir: str, output_dir: str) -> None:
    """
    Process raw PDF files and save them page-wise as markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = load_documents_from_directory(input_dir, output_dir)
    logger.info(f"Processed {count} pages into markdown.")


if __name__ == "__main__":
    main()
