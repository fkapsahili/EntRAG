"""
This script processes raw PDF files from a specified input directory and saves the structured data as markdown.
"""

import os

import click
from docling.document_converter import DocumentConverter
from loguru import logger
from pypdf import PdfReader, PdfWriter


def split_pdf_to_pages(pdf_path, output_dir):
    """
    Splits a PDF into individual pages and saves them as separate PDF files.
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    page_files = []
    for i in range(num_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])

        page_filename = os.path.join(output_dir, f"{base_filename}_page_{i + 1}.pdf")
        with open(page_filename, "wb") as f:
            writer.write(f)
        page_files.append(page_filename)

    return page_files


def process_pdf_content(pdf_path: str) -> str:
    """
    Processes a PDF file with Docling and returns the extracted markdown content.
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown(add_page_markers=False)
    return markdown


def process_documents(input_dir, output_dir):
    """
    Processes each PDF in the input directory, splits it into pages,
    runs Docling on each page, and reassembles the content with page markers.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            # Skip non-PDF files for now
            if not file.endswith(".pdf"):
                continue

            full_path = os.path.join(root, file)
            output_filename = os.path.splitext(file)[0] + ".md"
            output_filepath = os.path.join(output_dir, output_filename)
            logger.info(f"Processing file: {full_path}")

            if os.path.exists(output_filepath):
                logger.warning(f"Output file already exists: {output_filepath}. Skipping document.")
                continue

            # Split PDF into individual pages
            page_files = split_pdf_to_pages(full_path, output_dir)
            if not page_files:
                logger.warning(f"No pages found in {full_path}. Skipping document.")
                continue

            # Process each page with Docling and assemble content
            assembled_content = ""
            for i, page_file in enumerate(page_files):
                logger.debug(f"Processing page {i + 1} of {len(page_files)}")
                page_markdown = process_pdf_content(page_file)
                if page_markdown.strip():
                    assembled_content += f"##PAGE {i + 1}##\n{page_markdown}\n"
                else:
                    logger.warning(f"No text extracted from {page_file}.")

                # Optionally, remove the individual page file after processing
                os.remove(page_file)

            if not assembled_content.strip():
                logger.warning(f"No content extracted from {full_path}. Skipping document.")
                continue

            # Save the assembled markdown content
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(assembled_content)
                logger.info(f"Document saved as markdown: {output_filepath}")


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
    help="Output directory where markdown files will be saved.",
)
def main(input_dir: str, output_dir: str) -> None:
    """
    Process raw PDF files, split them into pages, run Docling on each page,
    and save the structured data as markdown with page markers.
    """
    process_documents(input_dir, output_dir)
    logger.info("Processing completed.")


if __name__ == "__main__":
    main()
