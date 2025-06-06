"""
This script processes raw input files from a specified input directory and saves the structured data as markdown.
"""

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import click
from docling.document_converter import DocumentConverter
from loguru import logger
from pypdf import PdfReader, PdfWriter


def split_pdf_to_pages(pdf_path, temp_dir):
    """
    Splits a PDF into individual pages and saves them as separate PDF files.
    Returns a list of page file paths.
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    page_files = []
    for i in range(num_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])

        page_filename = os.path.join(temp_dir, f"{base_filename}_page_{i + 1}.pdf")
        with open(page_filename, "wb") as f:
            writer.write(f)
        page_files.append((i + 1, page_filename))

    return page_files


def process_page(page_tuple):
    """
    Process a single page with Docling.
    Returns the page number and the markdown content.
    """
    page_num, page_file = page_tuple
    try:
        converter = DocumentConverter()
        result = converter.convert(page_file)
        markdown = result.document.export_to_markdown()
        return page_num, markdown, page_file
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return page_num, "", page_file


def process_pdf(pdf_path, temp_dir, max_workers=None):
    """
    Process a single PDF file, splitting it into pages and processing each page in parallel.
    """
    logger.info(f"Processing file: {pdf_path}")

    # Split PDF into individual pages
    page_files = split_pdf_to_pages(pdf_path, temp_dir)
    if not page_files:
        logger.warning(f"No pages found in {pdf_path}.")
        return None

    # Process each page with Docling in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(process_page, page_tuple): page_tuple for page_tuple in page_files}

        for future in as_completed(future_to_page):
            page_num, markdown, page_file = future.result()
            results[page_num] = markdown
            # Clean up the page file after processing
            try:
                os.remove(page_file)
            except Exception as e:
                logger.warning(f"Error removing temporary page file {page_file}: {e}")

    # Assemble content in correct page order
    assembled_content = ""
    for page_num in sorted(results.keys()):
        markdown = results[page_num]
        if markdown.strip():
            assembled_content += f"##PAGE {page_num}##\n{markdown}\n"
        else:
            logger.warning(f"No text extracted from page {page_num}.")

    return assembled_content


def process_documents(input_dir, output_dir, max_workers=None, batch_size=1):
    """
    Processes each PDF in the input directory, processing multiple PDFs concurrently.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all PDF files to process
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                output_filename = os.path.splitext(file)[0] + ".md"
                output_filepath = os.path.join(output_dir, output_filename)

                if os.path.exists(output_filepath):
                    logger.warning(f"Output file already exists: {output_filepath}. Skipping document.")
                    continue

                pdf_files.append((full_path, output_filepath))

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process PDFs in batches to avoid memory issues
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(pdf_files) - 1) // batch_size + 1}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each PDF in the batch
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                process_pdf_partial = partial(process_single_pdf, temp_dir=temp_dir, max_workers=1)
                futures = {
                    executor.submit(process_pdf_partial, pdf_path, output_path): pdf_path
                    for pdf_path, output_path in batch
                }

                for future in as_completed(futures):
                    pdf_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path}: {e}")


def process_single_pdf(pdf_path, output_path, temp_dir, max_workers):
    """
    Helper function to process a single PDF and save it to the output path.
    """
    try:
        assembled_content = process_pdf(pdf_path, temp_dir, max_workers)
        if assembled_content and assembled_content.strip():
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(assembled_content)
            logger.info(f"Document saved as markdown: {output_path}")
            return True
        else:
            logger.warning(f"No content extracted from {pdf_path}. Skipping document.")
            return False
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return False


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
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Maximum number of worker processes. Defaults to number of CPU cores.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Number of PDFs to process in parallel. Default is 1.",
)
def main(input_dir: str, output_dir: str, workers: int, batch_size: int) -> None:
    """
    Process raw PDF files, split them into pages, run Docling on each page,
    and save the structured data as markdown with page markers.
    """
    logger.info(f"Starting PDF processing with batch size: {batch_size}, workers: {workers or 'auto'}")
    process_documents(input_dir, output_dir, max_workers=workers, batch_size=batch_size)
    logger.info("Processing completed.")


if __name__ == "__main__":
    main()
