import glob
import json
import os

import click
from bs4 import BeautifulSoup
from bs4.element import Comment


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--output-dir", "-o", type=click.Path(), default="cleaned_html", help="Directory to save cleaned HTML files"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "html"]),
    default="json",
    help="Output format: json with page_result field or plain html",
)
@click.option("--pretty", "-p", is_flag=True, help="Pretty print the output HTML")
@click.option("--recursive", "-r", is_flag=True, help="Recursively search for HTML files in subdirectories")
@click.option("--pattern", default="*.html", help="File pattern to match (e.g., *.html, *.htm)")
def clean_html(input_dir, output_dir, format, pretty, recursive, pattern):
    """Clean HTML files in a directory by removing unnecessary elements."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")

    # Find all matching files in the input directory
    html_files = []
    if recursive:
        for root, _, _ in os.walk(input_dir):
            files = glob.glob(os.path.join(root, pattern))
            html_files.extend(files)
    else:
        html_files = glob.glob(os.path.join(input_dir, pattern))

    if not html_files:
        click.echo(f"No files matching pattern '{pattern}' found in {input_dir}")
        return

    click.echo(f"Found {len(html_files)} HTML files to process")

    for input_file in html_files:
        click.echo(f"Processing {input_file}...")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse HTML using BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")

            # Remove scripts, styles, comments and other unnecessary elements
            for element in soup(["script", "style", "noscript", "iframe", "link"]):
                element.decompose()

            # Remove comments
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Simplify meta tags (keep only essential ones)
            meta_tags = soup.find_all("meta")
            for meta in meta_tags:
                # Keep only essential meta tags like charset, viewport, description, title
                if meta.get("charset") or meta.get("name") in ["viewport", "description", "title"]:
                    continue
                meta.decompose()

            # Get the cleaned HTML
            if pretty:
                cleaned_html = soup.prettify()
            else:
                cleaned_html = str(soup)

            # Create relative path for output to maintain directory structure
            rel_path = os.path.relpath(input_file, input_dir)
            output_subdir = os.path.dirname(rel_path)
            output_full_dir = os.path.join(output_dir, output_subdir)

            if output_subdir and not os.path.exists(output_full_dir):
                os.makedirs(output_full_dir)

            # Save the cleaned HTML in the specified format
            filename = os.path.basename(input_file)
            base_name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_full_dir, f"{base_name}_cleaned")

            if format == "json":
                with open(f"{output_path}.json", "w", encoding="utf-8") as f:
                    json_data = {"page_result": cleaned_html}
                    json.dump(json_data, f, ensure_ascii=False, indent=2 if pretty else None)
                click.echo(f"Saved cleaned JSON to {output_path}.json")
            else:
                with open(f"{output_path}.html", "w", encoding="utf-8") as f:
                    f.write(cleaned_html)
                click.echo(f"Saved cleaned HTML to {output_path}.html")

        except Exception as e:
            click.echo(f"Error processing {input_file}: {str(e)}")


if __name__ == "__main__":
    clean_html()
