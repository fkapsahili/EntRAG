import os
import shutil

import click


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
    help="Absolute path to the directory containing company folders",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    required=True,
    help="Path to output directory where flattened files will be stored",
)
def main(input_dir: str, output_dir: str) -> None:
    """
    Collects all files from company directories and copies them to the output directory
    with flattened names in the format: Company_Filename.ext
    """
    os.makedirs(output_dir, exist_ok=True)

    # Count for statistics
    total_files = 0

    for company_dir in os.listdir(input_dir):
        company_path = os.path.join(input_dir, company_dir)

        # Skip if not a directory
        if not os.path.isdir(company_path):
            continue

        parent_name = os.path.basename(company_path)

        for root, _, files in os.walk(company_path):
            for file in files:
                # Source file path
                source_file = os.path.join(root, file)

                target_filename = f"{parent_name}_{file}"
                target_file = os.path.join(output_dir, target_filename)

                shutil.copy2(source_file, target_file)
                total_files += 1
                click.echo(f"Copied: {source_file} -> {target_file}")

    click.echo(f"\nComplete! {total_files} files were processed and copied to {output_dir}")


if __name__ == "__main__":
    main()
