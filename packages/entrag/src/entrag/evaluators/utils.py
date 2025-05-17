import os


def normalize_filename(filename: str) -> str:
    return os.path.splitext(filename)[0].lower()


def is_api_source(source: str) -> bool:
    """
    Check if the source is an API source.
    """
    return source.startswith("api-")
