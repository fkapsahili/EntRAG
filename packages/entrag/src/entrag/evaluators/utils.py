import os


def normalize_filename(filename: str) -> str:
    return os.path.splitext(filename)[0].lower()
