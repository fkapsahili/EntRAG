import re


def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)
