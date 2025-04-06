import html
import re


def split_content_by_markers(content: str, markers: list[str]) -> list[str]:
    """
    Split a given content by many markers.
    """
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# https://github.com/microsoft/graphrag
def clean_str(input_str: str) -> str:
    """
    Clean an input string by removing HTML escapes, control characters, and other unwanted characters.
    """
    # If we get non-string input, just give it back
    if not isinstance(input_str, str):
        return input_str

    result = html.unescape(input_str.strip()).strip("\\").strip('"').strip("'")
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
