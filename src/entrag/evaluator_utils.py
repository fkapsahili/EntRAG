from pathlib import Path
import json

from entrag.api.model import Document


def load_documents(file_path: str) -> list[Document]:
    """
    Load documents from a JSON file and convert them into a list of Document objects.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("documents", [])
    elif not isinstance(data, list):
        raise ValueError(
            "Invalid data format. Expected a list or a dictionary with a 'documents' key."
        )

    documents = [Document(**doc) for doc in data]
    return documents
