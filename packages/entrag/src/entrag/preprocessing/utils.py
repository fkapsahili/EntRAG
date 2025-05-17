import hashlib
import uuid

from entrag.config.dataset import DOCUMENT_NAMESPACE


def generate_document_id(filename: str, content: str) -> str:
    """
    Generate a deterministic UUID based on a filename and content.
    """
    content_bytes = content.encode("utf-8")
    content_hash = hashlib.md5(content_bytes).hexdigest()[:8]
    unique_key = f"{filename}_{content_hash}"

    return str(uuid.uuid5(DOCUMENT_NAMESPACE, unique_key))
