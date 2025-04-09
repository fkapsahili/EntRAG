import datetime

import pytz

from entrag.data_model.document import Chunk


def get_query_time() -> str:
    """
    Get the current time in a human-readable format.
    """
    query_time = datetime.datetime.now(pytz.timezone("Europe/Zurich"))
    return query_time.strftime("%Y-%m-%d %H:%M:%S")


def get_formatted_chunks(chunks: list[Chunk]) -> str:
    """
    Format the chunks in an LLM-readable format.
    """
    return "\n\n".join([f"{chunk.document_name}\n{chunk.chunk_text}" for chunk in chunks])
