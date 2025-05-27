import json
import os

from loguru import logger
from openai import Client

from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.document import Chunk, ChunkEmbedding
from entrag.preprocessing.create_chunks import load_document_chunks


def _embed_chunks(openai_client: Client, chunks: list[Chunk], model: str) -> list[list[float]]:
    response = openai_client.embeddings.create(input=[chunk.chunk_text for chunk in chunks], model=model)
    return [x.embedding for x in response.data]


def create_embeddings_for_chunks(config: EvaluationConfig) -> list[ChunkEmbedding]:
    """
    Compute the embeddings for the configured chunks, skipping chunks that already have embeddings.
    """
    openai_client = Client(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("Starting embedding creation.")
    chunks = load_document_chunks(config.chunking)
    logger.info(f"Loaded {len(chunks)} chunks.")

    os.makedirs(config.embedding.output_directory, exist_ok=True)
    output_file = os.path.join(config.embedding.output_directory, "embeddings.jsonl")

    existing_embeddings = {}
    if os.path.exists(output_file):
        logger.info("Found existing embeddings file, loading to avoid recomputation.")
        with open(output_file, "r", encoding="utf-8") as file:
            for line in file:
                embedding_data = json.loads(line)
                key = f"{embedding_data['document_id']}:{embedding_data['document_page']}:{embedding_data['chunk_location_id']}"
                existing_embeddings[key] = embedding_data

    # Identify chunks that we need to embed
    chunks_to_embed = []
    chunk_indices = []

    for i, chunk in enumerate(chunks):
        key = f"{chunk.document_id}:{chunk.document_page}:{chunk.chunk_location_id}"
        if key not in existing_embeddings:
            chunks_to_embed.append(chunk)
            chunk_indices.append(i)

    logger.info(f"Found {len(existing_embeddings)} existing embeddings.")
    logger.info(f"Need to compute embeddings for {len(chunks_to_embed)} new chunks.")

    new_embeddings = []
    for i in range(0, len(chunks_to_embed), config.embedding.batch_size):
        batch = chunks_to_embed[i : i + config.embedding.batch_size]
        logger.info(f"Creating embeddings for batch {i // config.embedding.batch_size + 1}, size: {len(batch)}")
        new_embeddings.extend(_embed_chunks(openai_client, batch, config.embedding.model))

    embedding_chunks = []

    with open(output_file, "w", encoding="utf-8") as file:
        for i, chunk in enumerate(chunks):
            key = f"{chunk.document_id}:{chunk.document_page}:{chunk.chunk_location_id}"

            if key in existing_embeddings:
                # Reuse the embedding
                embedding_data_dict = existing_embeddings[key]
                embedding_data = ChunkEmbedding(
                    document_id=embedding_data_dict["document_id"],
                    chunk_location_id=embedding_data_dict["chunk_location_id"],
                    document_name=embedding_data_dict["document_name"],
                    document_page=embedding_data_dict["document_page"],
                    embedding=embedding_data_dict["embedding"],
                )
            else:
                new_idx = chunk_indices.index(i)
                embedding_data = ChunkEmbedding(
                    document_id=chunk.document_id,
                    chunk_location_id=chunk.chunk_location_id,
                    embedding=new_embeddings[new_idx],
                    document_name=chunk.document_name,
                    document_page=chunk.document_page,
                )

            embedding_chunks.append(embedding_data)
            file.write(json.dumps(embedding_data.model_dump(), ensure_ascii=False) + "\n")

    logger.success("Embeddings created and saved successfully.")
    return embedding_chunks
