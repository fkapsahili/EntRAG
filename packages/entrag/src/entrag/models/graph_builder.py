import json
import os
import pickle

import networkx as nx
from loguru import logger
from openai import Client

from .graph_schema import ChunkGraphExtraction


ALLOWED_ENTITY_TYPES = {
    "company",
    "ticker",
    "person",
    "product",
    "document_type",
    "metric",
    "regulation",
    "location",
    "department",
}
ALLOWED_RELATIONSHIP_TYPES = {
    "mentions",
    "reports_on",
    "belongs_to",
    "regulated_by",
    "located_in",
    "represents",
    "co_occurs",
    "has_role",
    "complies_with",
    "describes",
}


class KnowledgeGraphBuilder:
    def __init__(self, chunk_jsonl_path: str, output_path: str):
        self.chunk_jsonl_path = chunk_jsonl_path
        self.output_path = output_path
        self.graph = nx.MultiDiGraph()
        self.client = Client(api_key=os.getenv("OPENAI_API_KEY"))

    def build_graph(self, max_chunks: int = None):
        logger.info(f"Building knowledge graph from {self.chunk_jsonl_path}")
        with open(self.chunk_jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if max_chunks is not None and line_num > max_chunks:
                    logger.info(f"Stopping after {max_chunks} chunks for testing.")
                    break
                chunk = json.loads(line)
                chunk_id = f"{chunk['document_id']}:{chunk['document_page']}:{chunk['chunk_location_id']}"
                logger.debug(f"Processing chunk {chunk_id} (line {line_num})")
                extraction = self.extract_entities_and_relationships(chunk["chunk_text"], chunk_id)
                self.add_to_graph(extraction)
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        self.save_graph()

    def extract_entities_and_relationships(self, text: str, chunk_id: str) -> ChunkGraphExtraction:
        prompt = (
            f"Extract entities and relationships from the text below.\n"
            f"Allowed entity types: {sorted(ALLOWED_ENTITY_TYPES)}\n"
            f"Allowed relationship types: {sorted(ALLOWED_RELATIONSHIP_TYPES)}\n"
            f"Text:\n{text}"
        )
        logger.info(f"LLM extraction prompt for chunk {chunk_id}:")
        logger.debug(prompt)
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an expert information extraction system."},
                {"role": "user", "content": prompt},
            ],
            response_format=ChunkGraphExtraction,
        )
        logger.info(f"LLM raw response for chunk {chunk_id}: {completion}")
        response = completion.choices[0].message.parsed if completion.choices else None
        if response is None:
            logger.warning(f"No entities/relationships extracted for chunk {chunk_id}")
            return ChunkGraphExtraction(entities=[], relationships=[])
        # Filter by allowed types (type safety)
        filtered_entities = [e for e in response.entities if e.type in ALLOWED_ENTITY_TYPES]
        filtered_relationships = [r for r in response.relationships if r.type in ALLOWED_RELATIONSHIP_TYPES]
        logger.info(f"Filtered entities for chunk {chunk_id}: {filtered_entities}")
        logger.info(f"Filtered relationships for chunk {chunk_id}: {filtered_relationships}")
        # Attach chunk_id to all entities/relationships if missing
        for e in filtered_entities:
            if not e.chunk_id:
                e.chunk_id = chunk_id
        for r in filtered_relationships:
            if not r.chunk_id:
                r.chunk_id = chunk_id
        return ChunkGraphExtraction(entities=filtered_entities, relationships=filtered_relationships)

    def add_to_graph(self, extraction: ChunkGraphExtraction):
        for entity in extraction.entities:
            # Use type:name as node id for consistency with retrieval
            node_id = f"{entity.type}:{entity.name}"
            self.graph.add_node(node_id, **entity.model_dump())
        for rel in extraction.relationships:
            # Use type:name for source/target
            source_id = f"{rel.source}" if ":" in rel.source else rel.source
            target_id = f"{rel.target}" if ":" in rel.target else rel.target
            # Try to resolve to type:name if possible
            # If rel.source/rel.target are entity IDs, map to type:name
            # (Assume entity IDs are unique in extraction.entities)
            entity_lookup = {e.id: f"{e.type}:{e.name}" for e in extraction.entities}
            if rel.source in entity_lookup:
                source_id = entity_lookup[rel.source]
            if rel.target in entity_lookup:
                target_id = entity_lookup[rel.target]
            # Ensure weight is always a float (use rel.weight if present, else 1.0)
            weight = rel.weight
            if weight is None:
                weight = 1.0
            else:
                try:
                    weight = float(weight)
                except Exception:
                    weight = 1.0
            rel_dict = rel.model_dump()
            rel_dict["weight"] = weight
            self.graph.add_edge(source_id, target_id, key=rel.type, **rel_dict)

    def save_graph(self):
        with open(self.output_path, "wb") as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"Knowledge graph saved to {self.output_path}")

    @staticmethod
    def load_graph(path: str) -> nx.MultiDiGraph:
        if not os.path.exists(path):
            logger.error(f"Knowledge graph file not found at {path}. Please build the graph first.")
            raise FileNotFoundError(f"Knowledge graph file not found at {path}. Please build the graph first.")

        with open(path, "rb") as f:
            return pickle.load(f)
