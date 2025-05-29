import os
from collections import defaultdict, deque
from typing import Dict, List

from loguru import logger
from openai import Client

from entrag.api.ai import BaseAIEngine
from entrag.data_model.document import Chunk, ExternalChunk
from entrag.models.hybrid_rag import HybridRAG

from .graph_builder import ALLOWED_ENTITY_TYPES, ALLOWED_RELATIONSHIP_TYPES, KnowledgeGraphBuilder
from .graph_schema import ChunkGraphExtraction


class GraphRAG(HybridRAG):
    """
    Graph RAG approach that builds an in-memory knowledge graph from chunks
    and uses graph traversal for enhanced retrieval and API calling.
    """

    def __init__(
        self,
        *,
        storage_dir="./test_rag_vector_store",
        chunks: list[Chunk],
        ai_engine: BaseAIEngine,
        model_name: str,
        reranking_model_name: str,
        max_graph_distance: int = 2,
        graph_path: str = "./knowledge_graph.gpickle",
        chunk_jsonl_path: str = None,  # Path to the chunk JSONL for building the graph if needed
    ) -> None:
        logger.debug(f"Initializing GraphRAG with graph_path={graph_path}")
        super().__init__(
            storage_dir=storage_dir,
            chunks=chunks,
            ai_engine=ai_engine,
            model_name=model_name,
            reranking_model_name=reranking_model_name,
        )
        self.max_graph_distance = max_graph_distance
        self.graph_path = graph_path

        # Auto-build the graph if it does not exist
        if not os.path.exists(self.graph_path):
            if chunk_jsonl_path is None:
                raise FileNotFoundError(
                    f"Knowledge graph file not found at {self.graph_path} and no chunk_jsonl_path provided to build it."
                )
            logger.info(f"Knowledge graph file not found at {self.graph_path}. Building it from {chunk_jsonl_path}...")
            builder = KnowledgeGraphBuilder(chunk_jsonl_path, self.graph_path)
            builder.build_graph()
        self.graph = KnowledgeGraphBuilder.load_graph(self.graph_path)
        logger.debug(
            f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )

        self.chunk_to_idx = {self._get_chunk_id(chunk): i for i, chunk in enumerate(self.chunks)}
        self.client = Client(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_chunk_id(self, chunk: Chunk) -> str:
        """Create a unique identifier for a chunk."""
        return f"{chunk.document_id}:{chunk.document_page}:{chunk.chunk_location_id}"

    def extract_query_entities(self, query: str) -> dict:
        logger.debug(f"Extracting entities from query: {query}")
        prompt = (
            f"Extract all relevant entities and relationsgips from the following text.\n"
            f"Allowed entity types: {sorted(ALLOWED_ENTITY_TYPES)}\n"
            f"Allowed relationship types: {sorted(ALLOWED_RELATIONSHIP_TYPES)}\n"
            f"Text:\n{query}"
        )
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an expert information extraction system."},
                {"role": "user", "content": prompt},
            ],
            response_format=ChunkGraphExtraction,
        )
        response = completion.choices[0].message.parsed if completion.choices else None
        if response is None:
            logger.debug("No entities extracted from query.")
            return {}
        entities = {}
        for e in response.entities:
            entities.setdefault(e.type, []).append(e.name)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def retrieve(self, query: str, top_k: int):
        logger.debug(f"Retrieving for query: '{query}' with top_k={top_k}")
        query_entities = self.extract_query_entities(query)
        entity_keys = []
        for entity_type, entity_list in query_entities.items():
            for entity in entity_list:
                entity_keys.append(f"{entity_type}:{entity}")
        logger.debug(f"Entity keys for graph traversal: {entity_keys}")
        related_entities = self._traverse_graph_nx(entity_keys, self.max_graph_distance)
        logger.debug(f"Related entities found: {related_entities}")
        chunk_idx_scores = defaultdict(float)
        for entity_key, relevance in related_entities.items():
            if entity_key in self.graph.nodes:
                chunk_ids = self.graph.nodes[entity_key].get("chunk_id", [])
                logger.debug(f"Entity {entity_key} has relevance {relevance} and chunk_ids {chunk_ids}")
                if isinstance(chunk_ids, str):
                    chunk_ids = [chunk_ids]
                for chunk_id in chunk_ids:
                    logger.debug(f"Checking if chunk_id {chunk_id} is in chunk_to_idx: {self.chunk_to_idx}")
                    if chunk_id in self.chunk_to_idx:
                        chunk_idx = self.chunk_to_idx[chunk_id]
                        chunk_idx_scores[chunk_idx] += relevance
        logger.debug(f"Chunk index scores: {dict(chunk_idx_scores)}")
        sorted_indices = sorted(chunk_idx_scores.items(), key=lambda x: x[1], reverse=True)
        graph_chunks = [self.chunks[idx] for idx, _ in sorted_indices[:top_k]]
        logger.debug(f"Returning {len(graph_chunks)} chunks from graph-based retrieval.")
        return graph_chunks, []

    def _traverse_graph_nx(self, start_entities: list, max_distance: int) -> dict:
        logger.debug(f"Traversing graph from entities: {start_entities} with max_distance={max_distance}")
        visited = {}
        queue = deque()
        for entity in start_entities:
            if entity in self.graph.nodes:
                queue.append((entity, 0))
                visited[entity] = 1.0
        while queue:
            current_entity, distance = queue.popleft()
            if distance >= max_distance:
                continue
            for neighbor in self.graph.neighbors(current_entity):
                if neighbor not in visited or visited[neighbor] < (1.0 / (distance + 2)):
                    relevance = 1.0 / (distance + 2)
                    visited[neighbor] = max(visited.get(neighbor, 0), relevance)
                    queue.append((neighbor, distance + 1))
        logger.debug(f"Graph traversal visited: {visited}")
        return visited

    def get_api_results(self, entities: Dict[str, List[str]], query: str, top_k: int) -> List[ExternalChunk]:
        """Enhanced API calling using graph-expanded entities."""
        # Get base results from parent class
        results = super().get_api_results(entities, query, top_k)

        # Add graph-based scoring boost
        scored_results = []
        for result in results:
            score = 1.0

            # Check if result mentions any graph-connected entities
            result_text = result.content.lower()
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.lower() in result_text:
                        entity_key = f"{entity_type}:{entity.lower()}"
                        # Boost score based on entity centrality in graph
                        centrality = len(self.graph["edges"].get(entity_key, []))
                        score += 0.1 * min(centrality, 5)  # Cap boost

            scored_results.append((result, score))

        # Re-sort by enhanced scores
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, _ in scored_results[:top_k]]

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Enhanced generation with graph context."""
        # Add graph statistics to the system prompt
        graph_context = "\n\nKnowledge Graph Statistics:\n"
        graph_context += f"- Total entities: {self.graph.number_of_nodes()}\n"
        graph_context += f"- Total relationships: {self.graph.number_of_edges()}\n"

        enhanced_system_prompt = system_prompt + graph_context

        return super().generate(system_prompt=enhanced_system_prompt, user_prompt=user_prompt)
