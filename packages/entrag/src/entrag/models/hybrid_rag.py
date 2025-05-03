import re
from collections import defaultdict

from loguru import logger

from entrag.data_model.document import Chunk
from entrag.models.baseline_rag import BaselineRAG
from entrag.prompts.default_prompts import ENTITY_EXTRACTION_PROMPT
from entrag.utils.entity_extraction import clean_str, split_content_by_markers
from entrag_mock_api.client import MockAPIClient


class HybridRAG(BaselineRAG):
    """
    Hybrid RAG approach that combines both vector-search and entity extraction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.api_client = MockAPIClient()

    def run_entity_extraction(self, *, query: str) -> defaultdict[str, list[str]]:
        logger.info(f"Running Entity Extraction for Query: {query}")

        query_context = {
            "tuple_delimiter": "<|>",
            "record_delimiter": "##",
            "completion_delimiter": "<|COMPLETE|>",
            "entity_types": ",".join(["ticker", "metric"]),
        }

        user_prompt = ENTITY_EXTRACTION_PROMPT.format(**query_context, query_text=query)
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": user_prompt}]
        )
        response = completion.choices[0].message.content
        logger.debug(f"Entity Extraction Response: {response}")
        entities = defaultdict(list)

        if response:
            records = split_content_by_markers(
                response, [query_context["record_delimiter"], query_context["completion_delimiter"]]
            )
            logger.debug(f"Extracted Records: {records}")

            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue

                record = record.group(1)
                attributes = split_content_by_markers(record, [query_context["tuple_delimiter"]])
                entity = self.extract_entity_from_attrs(attributes)
                if entity is not None:
                    entities[clean_str(entity["entity_type"])].append(clean_str(entity["entity_name"]))

        logger.debug(f"Extracted Entities: {entities}")
        return entities

    def get_api_results(self, entities: dict[str, list[str]]) -> list[str]:
        ticker = entities["ticker"][0] if entities["ticker"] else None
        metric = entities["metric"][0] if entities["metric"] else None

        if ticker is not None:
            try:
                company_metrics = self.api_client.get_finance_company_metrics(ticker)
            except Exception as e:
                logger.error(f"Failed to retrieve company metrics for {ticker}: {e}")
                return []

            if metric is not None:
                metric_value = getattr(company_metrics, metric, None)
                if metric_value is not None:
                    return [f"{metric}: {metric_value}"]
            return [f"Company Metrics for {ticker}: {company_metrics}"]

        return []

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[Chunk], list[str]]:
        entities = self.run_entity_extraction(query=query)
        results = self.get_api_results(entities)

        chunks, _ = super().retrieve(query, top_k)
        return chunks, results

    def extract_entity_from_attrs(self, entity_attributes: list[str]) -> dict:
        if len(entity_attributes) < 3 or entity_attributes[0] != '"entity"':
            return None

        try:
            return {
                "entity_name": entity_attributes[1],
                "entity_type": entity_attributes[2],
                "description": entity_attributes[3],
            }
        except ValueError as e:
            logger.warning(f"Invalid entity data: {e}")
            return None
