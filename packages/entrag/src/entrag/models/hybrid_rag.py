import math
import re
from collections import defaultdict

from loguru import logger

from entrag.api.ai import BaseAIEngine
from entrag.data_model.document import Chunk, ExternalChunk
from entrag.models.baseline_rag import BaselineRAG
from entrag.prompts.default_prompts import (
    API_RERANKING_SYSTEM_PROMPT,
    API_RERANKING_USER_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
)
from entrag.utils.entity_extraction import clean_str, split_content_by_markers
from entrag.utils.prompt import truncate_to_token_limit
from entrag_mock_api.client import MockAPIClient, MockAPIError


class HybridRAG(BaselineRAG):
    """
    Hybrid RAG approach that combines both similarity search and entity extraction with dynamic API selection.
    """

    def __init__(
        self,
        *,
        storage_dir="./test_rag_vector_store",
        chunks: list[Chunk],
        ai_engine: BaseAIEngine,
        model_name: str,
        reranking_model_name: str,
    ) -> None:
        super().__init__(storage_dir=storage_dir, chunks=chunks, ai_engine=ai_engine, model_name=model_name)
        self.api_client = MockAPIClient()
        self.reranking_model_name = reranking_model_name

    def run_entity_extraction(self, *, query: str) -> defaultdict[str, list[str]]:
        logger.info(f"Running Entity Extraction for Query: [{query}]")

        query_context = {
            "tuple_delimiter": "<|>",
            "record_delimiter": "##",
            "completion_delimiter": "<|COMPLETE|>",
            "entity_types": ",".join([
                "ticker",
                "metric_symbol",
                "company",
                "sec_form_type",
                "employer",
                "date",
                "search_term",
            ]),
        }

        user_prompt = ENTITY_EXTRACTION_PROMPT.format(**query_context, query_text=query)
        response = self.ai_engine.chat_completion(model=self.model_name, user=user_prompt)
        logger.debug(f"Entity Extraction Response: {response}")

        entities = defaultdict(list)
        if not response:
            return entities

        if not re.search(r'\("entity".*?\)', response):
            logger.warning("Entity extraction failed. No valid entities found in response")
            return entities

        records = split_content_by_markers(
            response, [query_context["record_delimiter"], query_context["completion_delimiter"]]
        )
        logger.debug(f"Extracted Records: {records}")

        for record in records:
            record_match = re.search(r"\((.*)\)", record)
            if not record_match:
                continue

            record_content = record_match.group(1)
            attributes = split_content_by_markers(record_content, [query_context["tuple_delimiter"]])
            entity = self.extract_entity_from_attrs(attributes)
            if entity:
                entities[clean_str(entity["entity_type"])].append(clean_str(entity["entity_name"]))

        logger.debug(f"Extracted Entities: [{entities}]")
        return entities

    def get_api_results(self, entities: dict[str, list[str]], query: str, top_k: int) -> list[ExternalChunk]:
        """
        Fetch APIs based on selection.
        """
        results: list[ExternalChunk] = []
        results.extend(self._get_finance_results(entities))
        results.extend(self._get_filing_results(entities))
        results.extend(self._get_gpg_results(entities))
        results.extend(self._get_search_results(entities))

        # Deduplicate the external results
        seen = set()
        unique_results: list[ExternalChunk] = []
        for res in results:
            identifier = (res.content, res.source)
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(res)

        # Rank and return top k results
        ranked_results = self._rerank_api_results(unique_results, query)
        return ranked_results[:top_k]

    def _get_finance_results(self, entities: dict[str, list[str]]) -> list[ExternalChunk]:
        results = []
        tickers = entities.get("ticker", [])
        metrics = entities.get("metric_symbol", [])
        companies = entities.get("company", [])
        dates = entities.get("date", [])

        for ticker in tickers:
            try:
                company_metrics = self.api_client.get_finance_company_metrics(ticker)

                if metrics:
                    for metric in metrics:
                        if dates:
                            for date in dates:
                                try:
                                    metric_response = self.api_client.get_finance_metric_by_date(
                                        ticker, metric.lower(), date
                                    )
                                    results.append(
                                        ExternalChunk(
                                            content=f"{ticker} {metric} on {date}: {metric_response.model_dump()}",
                                            source="finance_api",
                                        )
                                    )
                                except MockAPIError:
                                    continue

                        try:
                            timeseries = self.api_client.get_finance_timeseries(ticker, metric.lower())
                            if timeseries.results:
                                results.append(
                                    ExternalChunk(
                                        content=f"{ticker} {metric} timeseries: {timeseries.model_dump()}",
                                        source="finance_api",
                                    )
                                )
                        except MockAPIError:
                            continue

                        metric_value = getattr(company_metrics, metric.lower(), None)
                        if metric_value is not None:
                            results.append(
                                ExternalChunk(
                                    content=f"{ticker} {metric}: {metric_value}",
                                    source="finance_ap",
                                )
                            )
                else:
                    results.append(
                        ExternalChunk(
                            content=f"Company Metrics for {ticker}: {company_metrics.model_dump()}",
                            source="finance_api",
                        )
                    )

            except MockAPIError as e:
                logger.warning(f"Failed to get finance data for {ticker}: {e}")

        for company in companies:
            if company.lower() not in [t.lower() for t in tickers]:
                try:
                    company_metrics = self.api_client.get_finance_company_metrics(company)
                    results.append(
                        ExternalChunk(
                            content=f"Company Metrics for {company}: {company_metrics.model_dump()}",
                            source="finance_api",
                        )
                    )
                except MockAPIError:
                    continue

        return results

    def _get_filing_results(self, entities: dict[str, list[str]]) -> list[ExternalChunk]:
        results = []
        filing_types = entities.get("sec_form_type", [])
        companies = entities.get("company", [])

        filing_type_map = {
            "quarterly": "10-Q",
            "10q": "10-Q",
            "annual": "10-K",
            "10k": "10-K",
            "Form 10-K": "10-K",
            "Form 10-Q": "10-Q",
            "Form 8-K": "8-K",
            "10-K Form": "10-K",
            "10-Q Form": "10-Q",
            "8-K Form": "8-K",
            "proxy": "DEF 14A",
            "def14a": "DEF 14A",
            "8k": "8-K",
        }

        for filing_type in filing_types:
            normalized_type = filing_type_map.get(filing_type.lower(), filing_type)
            try:
                if companies:
                    for company in companies:
                        filings_data = self.api_client.get_filings_by_type(normalized_type, company)
                        if filings_data.get("filings"):
                            results.append(
                                ExternalChunk(
                                    content=f"Found {normalized_type} filings for {company}: {filings_data.get('filings')}",
                                    source="filings_type_api",
                                )
                            )
                else:
                    filings_data = self.api_client.get_filings_by_type(normalized_type)
                    if filings_data.get("filings"):
                        results.append(
                            ExternalChunk(
                                content=f"Found {normalized_type} filings: {filings_data.get('filings')}",
                                source="filings_type_api",
                            )
                        )
            except MockAPIError as e:
                logger.warning(f"Failed to get filings for {normalized_type}: {e}")

        if companies and not filing_types:
            for company in companies:
                try:
                    search_results = self.api_client.search_filings(company=company)
                    if search_results.get("results"):
                        results.append(
                            ExternalChunk(
                                content=f"Filings for {company}: {search_results}",
                                source="filings_search_api",
                            )
                        )
                except MockAPIError:
                    continue

        return results

    def _get_gpg_results(self, entities: dict[str, list[str]]) -> list[ExternalChunk]:
        results = []
        employers = entities.get("employer", []) + entities.get("company", [])

        for employer in employers:
            try:
                gpg_data = self.api_client.get_gpg_statistics(employer)
                if gpg_data.statistics:
                    results.append(
                        ExternalChunk(
                            content=f"GPG statistics for {employer}: {gpg_data.model_dump()}",
                            source="gpg_statistics_api",
                        )
                    )
            except MockAPIError:
                continue

        return results

    def _get_search_results(self, entities: dict[str, list[str]]) -> list[ExternalChunk]:
        results = []
        search_terms = entities.get("search_term", [])

        for term in search_terms:
            try:
                search_data = self.api_client.search_websites(term)
                if search_data.results:
                    results.append(
                        ExternalChunk(
                            content=f"Website search: {search_data.model_dump()}",
                            source="search_api",
                        )
                    )
            except MockAPIError:
                continue

        return results

    def _rerank_api_results(
        self, results: list[ExternalChunk], query: str, threshold: float = 0.8
    ) -> list[ExternalChunk]:
        if not results:
            return results

        reranked = []
        for res in results:
            if self.model_name in ["gpt-4o", "gpt-4o-mini"]:
                content = truncate_to_token_limit(res.content, model=self.model_name, max_tokens=124_000)
            else:
                content = res.content
            user_prompt = API_RERANKING_USER_PROMPT.format(query=query, source=res.source, api_result=content)
            completion = self.openai_client.chat.completions.create(
                model=self.reranking_model_name,
                messages=[
                    {"role": "system", "content": API_RERANKING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                logprobs=True,
                max_tokens=1,
            )
            response = completion.choices[0].message.content.strip().lower()
            logger.debug(f"Reranking Result from: {res.source}, Response: {response}")

            if response == "yes" and completion.choices[0].logprobs.content:
                yes_logprob = completion.choices[0].logprobs.content[0].logprob
                logger.debug(f"API Result from: {res.source}, Response: {response}, Logprob: {yes_logprob}")
                prob = math.exp(yes_logprob)
                logger.debug(f"Reranked API Result Confidence: {prob}")
                if prob >= threshold:
                    logger.debug(f"API Result from: {res.source} - Accepted with probability {prob}")
                    reranked.append((res, prob))
            else:
                logger.debug(f"API Result from: {res.source}, Response: {response} - Skipped")

        return [chunk for chunk, _ in sorted(reranked, key=lambda x: x[1], reverse=True)]

    def retrieve(self, query: str, top_k: int) -> tuple[list[Chunk], list[ExternalChunk]]:
        entities = self.run_entity_extraction(query=query)

        ext_chunks = self.get_api_results(entities, query, top_k)
        chunks, _ = super().retrieve(query, top_k)
        return chunks, ext_chunks

    def extract_entity_from_attrs(self, entity_attributes: list[str]) -> dict:
        if len(entity_attributes) < 4 or entity_attributes[0] != '"entity"':
            return None

        try:
            entity = {
                "entity_name": entity_attributes[1],
                "entity_type": entity_attributes[2],
                "description": entity_attributes[3],
            }
            logger.debug(f"Extracted Entity: {entity}")
            return entity
        except ValueError as e:
            logger.warning(f"Invalid entity data: {e}")
            return None
