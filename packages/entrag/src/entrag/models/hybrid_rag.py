import math
import re
from collections import defaultdict

from loguru import logger

from entrag.data_model.document import Chunk, ExternalChunk
from entrag.models.baseline_rag import BaselineRAG
from entrag.prompts.default_prompts import API_RERANKING_PROMPT, ENTITY_EXTRACTION_PROMPT
from entrag.utils.entity_extraction import clean_str, split_content_by_markers
from entrag_mock_api.client import MockAPIClient, MockAPIError


class HybridRAG(BaselineRAG):
    """
    Hybrid RAG approach that combines both vector-search and entity extraction with API selection.
    """

    def __init__(self, *, storage_dir="./test_rag_vector_store", chunks: list[Chunk]) -> None:
        super().__init__(storage_dir=storage_dir, chunks=chunks)
        self.api_client = MockAPIClient()

    def run_entity_extraction(self, *, query: str) -> defaultdict[str, list[str]]:
        logger.info(f"Running Entity Extraction for Query: [{query}]")

        query_context = {
            "tuple_delimiter": "<|>",
            "record_delimiter": "##",
            "completion_delimiter": "<|COMPLETE|>",
            "entity_types": ",".join([
                "ticker",
                "metric",
                "company",
                "filing_type",
                "employer",
                "date",
                "search_term",
            ]),
        }

        user_prompt = ENTITY_EXTRACTION_PROMPT.format(**query_context, query_text=query)
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": user_prompt}]
        )
        response = completion.choices[0].message.content
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

    def get_api_results(self, entities: dict[str, list[str]], query: str, top_k: int = 5) -> list[ExternalChunk]:
        """
        Fetch APIs based on selection.
        """
        results = []
        results.extend(self._get_finance_results(entities))
        results.extend(self._get_filing_results(entities))
        results.extend(self._get_gpg_results(entities))
        results.extend(self._get_search_results(entities))

        # Rank and return top k results
        ranked_results = self._rerank_api_results(results, query)
        return ranked_results[:top_k]

    def _get_finance_results(self, entities: dict[str, list[str]]) -> list[ExternalChunk]:
        results = []
        tickers = entities.get("ticker", [])
        metrics = entities.get("metric", [])
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
                                    metric_response = self.api_client.get_finance_metric_by_date(ticker, metric, date)
                                    results.append(
                                        ExternalChunk(
                                            content=f"{ticker} {metric} on {date}: {metric_response.value}",
                                            source="finance_api",
                                        )
                                    )
                                except MockAPIError:
                                    continue

                        try:
                            timeseries = self.api_client.get_finance_timeseries(ticker, metric)
                            if timeseries.results:
                                latest_date = max(timeseries.results.keys())
                                latest_value = timeseries.results[latest_date]
                                results.append(
                                    ExternalChunk(
                                        content=f"{ticker} {metric} timeseries (latest {latest_date}): {latest_value}",
                                        source="finance_api",
                                    )
                                )
                        except MockAPIError:
                            continue

                        metric_value = getattr(company_metrics, metric, None)
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
                            content=f"Company Metrics for {ticker}: {company_metrics}",
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
                            content=f"Company Metrics for {company}: {company_metrics}",
                            source="finance_api",
                        )
                    )
                except MockAPIError:
                    continue

        return results

    def _get_filing_results(self, entities: dict[str, list[str]]) -> list[ExternalChunk]:
        results = []
        filing_types = entities.get("filing_type", [])
        companies = entities.get("company", [])

        filing_type_map = {
            "quarterly": "10-Q",
            "10q": "10-Q",
            "annual": "10-K",
            "10k": "10-K",
            "Form 10-K": "10-K",
            "Form 10-Q": "10-Q",
            "Form 8-K": "8-K",
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
                                    content=f"Found {len(filings_data['filings'])} {normalized_type} filings for {company}",
                                    source="filings_typi_api",
                                )
                            )
                else:
                    filings_data = self.api_client.get_filings_by_type(normalized_type)
                    if filings_data.get("filings"):
                        results.append(
                            ExternalChunk(
                                content=f"Found {len(filings_data['filings'])} {normalized_type} filings",
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
                                content=f"Found {len(search_results['results'])} filings for {company}",
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
                            content=f"GPG statistics for {employer}: {len(gpg_data.statistics)} records found",
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
                            content=f"Website search for '{term}': {len(search_data.results)} results found",
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
            prompt = API_RERANKING_PROMPT.format(query=query, source=res.source, api_result=res.content)
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], logprobs=True, max_tokens=1
            )
            response = completion.choices[0].message.content.strip().lower()

            if response == "yes" and completion.choices[0].logprobs.content:
                yes_logprob = completion.choices[0].logprobs.content[0].logprob
                logger.debug(f"API Result from: {res.source}, Response: {response}, Logprob: {yes_logprob}")
                prob = math.exp(yes_logprob)
                logger.debug(f"Reranked API Result Confidence: {prob}")
                if prob >= threshold:
                    reranked.append((res, prob))
            else:
                logger.debug(f"API Result from: {res.source}, Response: {response} - Skipped")

        return [chunk for chunk, _ in sorted(reranked, key=lambda x: x[1], reverse=True)]

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[Chunk], list[ExternalChunk]]:
        entities = self.run_entity_extraction(query=query)

        ext_chunks = self.get_api_results(entities, query)
        chunks, _ = super().retrieve(query, top_k)
        return chunks, ext_chunks

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
