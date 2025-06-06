import json
import textwrap
from typing import Any, override

import faiss
import numpy as np
from loguru import logger
from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam
from pydantic import BaseModel, Field

from entrag.api.ai import BaseAIEngine
from entrag.data_model.document import Chunk, ExternalChunk
from entrag.models.baseline_rag import BaselineRAG
from entrag_mock_api.client import MockAPIClient, MockAPIError


class SearchKnowledgeBase(BaseModel):
    query: str = Field(..., description="Search query to find relevant documents")
    top_k: int = Field(5, description="Maximum number of chunks to retrieve")


class GetFinanceCompanyMetrics(BaseModel):
    ticker: str = Field(..., description="Company ticker symbol (e.g. 'AAPL', 'GOOGL')")


class GetFinanceMetricByDate(BaseModel):
    ticker: str = Field(..., description="Company ticker symbol (e.g. 'AAPL', 'GOOGL')")
    metric: str = Field(..., description="Financial metric to retrieve (currently only 'eps' is supported)")
    date: str = Field(..., description="Date in YYYY-MM-DD format. Use 'yyyy-12-31' for EPS data of a specific year.")


class GetFinanceTimeseries(BaseModel):
    ticker: str = Field(..., description="Company ticker symbol (e.g. 'AAPL', 'GOOGL')")
    metric: str = Field(
        "eps", description="Financial metric to get timeseries for (currently only 'eps' is supported)"
    )


class GetFilingsByType(BaseModel):
    filing_type: str = Field(..., description="SEC filing type (e.g. '10-K', '10-Q', '8-K', 'DEF 14A')")
    company: str | None = Field(None, description="Optional company name to filter filings")


class SearchFilings(BaseModel):
    filing_type: str | None = Field(
        None, description="Optional filing type to filter results (e.g. '10-K', '10-Q', '8-K', 'DEF 14A')"
    )
    company: str = Field(..., description="Company name to search filings for")


class GetGPGStatistics(BaseModel):
    employer: str = Field(..., description="Employer/company name to get GPG statistics for, e.g. 'Apple'")


class SearchWebsites(BaseModel):
    search_term: str = Field(..., description="Term or topic to search for on websites")


class FunctionCallingRAG(BaselineRAG):
    def __init__(
        self,
        *,
        storage_dir="./test_rag_vector_store",
        chunks: list[Chunk],
        ai_engine: BaseAIEngine,
        model_name: str,
        max_iterations: int = 10,
    ):
        super().__init__(storage_dir=storage_dir, chunks=chunks, ai_engine=ai_engine, model_name=model_name)
        self.max_iterations = max_iterations
        self.mock_api_client = MockAPIClient()

        self.tools = self._define_tools()
        self.tool_functions = self._define_tool_functions()

    def _define_tools(self) -> list[ChatCompletionToolParam]:
        return [
            pydantic_function_tool(
                SearchKnowledgeBase,
                name="search_knowledge_base",
                description="Search the knowledge base for relevant document chunks based on a query. Use this to find information from the knowledge base.",
            ),
            pydantic_function_tool(
                GetFinanceCompanyMetrics,
                name="get_finance_company_metrics",
                description="Get comprehensive financial metrics for a company including revenue, market_cap, pe_ratio, etc.",
            ),
            pydantic_function_tool(
                GetFinanceMetricByDate,
                name="get_finance_metric_by_date",
                description="Get a specific financial metric for a company on a specific date.",
            ),
            pydantic_function_tool(
                GetFinanceTimeseries,
                name="get_finance_timeseries",
                description="Get historical timeseries data for a specific financial metric.",
            ),
            pydantic_function_tool(
                GetFilingsByType,
                name="get_filings_by_type",
                description="Get SEC filings by type (10-K, 10-Q, etc.) and optionally by company.",
            ),
            pydantic_function_tool(
                SearchFilings,
                name="search_filings",
                description="Search for SEC filings related to a specific company. Optionally filter by filing type.",
            ),
            pydantic_function_tool(
                GetGPGStatistics,
                name="get_gpg_statistics",
                description="Get gender pay gap statistics for a specific employer/company.",
            ),
            pydantic_function_tool(
                SearchWebsites,
                name="search_websites",
                description="Search websites for information about a specific term or topic.",
            ),
        ]

    def _define_tool_functions(self) -> dict[str, Any]:
        return {
            "search_knowledge_base": self._execute_vector_search,
            "get_finance_company_metrics": self._execute_finance_company_metrics,
            "get_finance_metric_by_date": self._execute_finance_metric_by_date,
            "get_finance_timeseries": self._execute_finance_timeseries,
            "get_filings_by_type": self._execute_filings_by_type,
            "search_filings": self._execute_search_filings,
            "get_gpg_statistics": self._execute_gpg_statistics,
            "search_websites": self._execute_search_websites,
        }

    @override
    def retrieve(self, query: str, top_k: int) -> tuple[list[Chunk], list[ExternalChunk]]:
        logger.info(f"Starting function calling RAG retrieval for query: {query}")

        # Initialize thread
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]

        retrieved_chunks = []
        external_chunks = []

        # Agent reasoning loop
        for iteration in range(self.max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            completion = self.openai_client.chat.completions.create(
                model=self.model_name, messages=messages, tools=self.tools, tool_choice="auto"
            )

            assistant_message = completion.choices[0].message
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                logger.info("Agent finished reasoning")
                break

            # Run tool calls
            for tool_call in assistant_message.tool_calls:
                logger.info(f"Executing tool: {tool_call.function.name}")

                try:
                    result = self._execute_tool_call(tool_call, retrieved_chunks, external_chunks)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})

                except Exception as e:
                    logger.error(f"Error executing tool {tool_call.function.name}: {e}")
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Error: {str(e)}"})

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks and {len(external_chunks)} external chunks")

        # Return top_k results
        return retrieved_chunks, external_chunks

    def _execute_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
        retrieved_chunks: list[Chunk],
        external_chunks: list[ExternalChunk],
    ) -> str:
        """Execute a single tool call and return the result"""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "search_knowledge_base":
            return self._execute_vector_search(arguments, retrieved_chunks)
        elif function_name == "get_finance_company_metrics":
            return self._execute_finance_company_metrics(arguments, external_chunks)
        elif function_name == "get_finance_metric_by_date":
            return self._execute_finance_metric_by_date(arguments, external_chunks)
        elif function_name == "get_finance_timeseries":
            return self._execute_finance_timeseries(arguments, external_chunks)
        elif function_name == "get_filings_by_type":
            return self._execute_filings_by_type(arguments, external_chunks)
        elif function_name == "search_filings":
            return self._execute_search_filings(arguments, external_chunks)
        elif function_name == "get_gpg_statistics":
            return self._execute_gpg_statistics(arguments, external_chunks)
        elif function_name == "search_websites":
            return self._execute_search_websites(arguments, external_chunks)
        else:
            raise ValueError(f"Unknown function: {function_name}")

    def _execute_vector_search(self, arguments: dict[str, Any], retrieved_chunks: list[Chunk]) -> str:
        query = arguments["query"]
        top_k = arguments["top_k"]

        logger.info(f"Searching vector database for: {query} with top_k={top_k}")

        if not self.index:
            return "Vector database not available"

        query_vector = np.array(self.embed_query(query), dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        _, indices = self.index.search(query_vector, top_k)

        chunks: list[Chunk] = [self.chunks[i] for i in indices[0]]
        retrieved_chunks.extend(chunks)

        # Format results for the model
        formatted_results = []
        for i, chunk in enumerate(chunks):
            formatted_results.append(
                f"[Rank {i + 1}] {chunk.document_name} (Page {chunk.document_page}):\n{chunk.chunk_text}"
            )

        result_text = f"Found {len(chunks)} relevant chunks:\n\n" + "\n\n---\n\n".join(formatted_results)

        logger.info(f"Vector search returned {len(chunks)} chunks")
        return result_text

    def _execute_finance_company_metrics(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        """Execute finance company metrics API call"""
        ticker = arguments["ticker"]

        try:
            result = self.mock_api_client.get_finance_company_metrics(ticker)

            external_chunk = ExternalChunk(
                content=f"Company Metrics for {ticker}: {result.model_dump()}", source="finance_api"
            )
            external_chunks.append(external_chunk)

            return f"Retrieved financial metrics for {ticker}: {result.model_dump()}"

        except MockAPIError as e:
            logger.error(f"Finance API call failed for {ticker}: {e}")
            return f"Failed to get financial data for {ticker}: {str(e)}"

    def _execute_finance_metric_by_date(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        ticker = arguments["ticker"]
        metric = arguments["metric"]
        date = arguments["date"]

        try:
            result = self.mock_api_client.get_finance_metric_by_date(ticker, metric.lower(), date)

            external_chunk = ExternalChunk(
                content=f"{ticker} {metric} on {date}: {result.model_dump()}", source="finance_api"
            )
            external_chunks.append(external_chunk)

            return f"Retrieved {metric} for {ticker} on {date}: {result.model_dump()}"

        except MockAPIError as e:
            logger.error(f"Finance metric API call failed: {e}")
            return f"Failed to get {metric} for {ticker} on {date}: {str(e)}"

    def _execute_finance_timeseries(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        ticker = arguments["ticker"]
        metric = arguments["metric"]

        try:
            result = self.mock_api_client.get_finance_timeseries(ticker, metric.lower())

            if result.results:
                external_chunk = ExternalChunk(
                    content=f"{ticker} {metric} timeseries: {result.model_dump()}", source="finance_api"
                )
                external_chunks.append(external_chunk)

                return f"Retrieved {metric} timeseries for {ticker}: {result.model_dump()}"
            else:
                return f"No timeseries data found for {ticker} {metric}"

        except MockAPIError as e:
            logger.error(f"Finance timeseries API call failed: {e}")
            return f"Failed to get {metric} timeseries for {ticker}: {str(e)}"

    def _execute_filings_by_type(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        filing_type = arguments["filing_type"]
        company = arguments.get("company")

        try:
            if company:
                result = self.mock_api_client.get_filings_by_type(filing_type, company)
                content = f"Found {filing_type} filings for {company}: {result.get('filings', [])}"
            else:
                result = self.mock_api_client.get_filings_by_type(filing_type)
                content = f"Found {filing_type} filings: {result.get('filings', [])}"

            if result.get("filings"):
                external_chunk = ExternalChunk(content=content, source="filings_type_api")
                external_chunks.append(external_chunk)

            return content

        except MockAPIError as e:
            logger.error(f"Filings API call failed: {e}")
            return f"Failed to get {filing_type} filings: {str(e)}"

    def _execute_search_filings(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        company = arguments["company"]

        try:
            result = self.mock_api_client.search_filings(company=company)

            if result.get("results"):
                external_chunk = ExternalChunk(content=f"Filings for {company}: {result}", source="filings_search_api")
                external_chunks.append(external_chunk)

                return f"Found filings for {company}: {result}"
            else:
                return f"No filings found for {company}"

        except MockAPIError as e:
            logger.error(f"Filings search API call failed: {e}")
            return f"Failed to search filings for {company}: {str(e)}"

    def _execute_gpg_statistics(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        employer = arguments["employer"]

        try:
            result = self.mock_api_client.get_gpg_statistics(employer)

            if result.statistics:
                external_chunk = ExternalChunk(
                    content=f"GPG statistics for {employer}: {result.model_dump()}", source="gpg_statistics_api"
                )
                external_chunks.append(external_chunk)

                return f"Retrieved GPG statistics for {employer}: {result.model_dump()}"
            else:
                return f"No GPG statistics found for {employer}"

        except MockAPIError as e:
            logger.error(f"GPG statistics API call failed: {e}")
            return f"Failed to get GPG statistics for {employer}: {str(e)}"

    def _execute_search_websites(self, arguments: dict[str, Any], external_chunks: list[ExternalChunk]) -> str:
        search_term = arguments["search_term"]

        try:
            result = self.mock_api_client.search_websites(search_term)

            if result.results:
                external_chunk = ExternalChunk(content=f"Website search: {result.model_dump()}", source="search_api")
                external_chunks.append(external_chunk)

                return f"Website search results for '{search_term}': {result.model_dump()}"
            else:
                return f"No website results found for '{search_term}'"

        except MockAPIError as e:
            logger.error(f"Website search API call failed: {e}")
            return f"Failed to search websites for '{search_term}': {str(e)}"

    def _build_system_prompt(self) -> str:
        return textwrap.dedent("""
        You are an expert question answering assistant with access to both a knowledge base and real-time APIs.

        Your role is to help find relevant information by:
        1. First searching the static knowledge base with the "search_knowledge_base" tool for relevant information
        2. Determining if additional real-time data is needed from APIs based on the question
        3. Making appropriate API calls to supplement (not replace) your knowledge base search
        4. You can run multiple searches with different queries to gather comprehensive information
        5. If the question requires data about multiple companies or topics, you can run multiple knowledge base searches with different queries

        TOOL SELECTION GUIDELINES:

        **Always start with:**
        - search_knowledge_base: Search for any relevant background information, definitions, or context from the knowledge base
        - You can call this multiple times with different search terms to gather comprehensive information

        **Finance APIs - Use to supplement knowledge base with simple metrics:**
        - get_finance_company_metrics: Basic financial metrics only (EPS, P/E ratio, dividend, market cap) for major companies
        - get_finance_metric_by_date: Historical EPS data for specific dates (limited data available). Use 'yyyy-12-31' for annual EPS.
        - get_finance_timeseries: Historical EPS trends only (not comprehensive financial data)

        **Filing/Document APIs - Use when query mentions:**
        - get_filings_by_type: Specific filing types (10-K, 10-Q, 8-K, DEF 14A), annual reports, quarterly reports
        - search_filings: Company filings, regulatory documents, compliance submissions

        **Statistics APIs - Use when query mentions:**
        - get_gpg_statistics: Gender pay gap, salary data, pay statistics, employment demographics

        **Web Search - Use when query mentions:**
        - search_websites: News, recent developments, current information, industry updates, external sources

        **Important Notes:**
        - The knowledge base is your primary source - APIs provide limited supplementary data
        - Finance APIs only have basic metrics (EPS, P/E, dividend, market cap) - not comprehensive financial analysis
        - Always search the knowledge base first and multiple times with different queries if needed
        - Use APIs to add specific data points, not as primary information sources
        - If you need detailed financial analysis, rely on the knowledge base, not the simple API metrics

        **Decision Logic Examples:**
        - "Apple's financial performance" → multiple search_knowledge_base calls + get_finance_company_metrics for basic metrics
        - "Apple's business strategy" → multiple search_knowledge_base calls (APIs won't have this detail)
        - "Apple's EPS last quarter" → search_knowledge_base + get_finance_metric_by_date if available
        """)
