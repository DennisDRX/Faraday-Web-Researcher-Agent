"""
This module contains wrapper functions for various search and information gathering tools
used by the Web Research Agent system, refactored as Langchain Tools.
"""

import logging
import json
import ipaddress
import os
import socket
from typing import List, Dict, Any, Optional, Type
from urllib.parse import urlparse
import requests
from . import config

from .schemas import ResearchReport, Source, IntermediateStep, ResearchRequest, ErrorResponse, GeminiParsedOutput
from .prompts import QUERY_DECOMPOSITION_TEMPLATE

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import tool, BaseTool
from .config import get_primary_llm, get_azure_openai_parser_llm
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikidata import WikidataAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from newsapi import NewsApiClient

# Add google-genai imports
import google.generativeai as genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_TOOL_QUERY_LENGTH = int(os.getenv("MAX_TOOL_QUERY_LENGTH", "500"))
ALLOW_FIRECRAWL_SCRAPE = os.getenv("ALLOW_FIRECRAWL_SCRAPE", "true").lower() in {"1", "true", "yes", "on"}
ALLOW_PRIVATE_SCRAPE_TARGETS = os.getenv("ALLOW_PRIVATE_SCRAPE_TARGETS", "false").lower() in {"1", "true", "yes", "on"}


def _clean_tool_query(query: str, *, label: str = "query") -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{label} must be a non-empty string.")

    cleaned = " ".join(query.split())
    if len(cleaned) > MAX_TOOL_QUERY_LENGTH:
        raise ValueError(f"{label} exceeds {MAX_TOOL_QUERY_LENGTH} characters.")

    return cleaned


def _is_blocked_ip(ip) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_public_http_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("URL must be a non-empty string.")

    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs can be scraped.")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed.")

    hostname = parsed.hostname.strip().rstrip(".")
    try:
        ip = ipaddress.ip_address(hostname)
        if not ALLOW_PRIVATE_SCRAPE_TARGETS and _is_blocked_ip(ip):
            raise ValueError("Private, local, reserved, and link-local scrape targets are not allowed.")
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise
        try:
            resolved = socket.getaddrinfo(hostname, parsed.port or None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as dns_error:
            raise ValueError(f"Unable to resolve URL hostname: {hostname}") from dns_error

        for result in resolved:
            address = result[4][0]
            ip = ipaddress.ip_address(address)
            if not ALLOW_PRIVATE_SCRAPE_TARGETS and _is_blocked_ip(ip):
                raise ValueError("Private, local, reserved, and link-local scrape targets are not allowed.")

    return raw_url.strip()

# --- Internal Helper Functions ---
def parse_gemini_output_with_llm(gemini_raw_text: str, query: str) -> Optional[Dict[str, Any]]:
    """
    Uses the secondary Azure OpenAI LLM to parse the raw text output from the
    Gemini call into a structured format, focusing on summary and key facts.

    Args:
        gemini_raw_text: The raw *text string* output from Gemini.
        query: The original research query, provided for context.

    Returns:
        A dictionary containing 'summary' and 'key_facts', or None if parsing fails.
        Source extraction is handled separately using grounding metadata when available.
    """
    parser_llm = get_azure_openai_parser_llm()
    if not parser_llm:
        logger.error("Azure OpenAI Parser LLM is not available for Gemini parsing.")
        return None

    # Define a simpler Pydantic model or just use a dict structure
    class GeminiParsedText(BaseModel):
        summary: str = Field(description="A concise summary of the findings regarding the query, based *only* on the provided text.")
        key_facts: List[str] = Field(description="A list of key facts or pieces of information presented in the text.")

    parser = JsonOutputParser(pydantic_object=GeminiParsedText)

    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert assistant tasked with parsing the raw text output of a Gemini model.
        The Gemini model was asked to investigate the following research query: '{query}'
        Its raw text output is provided below.
        Your goal is to extract the key information and structure it into a JSON object matching the requested format.

        Focus *only* on identifying from the provided text:
        1.  A concise summary of the findings regarding the query.
        2.  A list of key facts or pieces of information presented.
        Do NOT attempt to extract source URLs from this text.

        Raw Gemini Text Output:
        ---
        {gemini_raw_text}
        ---

        Format Instructions:
        {format_instructions}
        """
    )

    chain = prompt_template | parser_llm | parser
    logger.info("Attempting to parse Gemini text output (summary/facts) using Azure OpenAI Parser LLM.")

    try:
        # Invoke chain to get summary and key facts
        parsed_text_dict = chain.invoke({
            "query": query,
            "gemini_raw_text": gemini_raw_text,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Successfully parsed Gemini text output (summary/facts).")
        # Return only the summary and key facts
        return {"summary": parsed_text_dict.get("summary", ""), "key_facts": parsed_text_dict.get("key_facts", [])}
    except OutputParserException as ope:
         logger.error(f"Failed to parse Gemini text output. Parser Error: {ope}. Raw output was:\n{gemini_raw_text[:500]}...")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gemini text output parsing: {e}")
        return None

# --- Tool Input Schemas ---

class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")

class WikidataInput(BaseModel):
    query: str = Field(description="A specific entity name (e.g., 'Eiffel Tower') or concept to query Wikidata for structured data.")

class FirecrawlInput(BaseModel):
    url: str = Field(description="A single, specific URL to scrape for its main content in Markdown format.")

class NewsSearchInput(BaseModel):
    query: str = Field(description="Keywords or phrase to search for in recent news articles.")
    language: str = Field(default='en', description="The 2-letter ISO 639-1 code of the language.")
    page_size: int = Field(default=10, description="Number of news results (max 100).")


# --- Query Decomposition Tool ---
class QueryDecompositionInput(BaseModel):
    query: str = Field(..., description="The complex research query to decompose into sub-topics or questions.")

class QueryDecompositionOutput(BaseModel):
    sub_queries: List[str] = Field(..., description="A list of simpler, searchable sub-queries or sub-topics derived from the original query.")


class QueryDecompositionTool(BaseTool):
    """Tool to decompose complex research queries."""
    name: str = "query_decomposition_tool"
    description: str = (
        "Decomposes a complex research query (containing multiple facets, entities, or questions) into a list of simpler, "
        "focused sub-queries or sub-topics. Use this *first* on complex queries to guide the research strategy and generate targeted searches for other tools."
    )
    args_schema: Type[BaseModel] = QueryDecompositionInput

    def _run(self, query: str) -> Dict[str, Any]:
        """Executes the query decomposition."""
        logger.info(f"Executing Query Decomposition Tool for query: '{query}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for Query Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        from langchain_core.output_parsers import StrOutputParser
        import ast
        prompt = QUERY_DECOMPOSITION_TEMPLATE
        chain = prompt | llm | StrOutputParser()

        try:
            result_str = chain.invoke({"query": query})
            try:
                sub_queries_list = ast.literal_eval(result_str.strip())
                if not isinstance(sub_queries_list, list):
                    raise ValueError("Parsed result is not a list")
                result = {"sub_queries": sub_queries_list}
                logger.info(f"Query decomposition successful. Found {len(result.get('sub_queries', []))} sub-queries.")
            except (SyntaxError, ValueError, TypeError) as parse_err:
                logger.error(f"Failed to parse sub-query list string: {parse_err}. Raw result: {result_str}")
                result = {"sub_queries": [result_str]}
            return result
        except Exception as e:
            logger.error(f"Error during query decomposition for '{query}': {e}", exc_info=True)
            return {"error": f"Query decomposition failed: {str(e)}"}

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Async execution."""
        logger.info(f"Executing async Query Decomposition Tool for query: '{query}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for async Query Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        from langchain_core.output_parsers import StrOutputParser
        import ast
        prompt = QUERY_DECOMPOSITION_TEMPLATE
        chain = prompt | llm | StrOutputParser()

        try:
            result_str = await chain.ainvoke({"query": query})
            try:
                sub_queries_list = ast.literal_eval(result_str.strip())
                if not isinstance(sub_queries_list, list):
                    raise ValueError("Parsed result is not a list")
                result = {"sub_queries": sub_queries_list}
                logger.info(f"Async query decomposition successful. Found {len(result.get('sub_queries', []))} sub-queries.")
            except (SyntaxError, ValueError, TypeError) as parse_err:
                logger.error(f"Failed to parse async sub-query list string: {parse_err}. Raw result: {result_str}")
                result = {"sub_queries": [result_str]}
            return result
        except Exception as e:
            logger.error(f"Error during async query decomposition for '{query}': {e}", exc_info=True)
            return {"error": f"Async query decomposition failed: {str(e)}"}

# --- Search & Retrieval Tools ---

@tool(args_schema=SearchInput)
def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Performs a comprehensive web search using Tavily. Excellent for broad exploration or finding diverse sources on a research query or sub-topic.
    Returns a list of relevant documents with URLs, titles, and content snippets.
    Good starting point for most research tasks.
    """
    tavily_client = config.get_tavily_client()
    if not tavily_client:
        logger.error("Tavily client is not available.")
        return {"error": "Tavily client unavailable."}
    try:
        query = _clean_tool_query(query)
        max_results = max(1, min(int(max_results), 10))
        logger.info(f"Performing Tavily search for query: '{query}'")
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        if isinstance(response, list):
            response_dict = {"results": response}
        else:
            response_dict = response
        logger.info(f"Tavily search completed. Found {len(response_dict.get('results', []))} results.")
        if 'results' in response_dict:
             for result in response_dict['results']:
                 result['source_tool'] = 'tavily_search'
        return response_dict
    except Exception as e:
        logger.error(f"Error during Tavily search for query '{query}': {e}")
        return {"error": f"Tavily search failed: {e}"}

@tool(args_schema=FirecrawlInput)
def firecrawl_scrape_tool(url: str) -> Dict[str, Any]:
    """Retrieves the main text content of a single web page URL in Markdown format using Firecrawl.
    Use this when a search result snippet (from Tavily, etc.) is insufficient, and you need the full text content of a specific, highly relevant page.
    Provides clean Markdown suitable for analysis.
    """
    logger.info(f"Attempting to scrape URL with Firecrawl: {url}")
    if not ALLOW_FIRECRAWL_SCRAPE:
        logger.warning("Firecrawl scraping is disabled by configuration.")
        return {"url": url, "error": "Firecrawl scraping is disabled by configuration."}

    try:
        url = _validate_public_http_url(url)
        firecrawl_client = config.get_firecrawl_client()
        if not firecrawl_client:
            logger.error("Firecrawl client is not available.")
            return {"error": "Firecrawl client unavailable."}

        # Firecrawl returns a list of pages, even for single URLs. Take the first result.
        result = firecrawl_client.scrape_url(url)
        if result and isinstance(result, list) and result[0]:
             scraped_data = result[0]
             # Add source_tool field
             scraped_data['source_tool'] = 'firecrawl_scrape_tool'
             logger.info(f"Successfully scraped {url}")
             return scraped_data # Return the dict
        else:
             logger.warning(f"Firecrawl scrape returned empty result for {url}")
             return {"url": url, "markdown_content": "", "metadata": {}, "source_tool": 'firecrawl_scrape_tool', "error": "Empty scrape result"}
    except Exception as e:
        logger.error(f"Error during Firecrawl scrape for URL {url}: {e}")
        return {"url": url, "markdown_content": "", "metadata": {}, "source_tool": 'firecrawl_scrape_tool', "error": f"Firecrawl scrape failed: {e}"}

@tool(args_schema=NewsSearchInput)
def news_search(query: str, language: str = 'en', page_size: int = 10) -> Dict[str, Any]:
    """
    Searches *recent news articles* (past ~30 days) related to a query using NewsAPI.
    Crucial for time-sensitive claims, recent events, or verifying information reported in the news.
    Returns articles with titles, sources, URLs, snippets, and publication dates.
    """
    news_client = config.get_news_api_client()  # Corrected function call
    if not news_client:
        logger.error("NewsAPI client is not available.")
        return {"error": "NewsAPI client unavailable."}
    try:
        query = _clean_tool_query(query)
        page_size = max(1, min(int(page_size), 25))
        logger.info(f"Performing NewsAPI search for query: '{query}', lang: {language}, size: {page_size}")
        # Use get_everything for broader search, or top_headlines for major news
        response = news_client.get_everything(
            q=query,
            language=language,
            page_size=min(page_size, 100), # Ensure page_size doesn't exceed NewsAPI limit
            sort_by='relevancy' # 'publishedAt' or 'popularity' also options
        )
        logger.info(f"NewsAPI search completed. Status: {response.get('status')}, Found {response.get('totalResults', 0)} results.")
        # Process articles if the request was successful and articles exist
        if response.get('status') == 'ok' and 'articles' in response:
             processed_articles = []
             for article in response.get('articles', []):
                 # Standardize snippet, title, url and add source tool
                 processed_article = {
                     'title': article.get('title'),
                     'url': article.get('url'),
                     'snippet': article.get('description') or article.get('content'), # Use description or content
                     'source': article.get('source', {}).get('name'), # Get source name
                     'publishedAt': article.get('publishedAt'),
                     'source_tool': 'news_search' # Consistent tool naming
                 }
                 processed_articles.append(processed_article)
             # Return the standardized list within the expected structure
             return {"articles": processed_articles, "totalResults": response.get('totalResults')}
        elif response.get('status') == 'error':
             logger.error(f"NewsAPI returned an error: Code={response.get('code')}, Message={response.get('message')}")
             return {"error": f"NewsAPI error: {response.get('message', 'Unknown error')}", "articles": []}
        else:
             # Handle cases like status 'ok' but no articles, or unexpected structure
             logger.warning(f"NewsAPI search for '{query}' returned status '{response.get('status')}' but format might be unexpected or no articles found.")
             return {"articles": [], "message": response.get('message', "No articles found or unexpected response.")}
    except Exception as e:
        logger.error(f"Error during NewsAPI search for query '{query}': {e}", exc_info=True) # Add traceback
        return {"error": f"NewsAPI search failed: {e}", "articles": []}

@tool(args_schema=SearchInput)
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a simple web search using DuckDuckGo.
    Provides concise results quickly, good for finding definitions or quick facts.
    Returns a list of search results with titles, URLs, and snippets.
    """
    logger.info(f"Performing DuckDuckGo search for query: '{query}'")
    try:
        query = _clean_tool_query(query)
        max_results = max(1, min(int(max_results), 10))
        # DuckDuckGoSearchRun directly returns a list of dicts
        ddg_search = DuckDuckGoSearchAPIWrapper(region="us-en", max_results=max_results)
        results = ddg_search.results(query, max_results=max_results)
        # Add source_tool to each result
        for result in results:
            result['source_tool'] = 'duckduckgo_search'
        logger.info(f"DuckDuckGo search completed. Found {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return {"error": f"DuckDuckGo search failed: {e}"}

@tool(args_schema=WikidataInput)
def wikidata_entity_search(query: str) -> Dict[str, Any]:
    """
    Searches Wikidata for structured data about a specific entity.
    Useful for retrieving factual attributes (dates, relationships, properties) for well-known people, places, or concepts.
    Input should be the entity name (e.g., 'Albert Einstein', 'Eiffel Tower').
    """
    logger.info(f"Performing Wikidata search for entity: '{query}'")
    try:
        query = _clean_tool_query(query)
        wikidata_wrapper = WikidataAPIWrapper()
        tool = WikidataQueryRun(api_wrapper=wikidata_wrapper)
        # The run method returns a string, need to parse it potentially
        result_str = tool.run(query)
        # Simple approach: Wrap the string result in a dict with tool name
        logger.info(f"Wikidata search completed for '{query}'.")
        return {"query": query, "result": result_str, "source_tool": 'wikidata_entity_search'}
    except Exception as e:
        logger.error(f"Error during Wikidata search for entity '{query}': {e}")
        return {"error": f"Wikidata search failed: {e}", "query": query, "result": None, "source_tool": 'wikidata_entity_search'}

# --- Gemini Google Search Tool ---
# This tool wraps a specific capability (e.g., a Gemini model with Google Search enabled)
# It's expected to return a specific format that needs parsing.

class GeminiGoogleSearchInput(BaseModel):
    query: str = Field(description="The query or sub-topic to investigate using Google Search synthesized by Gemini.")

def _gemini_google_search_and_parse_internal(query: str) -> Dict[str, Any]:
    """
    Internal function to call Gemini with Google Search, parse its text output
    for summary/facts, and combine with grounding metadata sources.
    Returns parsed structured data or an error.
    """
    logger.info(f"Calling internal Gemini Google Search for query: '{query}'")
    
    gemini_result: Dict[str, Any] = {}
    parsed_text_data: Optional[Dict[str, Any]] = None
    final_output: Dict[str, Any] = {"query": query, "source_tool": "GeminiGoogleSearch"}

    try:
        query = _clean_tool_query(query)
        # --- Call the config.py function to perform the search --- 
        # Returns {"text": str, "source_urls": List[str], "error": Optional[str]}
        gemini_result = config.perform_gemini_google_search(query)

        # Check for errors from the config function
        if gemini_result.get("error"):
            logger.error(f"Gemini search failed: {gemini_result['error']}")
            final_output["error"] = f"Gemini search failed: {gemini_result['error']}"
            # Include raw text if available, even with error
            final_output["raw_output"] = gemini_result.get("text", "")[:1000] 
            return final_output
        
        raw_gemini_text = gemini_result.get("text", "")
        grounding_source_urls = gemini_result.get("source_urls", [])

        logger.info(f"Raw text received from Gemini Search function: {raw_gemini_text[:200]}...")
        if grounding_source_urls:
             logger.info(f"Received {len(grounding_source_urls)} source URLs from grounding metadata.")
        else:
             logger.info("No source URLs received from grounding metadata.")

        # --- Call the helper function to parse *only* the text for summary/facts ---
        if raw_gemini_text:
            parsed_text_data = parse_gemini_output_with_llm(raw_gemini_text, query)
        else:
            logger.warning("Gemini result contained no text to parse.")

        # --- Construct the final output dictionary ---
        if parsed_text_data:
            final_output["summary"] = parsed_text_data.get("summary", "")
            final_output["key_facts"] = parsed_text_data.get("key_facts", [])
            # Use grounding metadata URLs as the primary source list
            final_output["sources"] = grounding_source_urls 
            logger.info("Gemini search and parse completed successfully using grounding metadata sources.")
        elif grounding_source_urls: 
             # If text parsing failed but we have sources, return sources with a note
             final_output["summary"] = "Could not parse summary/facts from Gemini response."
             final_output["key_facts"] = []
             final_output["sources"] = grounding_source_urls
             final_output["error"] = "Failed to parse Gemini text output, but grounding sources are available."
             logger.warning("Failed to parse Gemini text, but returning grounding sources.")
        else:
            # If both text parsing failed AND no grounding sources were found
            logger.error("Failed to parse Gemini output and no grounding sources found.")
            final_output["error"] = "Failed to parse Gemini output and no grounding sources found."
            final_output["raw_output"] = raw_gemini_text[:1000] # Include raw text for context

        return final_output

    except Exception as e:
        # Catch exceptions occurring during this orchestration
        logger.error(f"Error during Gemini Google Search orchestration for query '{query}': {e}", exc_info=True)
        final_output["error"] = f"Gemini Google Search orchestration failed: {str(e)}"
        # Include raw text if available from gemini_result
        final_output["raw_output"] = gemini_result.get("text", "")[:1000] 
        return final_output


@tool(args_schema=GeminiGoogleSearchInput)
def gemini_google_search_tool(query: str) -> Dict[str, Any]:
    """
    Leverages a Google Search-enabled Gemini model to find and synthesize information for a query.
    Excellent for getting a quick, synthesized overview and relevant sources directly from the LLM.
    """
    return _gemini_google_search_and_parse_internal(query)

# --- FINISH Tool ---
# This is a special tool to signal the agent has completed its task.

class FinishSchema(BaseModel):
    reason: str = Field(default="Research complete. Sufficient information gathered.", description="A brief reason why the research is being concluded (e.g., 'sufficient evidence found', 'conflicting evidence found', 'scope covered').")

@tool(args_schema=FinishSchema)
def FINISH(reason: str) -> Dict[str, Any]:
    """
    Signals that the agent has completed its research task and is ready to return a final answer.
    Use this tool when you have gathered sufficient information to synthesize a comprehensive report.
    Provide a brief reason for finishing.
    """
    logger.info(f"FINISH tool called with reason: {reason}")
    # This tool doesn't perform an action that returns data for the agent loop
    # Its purpose is to route the graph to the final synthesis step.
    # We return a simple confirmation.
    return {"status": "FINISH signal received", "reason": reason}

# --- Tool Creation Factory ---
def create_agent_tools(cfg: Optional[Dict] = None) -> List[BaseTool]:
    """
    Creates and returns a list of initialized tools for the agent.
    Based on provided configuration.
    """
    logger.info("Creating agent tools...")
    cfg = cfg or {}
    enable_firecrawl = cfg.get("allow_firecrawl_scrape", ALLOW_FIRECRAWL_SCRAPE)

    tools: List[BaseTool] = [
        # Initialize standard tools
        # Remove FirecrawlInput as it's likely just a schema, not an executable tool
        # FirecrawlInput,
        tavily_search,
        news_search,
        duckduckgo_search,
        wikidata_entity_search,
        QueryDecompositionTool(), # Query Decomposition is a class
        gemini_google_search_tool,
        FINISH # Include the FINISH tool
    ]
    if enable_firecrawl:
        tools.insert(1, firecrawl_scrape_tool)
    logger.info(f"Created {len(tools)} tools.")
    return tools
