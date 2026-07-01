import unittest
# Removed unittest.mock imports as we are doing an integration test now
# from unittest.mock import patch, MagicMock
import sys
import os
import logging # Added logging

# --- Start: Add project root to sys.path ---
# Adjust the number of os.path.dirname calls based on the location of test_tools.py
# Assuming tests/test_tools.py is in the project root's 'tests' directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End: Add project root to sys.path ---

# Now import the function to test and related schemas/exceptions
from research_system.tools import (
    _gemini_google_search_and_parse_internal,
    _validate_public_http_url,
    create_agent_tools,
)
# Import config to ensure environment variables are loaded (if using dotenv in config)
from research_system import config
from research_system.schemas import GeminiParsedOutput, ErrorResponse
# from langchain_core.exceptions import OutputParserException # Likely not needed for integration test assertions
# from google.genai import types as genai_types # No longer mocking the API call

# Configure basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestNetworkToolGuards(unittest.TestCase):
    """Unit tests for network tool safety checks."""

    def test_validate_public_http_url_allows_public_https(self):
        self.assertEqual(
            _validate_public_http_url("https://example.com/research"),
            "https://example.com/research",
        )

    def test_validate_public_http_url_rejects_localhost(self):
        with self.assertRaises(ValueError):
            _validate_public_http_url("http://localhost:8000/admin")

    def test_validate_public_http_url_rejects_loopback_ip(self):
        with self.assertRaises(ValueError):
            _validate_public_http_url("http://127.0.0.1:8000/admin")

    def test_validate_public_http_url_rejects_private_ip(self):
        with self.assertRaises(ValueError):
            _validate_public_http_url("http://192.168.1.10/status")

    def test_validate_public_http_url_rejects_metadata_ip(self):
        with self.assertRaises(ValueError):
            _validate_public_http_url("http://169.254.169.254/latest/meta-data/")

    def test_validate_public_http_url_rejects_non_http_scheme(self):
        with self.assertRaises(ValueError):
            _validate_public_http_url("file:///etc/passwd")

    def test_firecrawl_can_be_removed_from_tool_registry(self):
        tools = create_agent_tools({"allow_firecrawl_scrape": False})
        tool_names = {tool.name for tool in tools}

        self.assertNotIn("firecrawl_scrape_tool", tool_names)
        self.assertIn("tavily_search", tool_names)

@unittest.skipIf(not config.GEMINI_API_KEY or not config.AZURE_OPENAI_API_KEY_ALT, "API keys for Gemini and Azure Parser LLM must be set for integration test")
class TestGeminiSearchToolIntegration(unittest.TestCase):
    """Integration tests for the Gemini Google Search tool logic."""

    # Removed @patch decorators
    def test_real_gemini_search_and_parse(self):
        """Tests the end-to-end Gemini search and parsing with real API calls."""
        logger.info("\nRunning integration test: test_real_gemini_search_and_parse")
        print("\nRunning integration test: test_real_gemini_search_and_parse (requires valid API keys and network access)")

        # --- Mock Configuration REMOVED --- 
        # We are now using the actual config and functions

        # --- Execute the function --- 
        # Use a simple, unambiguous query for testing
        query = "how many people live in the world in 1899?"
        print(f"Executing with query: '{query}'")
        try:
            result = _gemini_google_search_and_parse_internal(query)
            print(f"Result: {result}")

            # --- Assertions (Adjusted for integration test) ---
            self.assertIsNotNone(result, "Result should not be None")
            
            # Check if an error occurred first
            if "error" in result:
                logger.error(f"Test failed due to error in function execution: {result['error']}")
                # Optionally fail harder or log more details
                self.fail(f"_gemini_google_search_and_parse_internal returned an error: {result}")

            # If no error, check for expected structure and types
            self.assertIn("summary", result, "Result should contain a 'summary' key")
            self.assertIsInstance(result.get("summary"), str, "Summary should be a string")
            self.assertTrue(result.get("summary"), "Summary should not be empty") # Check if summary is non-empty

            self.assertIn("key_facts", result, "Result should contain a 'key_facts' key")
            self.assertIsInstance(result.get("key_facts"), list, "Key facts should be a list")
            # We don't know how many facts, but the list should exist

            self.assertIn("sources", result, "Result should contain a 'sources' key")
            self.assertIsInstance(result.get("sources"), list, "Sources should be a list")
            # Check if sources list contains valid-looking URLs (if any)
            if result.get("sources"):
                for source in result.get("sources"):
                    self.assertTrue(source.startswith(('http://', 'https://')), f"Source '{source}' should be a valid URL")

            self.assertEqual(result.get("source_tool"), "GeminiGoogleSearch", "Source tool should be correctly identified")
            logger.info("Integration test passed.")

        except Exception as e:
             # Catch exceptions happening *outside* the tested function during the call
             logger.error(f"Exception during test execution: {e}", exc_info=True)
             self.fail(f"Test execution raised an unexpected exception: {e}")


    # --- Mocked test cases are commented out as they don't apply to integration testing --- 
    # @patch('research_system.tools.parse_gemini_output_with_llm')
    # @patch('research_system.config.get_gemini_google_search_llm')
    # def test_gemini_search_parsing_failure(self, mock_get_llm, mock_parse_output):
    #     """Tests failure during the parsing stage."""
    #     # ... (mocked implementation) ...
    #     pass 

    # @patch('research_system.config.get_gemini_google_search_llm')
    # def test_gemini_search_llm_unavailable(self, mock_get_llm):
    #     """Tests scenario where the Gemini LLM configuration returns None."""
    #     # ... (mocked implementation) ...
    #     pass

    # @patch('research_system.tools.parse_gemini_output_with_llm')
    # @patch('research_system.config.get_gemini_google_search_llm')
    # def test_gemini_search_api_error(self, mock_get_llm, mock_parse_output):
    #     """Tests an exception during the Gemini API call."""
    #     # ... (mocked implementation) ...
    #     pass

    # @patch('research_system.tools.parse_gemini_output_with_llm')
    # @patch('research_system.config.get_gemini_google_search_llm')
    # def test_gemini_search_empty_response(self, mock_get_llm, mock_parse_output):
    #     """Tests the case where Gemini returns a response object but it has no text."""
    #     # ... (mocked implementation) ...
    #     pass

if __name__ == '__main__':
    unittest.main()
