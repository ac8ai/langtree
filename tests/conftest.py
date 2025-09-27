"""
Shared test fixtures and utilities for the langtree test suite.
"""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider that returns dummy models for testing.

    This fixture patches the global _llm_provider to return mock LLM instances
    instead of trying to instantiate real models, preventing API calls during tests.

    Usage:
        def test_something(mock_llm_provider):
            # Test code that uses LLM models
            pass
    """
    mock_llm = Mock()
    mock_llm.get_graph.return_value = Mock()  # Mock LangChain graph

    with patch("langtree.chains._llm_provider") as mock_provider:
        mock_provider.get_llm.return_value = mock_llm
        mock_provider.list_models.return_value = ["reasoning", "gpt-4", "test-model"]
        yield mock_provider
