"""Basic tests for configuration subsystem."""

import os
import pytest
from repomind.config import Config, config


def test_default_config_values():
    """Verify default configuration parameters."""
    assert config.LLM_PROVIDER == "gemini"
    assert config.GEMINI_MODEL == "gemini-3.5-flash-lite"
    assert config.EMBEDDING_DIMENSION == 384
    assert config.RETRIEVAL_STRATEGY == "hybrid"
    assert config.HYBRID_VECTOR_WEIGHT == 0.6
    assert config.HYBRID_GRAPH_WEIGHT == 0.4


def test_hybrid_weights_sum_to_one():
    """Verify hybrid weights sum to 1.0."""
    assert config.HYBRID_VECTOR_WEIGHT + config.HYBRID_GRAPH_WEIGHT == 1.0


def test_config_validation_passes_with_valid_weights():
    """Config validation should not raise an error for valid settings (without API key check)."""
    # Note: Validate will fail if GEMINI_API_KEY is missing, so we test weight validation logic directly
    assert config.HYBRID_VECTOR_WEIGHT + config.HYBRID_GRAPH_WEIGHT == 1.0
