"""Pytest configuration and shared fixtures.

This file contains pytest fixtures that are shared across multiple test modules.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_repo():
    """Create a temporary directory for testing repository operations.

    Yields:
        Path: Path to temporary directory

    Cleanup:
        Removes the temporary directory after test completes
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_python_code():
    """Provide sample Python code for parser testing.

    Returns:
        str: Valid Python code with functions, classes, and imports
    """
    return '''
import os
from typing import List

def helper_function(x: int) -> int:
    """A simple helper function."""
    return x * 2

class SampleClass:
    """A sample class for testing."""

    def __init__(self, value: int):
        self.value = value

    def process(self) -> int:
        """Process the value."""
        return helper_function(self.value)
'''


# Additional fixtures will be added as subsystems are implemented
